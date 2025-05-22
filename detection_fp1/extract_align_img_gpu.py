import cv2
import numpy as np
from tqdm import tqdm
import os
import multiprocessing
import torch
import torch.nn.functional as F

# 定义Flann参数常量
FLANN_INDEX_KDTREE = 1
SEARCH_WINDOW_SECONDS = 1  # 搜索窗口前后扩展秒数
REF_FRAMES_PER_SECOND = 3  # 每秒采样参考帧数

class VideoComparator:
    _window_cache = {}


    def __init__(self, ref_video_path, aligned_video_path):
        # 初始化视频源
        self.ref_cap = cv2.VideoCapture(ref_video_path)
        self.aligned_cap = cv2.VideoCapture(aligned_video_path)
        self.aligned_video_path = aligned_video_path
        
        # 验证视频参数
        self._validate_video_params()
        
        # 预加载参考视频多特征帧
        self.ref_frames = []
        total_ref_seconds = self.total_ref // self.fps
        for second in range(total_ref_seconds):
            frame_set = []
            # 在每秒内均匀采样多个参考帧
            for ratio in np.linspace(0.1, 0.9, REF_FRAMES_PER_SECOND):
                frame_idx = int((second + ratio) * self.fps)
                frame_idx = min(frame_idx, self.total_ref-1)
                self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.ref_cap.read()
                if ret:
                    if frame.shape[:2] != self.frame_size[::-1]:
                        frame = cv2.resize(frame, self.frame_size)
                    frame_set.append(frame)
                else:
                    frame_set.append(None)
            self.ref_frames.append(frame_set)
        
        # 特征检测参数
        self.index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        self.search_params = dict(checks=100)
        
        # 窗口缓存
        self._window_cache = {}
        
        # 结果存储
        self.similarity_data = {}

    def _validate_video_params(self):
        ref_fps = int(self.ref_cap.get(cv2.CAP_PROP_FPS))
        aligned_fps = int(self.aligned_cap.get(cv2.CAP_PROP_FPS))
        if ref_fps != aligned_fps:
            raise ValueError(f"帧率不匹配: 参考视频{ref_fps}fps vs 对齐视频{aligned_fps}fps")
        
        ref_width = int(self.ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ref_height = int(self.ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        aligned_width = int(self.aligned_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        aligned_height = int(self.aligned_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.frame_size = (ref_width, ref_height)
        self.fps = ref_fps
        self.total_ref = int(self.ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_aligned = int(self.aligned_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @staticmethod
    def _enhance_image(frame):
        """图像增强处理"""
        # 直方图均衡化
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        
        # 锐化处理
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(enhanced, -1, kernel)

    @staticmethod
    def compute_frame_similarity(ref_frame, target_frame):
        """GPU加速的SSIM相似度计算"""
        if ref_frame is None or target_frame is None:
            return 0.0
        if ref_frame.shape != target_frame.shape:
            return 0.0
        try:
            ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
            target_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
        except:
            return 0.0
        
        # 转换为PyTorch张量并归一化
        ref_tensor = torch.from_numpy(ref_gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        target_tensor = torch.from_numpy(target_gray).float().unsqueeze(0).unsqueeze(0) / 255.0
        
        # 移至GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ref_tensor = ref_tensor.to(device)
        target_tensor = target_tensor.to(device)
        
        # 计算SSIM
        ssim_val = VideoComparator._ssim(ref_tensor, target_tensor).item()
        
        # 释放显存
        del ref_tensor, target_tensor
        torch.cuda.empty_cache()
        
        return ssim_val

    @staticmethod
    def _ssim(x, y, window_size=11):
        """PyTorch实现SSIM计算"""
        # 获取或创建窗口
        device = x.device
        key = (window_size, device)
        
        if key not in VideoComparator._window_cache:
            window = torch.ones((1, 1, window_size, window_size), dtype=torch.float32, device=device)
            window /= window_size**2  # 均匀窗口
            VideoComparator._window_cache[key] = window
        
        window = VideoComparator._window_cache[key]
        
        mu_x = F.conv2d(x, window, padding=window_size//2)
        mu_y = F.conv2d(y, window, padding=window_size//2)
        
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(x*x, window, padding=window_size//2) - mu_x_sq
        sigma_y_sq = F.conv2d(y*y, window, padding=window_size//2) - mu_y_sq
        sigma_xy = F.conv2d(x*y, window, padding=window_size//2) - mu_xy
        
        C1 = (0.01 * 1)**2  # data_range=1
        C2 = (0.03 * 1)**2
        
        numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        ssim_map = numerator / (denominator + 1e-8)
        return ssim_map.mean()

    @staticmethod
    def _process_second(args):
        (second, ref_paths, aligned_video_path, method, 
         output_dir, frame_size, fps, total_aligned) = args
        
        # 初始化特征检测器
        detector = cv2.SIFT_create(nfeatures=2000)
        matcher = cv2.FlannBasedMatcher(
            dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
            dict(checks=100)
        )
        
        # 加载多个参考帧
        ref_frames = []
        for path in ref_paths:
            if path and os.path.exists(path):
                frame = cv2.imread(path)
                ref_frames.append(frame)
        
        # 计算目标视频搜索范围
        start_sec = max(0, second - SEARCH_WINDOW_SECONDS)
        end_sec = min(total_aligned // fps, second + SEARCH_WINDOW_SECONDS)
        target_start = start_sec * fps
        target_end = min((end_sec + 1) * fps - 1, total_aligned - 1)
        
        # 读取目标视频帧
        aligned_cap = cv2.VideoCapture(aligned_video_path)
        aligned_cap.set(cv2.CAP_PROP_POS_FRAMES, target_start)
        target_frames = []
        for _ in range(target_end - target_start + 1):
            ret, frame = aligned_cap.read()
            if ret:
                frame = cv2.resize(frame, frame_size) if frame.shape[:2] != frame_size[::-1] else frame
                target_frames.append(frame)
            else:
                target_frames.append(None)
        aligned_cap.release()
        
        max_similarity = 0.0
        best_frame = None
        best_idx = target_start
        
        for i, target_frame in enumerate(target_frames):
            if target_frame is None:
                continue
            current_idx = target_start + i
            
            # 多参考帧综合评分
            frame_similarity = 0.0
            for ref_frame in ref_frames:
                if ref_frame is None:
                    continue
                
                # 混合相似度计算
                if method == 'feature':
                    # 特征匹配相似度
                    gray_ref = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
                    gray_target = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
                    
                    kp_ref, des_ref = detector.detectAndCompute(gray_ref, None)
                    kp_target, des_target = detector.detectAndCompute(gray_target, None)
                    
                    if des_ref is None or des_target is None:
                        feat_sim = 0.0
                    else:
                        matches = matcher.knnMatch(des_ref, des_target, k=2)
                        good = [m for m, n in matches if m.distance < 0.6 * n.distance]
                        
                        # 几何验证
                        if len(good) > 4:
                            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good])
                            dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good])
                            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            inliers = np.sum(mask) if mask is not None else 0
                            feat_sim = (inliers + len(good)) / (2 * min(len(kp_ref), len(kp_target)))
                        else:
                            feat_sim = 0.0
                else:
                    feat_sim = 0.0
                
                # 使用GPU计算结构相似度
                ssim_score = VideoComparator.compute_frame_similarity(ref_frame, target_frame)
                
                # 综合评分（可调节权重）
                combined_score = 0.7 * feat_sim + 0.3 * ssim_score
                frame_similarity = max(frame_similarity, combined_score)
            
            if frame_similarity > max_similarity:
                max_similarity = frame_similarity
                best_frame = target_frame
                best_idx = current_idx
        
        # 保存结果
        if best_frame is not None and max_similarity > 0.01:
            os.makedirs(output_dir, exist_ok=True)
            comp_path = os.path.join(output_dir, f"sec{second}_comp.jpg")
            ref_path = os.path.join(output_dir, f"sec{second}_ref.jpg")
            target_path = os.path.join(output_dir, f"sec{second}_target.jpg")
            
            # 保存最佳参考帧
            best_ref = ref_frames[np.argmax([
                VideoComparator.compute_frame_similarity(r, best_frame) 
                for r in ref_frames if r is not None
            ])]
            cv2.imwrite(ref_path, best_ref)
            cv2.imwrite(target_path, best_frame)
            cv2.imwrite(comp_path, np.hstack((best_ref, best_frame)))
            
            return (second, max_similarity, ref_path, target_path, 
                    best_idx - (second * fps))
        return (second, 0.0, None, None, 0)

    def compare_seconds(self, method='feature', output_dir='enhanced_output'):
        os.makedirs(output_dir, exist_ok=True)
        total_seconds = len(self.ref_frames)
        
        # 保存多参考帧
        ref_frame_dir = os.path.join(output_dir, "multi_ref_frames")
        os.makedirs(ref_frame_dir, exist_ok=True)
        ref_paths = []
        for idx, frame_set in enumerate(self.ref_frames):
            sec_paths = []
            for f_idx, frame in enumerate(frame_set):
                if frame is not None:
                    path = os.path.join(ref_frame_dir, f"ref_{idx}_{f_idx}.jpg")
                    cv2.imwrite(path, frame)
                    sec_paths.append(path)
                else:
                    sec_paths.append(None)
            ref_paths.append(sec_paths)
        
        # 准备并行参数
        args_list = []
        for second in range(total_seconds):
            args = (
                second,
                ref_paths[second],
                self.aligned_video_path,
                method,
                output_dir,
                self.frame_size,
                self.fps,
                self.total_aligned
            )
            args_list.append(args)
        
        # 并行处理
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = list(tqdm(pool.imap(self._process_second, args_list), 
                             total=len(args_list)))
        
        # 整合结果
        for result in results:
            second, similarity, ref_path, target_path, offset = result
            if similarity > 0:
                self.similarity_data[second] = (similarity, ref_path, target_path, offset)

    def generate_report(self, output_file='enhanced_report.csv'):
        with open(output_file, 'w') as f:
            f.write("Second,Similarity,RefPath,TargetPath,FrameOffset\n")
            for sec in sorted(self.similarity_data.keys()):
                data = self.similarity_data[sec]
                f.write(f"{sec},{data[0]:.4f},{data[1]},{data[2]},{data[3]}\n")

# 使用示例
if __name__ == "__main__":
    comparator = VideoComparator(
        ref_video_path="Trainingvideos_3.mp4",
        aligned_video_path="compare_Trainingvideos_3_vs_Test-3/aligned_Test-3.mp4"       
    )
    comparator.compare_seconds(method='feature', output_dir='enhanced_output')
    comparator.generate_report('enhanced_report.csv')