import cv2
import numpy as np
from tqdm import tqdm
import os
import multiprocessing
from skimage.metrics import structural_similarity as compare_ssim

# 定义Flann参数常量
FLANN_INDEX_KDTREE = 1
SEARCH_WINDOW_SECONDS = 1  # 搜索窗口前后扩展秒数
REF_FRAMES_PER_SECOND = 3  # 每秒采样参考帧数

class VideoComparator:
    def __init__(self, ref_video_path, aligned_video_path):
        # 初始化视频源
        self.ref_video_path = ref_video_path
        self.aligned_video_path = aligned_video_path
        self.ref_cap = cv2.VideoCapture(ref_video_path)
        self.aligned_cap = cv2.VideoCapture(aligned_video_path)
        
        # 验证视频参数并确定处理尺寸
        self._validate_video_params()
        
        # 预加载参考视频多特征帧（包含原始帧号）
        self.ref_frames_info = []
        total_ref_seconds = self.total_ref // self.fps
        for second in range(total_ref_seconds):
            frame_info_set = []
            # 在每秒内均匀采样多个参考帧
            for ratio in np.linspace(0.1, 0.9, REF_FRAMES_PER_SECOND):
                frame_idx = int((second + ratio) * self.fps)
                frame_idx = min(frame_idx, self.total_ref-1)
                self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.ref_cap.read()
                if ret:
                    # 调整到处理尺寸
                    processed_frame = cv2.resize(frame, self.frame_size) if frame.shape[:2] != self.frame_size[::-1] else frame
                    frame_info = {
                        'processed_frame': processed_frame,
                        'frame_idx': frame_idx
                    }
                    frame_info_set.append(frame_info)
                else:
                    frame_info_set.append(None)
            self.ref_frames_info.append(frame_info_set)
        
        # 特征检测参数
        self.index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        self.search_params = dict(checks=100)
        
        # 结果存储
        self.similarity_data = {}

    def _validate_video_params(self):
        # 校验基础参数并确定处理尺寸
        ref_fps = int(self.ref_cap.get(cv2.CAP_PROP_FPS))
        aligned_fps = int(self.aligned_cap.get(cv2.CAP_PROP_FPS))
        if ref_fps != aligned_fps:
            raise ValueError(f"帧率不匹配: 参考视频{ref_fps}fps vs 对齐视频{aligned_fps}fps")
        
        # 确定处理尺寸
        ref_width = int(self.ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ref_height = int(self.ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 调整分辨率逻辑
        max_width = 1280
        if ref_width > max_width:
            scaling_factor = max_width / ref_width
            new_width = max_width
            new_height = int(ref_height * scaling_factor)
            new_height = new_height if new_height % 2 == 0 else new_height + 1
            self.frame_size = (new_width, new_height)
        else:
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
    def _process_second(args):
        (second, ref_paths_with_idx, ref_video_path, aligned_video_path, method, 
         output_dir, frame_size, fps, total_aligned) = args
        
        # 初始化特征检测器
        detector = cv2.SIFT_create(nfeatures=2000)
        matcher = cv2.FlannBasedMatcher(
            dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
            dict(checks=100)
        )
        
        # 加载参考帧信息
        ref_frames = []
        for item in ref_paths_with_idx:
            if item is None or item['path'] is None:
                ref_frames.append(None)
                continue
            frame = cv2.imread(item['path'])
            if frame is not None:
                ref_frames.append({
                    'processed_frame': frame,
                    'frame_idx': item['frame_idx']
                })
            else:
                ref_frames.append(None)
        
        # 计算目标视频搜索范围
        start_sec = max(0, second - SEARCH_WINDOW_SECONDS)
        end_sec = min(total_aligned // fps, second + SEARCH_WINDOW_SECONDS)
        target_start = start_sec * fps
        target_end = min((end_sec + 1) * fps - 1, total_aligned - 1)
        
        # 读取目标视频帧（处理尺寸）
        aligned_cap = cv2.VideoCapture(aligned_video_path)
        aligned_cap.set(cv2.CAP_PROP_POS_FRAMES, target_start)
        target_frames = []
        for _ in range(target_end - target_start + 1):
            ret, frame = aligned_cap.read()
            if ret:
                # 调整到处理尺寸
                processed_frame = cv2.resize(frame, frame_size) if frame.shape[:2] != frame_size[::-1] else frame
                target_frames.append(processed_frame)
            else:
                target_frames.append(None)
        aligned_cap.release()
        
        max_similarity = 0.0
        best_ref_info = None
        best_target_idx = target_start
        
        for i, target_frame in enumerate(target_frames):
            if target_frame is None:
                continue
            current_idx = target_start + i
            
            # 多参考帧综合评分
            frame_similarity = 0.0
            current_best_ref = None
            
            for ref_info in ref_frames:
                if ref_info is None:
                    continue
                ref_frame = ref_info['processed_frame']
                
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
                
                # 结构相似度
                ssim_score = compare_ssim(
                    cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY),
                    data_range=255
                ) if ref_frame.shape == target_frame.shape else 0.0
                
                # 综合评分（可调节权重）
                combined_score = 0.7 * feat_sim + 0.3 * ssim_score
                if combined_score > frame_similarity:
                    frame_similarity = combined_score
                    current_best_ref = ref_info
            
            if frame_similarity > max_similarity:
                max_similarity = frame_similarity
                best_ref_info = current_best_ref
                best_target_idx = current_idx
        
        # 保存结果
        result_data = (second, 0.0, None, None, 0)
        if max_similarity > 0.01 and best_ref_info is not None:
            os.makedirs(output_dir, exist_ok=True)
            
            # 读取原始参考帧
            ref_cap = cv2.VideoCapture(ref_video_path)
            ref_cap.set(cv2.CAP_PROP_POS_FRAMES, best_ref_info['frame_idx'])
            ret_ref, original_ref = ref_cap.read()
            ref_cap.release()
            
            # 读取原始目标帧
            aligned_cap = cv2.VideoCapture(aligned_video_path)
            aligned_cap.set(cv2.CAP_PROP_POS_FRAMES, best_target_idx)
            ret_target, original_target = aligned_cap.read()
            aligned_cap.release()
            
            if ret_ref and ret_target:
                # 保存文件路径
                comp_path = os.path.join(output_dir, f"sec{second}_comp.jpg")
                ref_path = os.path.join(output_dir, f"sec{second}_ref.jpg")
                target_path = os.path.join(output_dir, f"sec{second}_target.jpg")
                
                # 保存原始尺寸图像
                cv2.imwrite(ref_path, original_ref)
                cv2.imwrite(target_path, original_target)
                cv2.imwrite(comp_path, np.hstack((original_ref, original_target)))
                
                # 计算帧偏移量
                frame_offset = best_target_idx - (second * fps)
                result_data = (second, max_similarity, ref_path, target_path, frame_offset)
        
        return result_data

    def compare_seconds(self, method='feature', output_dir='enhanced_output'):
        os.makedirs(output_dir, exist_ok=True)
        total_seconds = len(self.ref_frames_info)
        
        # 保存处理后的参考帧（用于比对）并记录原始帧号
        ref_frame_dir = os.path.join(output_dir, "multi_ref_frames")
        os.makedirs(ref_frame_dir, exist_ok=True)
        ref_paths_with_idx = []
        for second_idx, frame_info_set in enumerate(self.ref_frames_info):
            sec_data = []
            for f_idx, frame_info in enumerate(frame_info_set):
                if frame_info is not None:
                    path = os.path.join(ref_frame_dir, f"ref_{second_idx}_{f_idx}.jpg")
                    cv2.imwrite(path, frame_info['processed_frame'])
                    sec_data.append({
                        'path': path,
                        'frame_idx': frame_info['frame_idx']
                    })
                else:
                    sec_data.append(None)
            ref_paths_with_idx.append(sec_data)
        
        # 准备并行参数
        args_list = []
        for second in range(total_seconds):
            args = (
                second,
                ref_paths_with_idx[second],
                self.ref_video_path,
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
        ref_video_path="alex1.mp4",
        aligned_video_path="aligned_alex.mp4"       
    )
    comparator.compare_seconds(method='feature', output_dir='enhanced_output')
    comparator.generate_report('enhanced_report.csv')