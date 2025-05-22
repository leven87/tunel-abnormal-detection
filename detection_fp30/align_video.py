import cv2
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn.functional as F

class AdvancedTrainAligner:
    def __init__(self, ref_video_path, target_video_path):
        # 初始化视频源
        self.ref_cap = cv2.VideoCapture(ref_video_path)
        self.target_cap = cv2.VideoCapture(target_video_path)
        
        # 获取视频参数
        self.fps = int(self.ref_cap.get(cv2.CAP_PROP_FPS))
        self.frame_size = (
            int(self.ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        self.total_ref = int(self.ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_target = int(self.target_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 增强特征检测参数
        self.detector = cv2.SIFT_create(nfeatures=2000)
        self.matcher = cv2.FlannBasedMatcher()
        self.roi_config = {
            'start': (0.3, 0.5, 0.7, 0.8),  # 起始检测区域
            'end': (0.3, 0.5, 0.7, 0.8),     # 结束检测区域
            'full': (0.0, 0.0, 1.0, 1.0)     # 整个帧
        }

        # 对齐参数
        self.ref_range = (0, 0)
        self.target_range = (0, 0)

    def _get_roi(self, frame, roi_type):
        """获取优化后的检测区域"""
        h, w = frame.shape[:2]
        cfg = self.roi_config[roi_type]
        x1 = int(w * cfg[0])
        y1 = int(h * cfg[1])
        x2 = int(w * cfg[2])
        y2 = int(h * cfg[3])
        return frame[y1:y2, x1:x2]

    def _extract_enhanced_features(self, frame, roi_type):
        """增强特征提取流程"""
        roi = self._get_roi(frame, roi_type)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        
        # 多尺度特征检测
        kp = self.detector.detect(gray, None)
        kp, des = self.detector.compute(gray, kp)
        return kp, des

    def _match_with_consistency_check(self, ref_frame, target_frames, roi_type):
        """带一致性检查的特征匹配"""
        best_matches = []
        ref_kp, ref_des = self._extract_enhanced_features(ref_frame, roi_type)
        
        for pos in target_frames:
            self.target_cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            _, target_frame = self.target_cap.read()
            target_kp, target_des = self._extract_enhanced_features(target_frame, roi_type)
            
            if target_des is None or ref_des is None:
                continue
                
            matches = self.matcher.knnMatch(ref_des, target_des, k=2)
            # Lowe's ratio test
            good = [m for m,n in matches if m.distance < 0.7*n.distance]
            best_matches.append((pos, len(good)))
        
        # 寻找连续稳定匹配区域
        best_matches.sort(key=lambda x: x[0])
        window_size = 5
        smoothed = []
        for i in range(len(best_matches)-window_size+1):
            avg = sum(m[1] for m in best_matches[i:i+window_size])/window_size
            smoothed.append((best_matches[i+window_size//2][0], avg))
        
        return max(smoothed, key=lambda x: x[1])

    def detect_end_point(self, ref_end_frame, initial_guess):
        """精准结束点检测"""
        # 在猜测点附近精细搜索
        search_start = max(0, initial_guess - 100)
        search_end = min(self.total_target, initial_guess + 100)
        return self._match_with_consistency_check(
            ref_end_frame,
            range(search_start, search_end),
            'end'
        )

    def alignment_workflow(self):
        """改进后的对齐流程"""
        # 起始点检测
        self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, ref_start = self.ref_cap.read()
        print("定位起始点...")
        start_guess = self._match_with_consistency_check(
            ref_start,
            range(0, min(100, self.total_target)),
            'start'
        )[0]
        
        # 结束点检测
        self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, self.total_ref-1)
        _, ref_end = self.ref_cap.read()
        print("\n初步结束点检测...")
        end_guess = self._match_with_consistency_check(
            ref_end,
            range(max(0, self.total_target-100), self.total_target-1),
            'end'
        )[0]
        
        print("\n精确结束点检测...")
        final_end = self.detect_end_point(ref_end, end_guess)[0]
        
        self.ref_range = (0, self.total_ref-1)
        self.target_range = (start_guess, final_end)
        print(f"\n目标视频有效范围：{self.target_range[0]} - {self.target_range[1]}")

    def compute_frame_similarity(self, frame1, frame2):
        """使用特征匹配数量计算相似度"""
        # 提取整个帧的特征
        kp1, des1 = self._extract_enhanced_features(frame1, 'full')
        kp2, des2 = self._extract_enhanced_features(frame2, 'full')
        
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return 0
        
        # 特征匹配
        matches = self.matcher.knnMatch(des1, des2, k=2)
        # Lowe's ratio test
        good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
        return len(good_matches)

    def save_similar_frames(self, similarity_data):
        """保存每秒钟最相似的参考帧和目标帧"""
        output_dir = "similar_frames"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for second in similarity_data:
            similarity, ref_frame, target_frame = similarity_data[second]
            ref_path = os.path.join(output_dir, f"second_{second}_ref.jpg")
            target_path = os.path.join(output_dir, f"second_{second}_target.jpg")
            cv2.imwrite(ref_path, ref_frame)
            cv2.imwrite(target_path, target_frame)
            print(f"Saved frames for second {second} with similarity {similarity}")


    def compute_frame_similarity2(self, ref_frame, target_frame):
        """GPU加速的SSIM相似度计算"""
        # 转换为PyTorch张量并归一化
        ref_tensor = torch.from_numpy(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)).float().unsqueeze(0).unsqueeze(0) / 255.0
        target_tensor = torch.from_numpy(cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)).float().unsqueeze(0).unsqueeze(0) / 255.0
        
        # 移至GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ref_tensor = ref_tensor.to(device)
        target_tensor = target_tensor.to(device)
        
        # 计算SSIM
        return self._ssim(ref_tensor, target_tensor).item()

    def _ssim(self, x, y, window_size=11, size_average=True):
        """PyTorch实现SSIM计算"""
        window = torch.ones(window_size, window_size).unsqueeze(0).unsqueeze(0)
        window = window.to(x.device)
        
        mu_x = F.conv2d(x, window, padding=window_size//2, groups=1)
        mu_y = F.conv2d(y, window, padding=window_size//2, groups=1)
        
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(x*x, window, padding=window_size//2, groups=1) - mu_x_sq
        sigma_y_sq = F.conv2d(y*y, window, padding=window_size//2, groups=1) - mu_y_sq
        sigma_xy = F.conv2d(x*y, window, padding=window_size//2, groups=1) - mu_xy
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu_xy + C1)*(2*sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1)*(sigma_x_sq + sigma_y_sq + C2))
        return ssim_map.mean()
    
    def generate_precise_video(self, output_path):
        """生成精准对齐视频并记录每秒钟最相似帧"""
        ref_duration = self.ref_range[1] - self.ref_range[0]
        target_duration = self.target_range[1] - self.target_range[0]
        speed_ratio = ref_duration / target_duration
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, self.frame_size)
        black = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        
        # 初始化相似度记录
        similarity_data = {}  # key: 秒数, value: (max_similarity, ref_frame, target_frame)
        
        print("\n生成视频并计算相似度...")
        for i in tqdm(range(self.total_ref)):
            # 读取参考视频的第i帧
            self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret_ref, ref_frame = self.ref_cap.read()
            if not ret_ref:
                ref_frame = black
            
            # 计算目标视频的位置
            target_pos = int(self.target_range[0] + i / speed_ratio)
            target_pos = max(self.target_range[0], min(target_pos, self.target_range[1]))
            
            self.target_cap.set(cv2.CAP_PROP_POS_FRAMES, target_pos)
            ret_target, target_frame = self.target_cap.read()
            if not ret_target:
                target_frame = black
            
            # 调整目标帧大小（如果必要）
            target_frame = cv2.resize(target_frame, self.frame_size) if ret_target else black
            
            # 计算相似度
            similarity = self.compute_frame_similarity2(ref_frame, target_frame)
            
            # 记录当前秒数
            current_second = i // self.fps
            
            # 更新相似度数据
            if current_second not in similarity_data or similarity > similarity_data[current_second][0]:
                similarity_data[current_second] = (similarity, ref_frame.copy(), target_frame.copy())
            
            # 写入目标视频帧到输出
            out.write(target_frame)
        
        out.release()
        
        # 保存相似帧
        self.save_similar_frames(similarity_data)

    def process(self, output_path):
        """完整处理流程"""
        self.alignment_workflow()
        self.generate_precise_video(output_path)
        self.ref_cap.release()
        self.target_cap.release()

if __name__ == "__main__":
    # aligner = AdvancedTrainAligner("alex2.mp4", "alex1.mp4")
    # aligner.process("aligned_alex_reversed.mp4")

    # aligner = AdvancedTrainAligner("alex1.mp4", "alex2.mp4")
    # aligner.process("aligned_alex.mp4")

    aligner = AdvancedTrainAligner("Trainingvideos_1_1280_720.mp4", "Test-1_1280_720.mp4")
    aligner.process("aligned_Test-1_1280_720.mp4")

