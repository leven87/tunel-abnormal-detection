# import cv2
# import numpy as np
# from tqdm import tqdm
# import os
# from skimage.metrics import structural_similarity as compare_ssim

# # 可配置参数
# FLANN_INDEX_KDTREE = 1
# SEARCH_WINDOW_SECONDS = 1
# SEARCH_INTERVAL = 0.1
# REF_FRAMES_PER_INTERVAL = 3
# MAX_FEATURES = 2000
# RESIZE_WIDTH = 1280
# SIMILARITY_THRESHOLD = 0.01
# FEATURE_WEIGHT = 0.7
# SSIM_WEIGHT = 0.3

# class VideoComparator:
#     def __init__(self, ref_video_path, aligned_video_path):
#         self.ref_video_path = ref_video_path
#         self.aligned_video_path = aligned_video_path
#         self.ref_cap = cv2.VideoCapture(ref_video_path)
#         self.aligned_cap = cv2.VideoCapture(aligned_video_path)
        
#         self._validate_video_params()
#         self.ref_frames_info = self._preload_reference_frames()
        
#         # 特征匹配器初始化
#         self.detector = cv2.SIFT_create(nfeatures=MAX_FEATURES)
#         self.matcher = cv2.FlannBasedMatcher(
#             dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
#             dict(checks=100)
#         )
        
#         self.similarity_data = {}
#         self.last_matched_frame = 0  # 跟踪最后匹配的帧位置

#     def _validate_video_params(self):
#         ref_fps = self.ref_cap.get(cv2.CAP_PROP_FPS)
#         aligned_fps = self.aligned_cap.get(cv2.CAP_PROP_FPS)
#         if abs(ref_fps - aligned_fps) > 1e-3:
#             raise ValueError(f"帧率不匹配: 参考视频{ref_fps}fps vs 对齐视频{aligned_fps}fps")
#         self.fps = int(ref_fps)
        
#         ref_width = int(self.ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         if ref_width > RESIZE_WIDTH:
#             self.frame_size = (RESIZE_WIDTH, int(self.ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_WIDTH / ref_width))
#         else:
#             self.frame_size = (ref_width, int(self.ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
#         self.total_ref = int(self.ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.total_aligned = int(self.aligned_cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     def _preload_reference_frames(self):
#         frames_info = []
#         total_intervals = int((self.total_ref / self.fps) / SEARCH_INTERVAL)
        
#         for interval in range(total_intervals):
#             current_time = interval * SEARCH_INTERVAL
#             frame_info_set = []
            
#             for ratio in np.linspace(0.1, 0.9, REF_FRAMES_PER_INTERVAL):
#                 target_time = current_time + ratio * SEARCH_INTERVAL
#                 frame_idx = int(target_time * self.fps)
#                 frame_idx = min(frame_idx, self.total_ref - 1)
                
#                 self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#                 ret, frame = self.ref_cap.read()
                
#                 if ret:
#                     processed = cv2.resize(frame, self.frame_size) if frame.shape[:2] != self.frame_size[::-1] else frame
#                     frame_info_set.append({
#                         'original_idx': frame_idx,
#                         'processed_frame': processed,
#                         'time': target_time
#                     })
#                 else:
#                     frame_info_set.append(None)
            
#             frames_info.append(frame_info_set)
#         return frames_info

#     def _calculate_similarity(self, ref_frame, target_frame):
#         gray_ref = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
#         gray_target = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
        
#         kp_ref, des_ref = self.detector.detectAndCompute(gray_ref, None)
#         kp_target, des_target = self.detector.detectAndCompute(gray_target, None)
        
#         feat_sim = 0.0
#         if des_ref is not None and des_target is not None:
#             matches = self.matcher.knnMatch(des_ref, des_target, k=2)
#             good = [m for m, n in matches if m.distance < 0.6 * n.distance]
            
#             if len(good) > 4:
#                 src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good])
#                 dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good])
#                 _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#                 inliers = np.sum(mask) if mask is not None else 0
#                 feat_sim = (inliers + len(good)) / (2 * min(len(kp_ref), len(kp_target)))
        
#         ssim_score = compare_ssim(gray_ref, gray_target, data_range=255) if gray_ref.shape == gray_target.shape else 0.0
#         return FEATURE_WEIGHT * feat_sim + SSIM_WEIGHT * ssim_score

#     def compare_seconds(self, output_dir='output'):
#         os.makedirs(output_dir, exist_ok=True)
#         self.similarity_data = {}
#         self.last_matched_frame = 0  # 初始化最后匹配的帧位置

#         total_intervals = len(self.ref_frames_info)
#         with tqdm(total=total_intervals, desc="Processing video") as pbar:
#             for idx in range(total_intervals):
#                 current_time = idx * SEARCH_INTERVAL
#                 ref_info_set = self.ref_frames_info[idx]
                
#                 # 计算动态搜索范围
#                 start_time = max(current_time - SEARCH_WINDOW_SECONDS, 
#                                self.last_matched_frame / self.fps)
#                 # start_frame = max(int(start_time * self.fps), self.last_matched_frame + 1)
#                 start_frame = max(int(start_time * self.fps), self.last_matched_frame)
#                 end_time = current_time + SEARCH_WINDOW_SECONDS

#                 # 第一次匹配可以放宽搜索范围
#                 if self.last_matched_frame == 0:
#                     end_frame = min(int(end_time * self.fps), self.total_aligned - 1)
#                 else:
#                     # 第二次匹配限制搜索范围为5 帧
#                     end_frame = min(int(end_time * self.fps), self.total_aligned - 1, start_frame+15)    

#                 if start_frame > end_frame:
#                     pbar.update(1)
#                     continue
                
#                 # 读取目标视频帧
#                 aligned_cap = cv2.VideoCapture(self.aligned_video_path)
#                 aligned_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
#                 max_similarity = 0.0
#                 best_match = None
                
#                 for _ in range(end_frame - start_frame + 1):
#                     ret, target_frame = aligned_cap.read()
#                     if not ret:
#                         break
                    
#                     current_pos = int(aligned_cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
#                     target_processed = cv2.resize(target_frame, self.frame_size) if target_frame.shape[:2] != self.frame_size[::-1] else target_frame
                    
#                     for ref_info in ref_info_set:
#                         if ref_info is None:
#                             continue
                        
#                         similarity = self._calculate_similarity(
#                             ref_info['processed_frame'], 
#                             target_processed
#                         )
                        
#                         if similarity > max_similarity:
#                             max_similarity = similarity
#                             best_match = (ref_info['original_idx'], current_pos)
                
#                 aligned_cap.release()
                
#                 if max_similarity > SIMILARITY_THRESHOLD and best_match:
#                     self.last_matched_frame = best_match[1]  # 更新最后匹配位置
                    
#                     # 保存结果
#                     ref_idx, target_idx = best_match
#                     self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
#                     _, ref_frame = self.ref_cap.read()
                    
#                     aligned_cap = cv2.VideoCapture(self.aligned_video_path)
#                     aligned_cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
#                     _, target_frame = aligned_cap.read()
#                     aligned_cap.release()
                    
#                     print(f"Matched at {current_time:.1f}s: Ref {ref_idx}, Target {target_idx}, Similarity {max_similarity:.4f}")
#                     time_str = f"{current_time:.1f}".replace('.', '_')
#                     ref_path = os.path.join(output_dir, f"ref_{time_str}.jpg")
#                     target_path = os.path.join(output_dir, f"target_{time_str}.jpg")
#                     comp_path = os.path.join(output_dir, f"comp_{time_str}.jpg")
                    
#                     cv2.imwrite(ref_path, ref_frame)
#                     cv2.imwrite(target_path, target_frame)
#                     cv2.imwrite(comp_path, np.hstack((ref_frame, target_frame)))
                    
#                     frame_offset = target_idx - int(current_time * self.fps)
#                     self.similarity_data[current_time] = (
#                         max_similarity, ref_path, target_path, frame_offset
#                     )
                
#                 pbar.update(1)

#     def generate_report(self, report_path='report.csv'):
#         with open(report_path, 'w') as f:
#             f.write("Time(sec),Similarity,RefPath,TargetPath,FrameOffset\n")
#             for time in sorted(self.similarity_data.keys()):
#                 data = self.similarity_data[time]
#                 f.write(f"{time:.1f},{data[0]:.4f},{data[1]},{data[2]},{data[3]}\n")

# if __name__ == "__main__":
#     comparator = VideoComparator("reference.mp4", "aligned.mp4")
#     comparator.compare_seconds("comparison_results")
#     comparator.generate_report("detailed_report.csv")



# gpu加速版本
import cv2
import numpy as np
from tqdm import tqdm
import os
import torch
import torchvision
import kornia
from torchmetrics.image import StructuralSimilarityIndexMeasure

# 配置参数
SEARCH_WINDOW_SECONDS = 2
SEARCH_INTERVAL = 1/30
REF_FRAMES_PER_INTERVAL = 1
RESIZE_WIDTH = 1280
SIMILARITY_THRESHOLD = 0.01
FEATURE_WEIGHT = 0.7
SSIM_WEIGHT = 0.3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureExtractor(torch.nn.Module):
    """使用 VGG16 的中间层进行特征提取"""
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(pretrained=True)
        self.features = torch.nn.Sequential(*list(vgg.features.children())[:15])
        
    def forward(self, x):
        return self.features(x)

class VideoComparator:
    def __init__(self, ref_video_path, aligned_video_path):
        # 初始化视频读取器
        self.ref_video_path = ref_video_path
        self.aligned_video_path = aligned_video_path
        self.ref_cap = cv2.VideoCapture(ref_video_path)
        self.aligned_cap = cv2.VideoCapture(aligned_video_path)
        
        # 新增关键初始化步骤
        self.feature_extractor = FeatureExtractor().to(DEVICE).eval()  # 初始化特征提取器
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)  # 初始化SSIM计算器
        
        self._validate_video_params()
        self.ref_frames_info = self._preload_reference_frames()
        
        # 状态变量
        self.similarity_data = {}
        self.last_matched_frame = 0

    # def _validate_video_params(self):
    #     # 保持原有参数验证逻辑不变
    #     ref_fps = self.ref_cap.get(cv2.CAP_PROP_FPS)
    #     aligned_fps = self.aligned_cap.get(cv2.CAP_PROP_FPS)
    #     if abs(ref_fps - aligned_fps) > 1e-3:
    #         raise ValueError(f"帧率不匹配: 参考视频{ref_fps}fps vs 对齐视频{aligned_fps}fps")
        
    #     self.fps = int(ref_fps)
    #     ref_width = int(self.ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     self.frame_size = (RESIZE_WIDTH, int(self.ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * RESIZE_WIDTH / ref_width)) \
    #         if ref_width > RESIZE_WIDTH else (ref_width, int(self.ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
    #     self.total_ref = int(self.ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     self.total_aligned = int(self.aligned_cap.get(cv2.CAP_PROP_FRAME_COUNT))


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
        max_width = RESIZE_WIDTH
        if ref_width > max_width:
            scaling_factor = max_width / ref_width
            new_width = max_width
            new_height = int(ref_height * scaling_factor)
            new_height = new_height if new_height % 2 == 0 else new_height + 1
            self.frame_size = (new_width, new_height)
        else:
            self.frame_size = (ref_width, ref_height)
        
        # self.frame_size = (ref_width, ref_height)

        self.fps = ref_fps
        self.total_ref = int(self.ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_aligned = int(self.aligned_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _preload_reference_frames(self):
        frames_info = []
        total_intervals = int((self.total_ref / self.fps) / SEARCH_INTERVAL)
        
        for interval in tqdm(range(total_intervals), desc="预加载参考帧"):
            frame_info_set = []
            current_time = interval * SEARCH_INTERVAL
            
            for ratio in np.linspace(0.1, 0.9, REF_FRAMES_PER_INTERVAL):
                target_time = current_time + ratio * SEARCH_INTERVAL
                frame_idx = min(int(target_time * self.fps), self.total_ref - 1)
                
                self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = self.ref_cap.read()
                
                if ret:
                    # 预处理帧
                    processed = cv2.resize(frame, self.frame_size) if frame.shape[1] != self.frame_size[0] else frame
                    tensor = self._frame_to_tensor(processed)
                    
                    # 提取特征
                    with torch.no_grad():
                        feature = self.feature_extractor(tensor.unsqueeze(0))
                        feature = torch.nn.functional.adaptive_avg_pool2d(feature, (1, 1)).view(-1).cpu()
                    
                    frame_info_set.append({
                        'original_idx': frame_idx,
                        'tensor': tensor.cpu(),
                        'feature': feature,
                        'time': target_time
                    })
                else:
                    frame_info_set.append(None)
            
            frames_info.append(frame_info_set)
        return frames_info

    def _frame_to_tensor(self, frame):
        """将 OpenCV 帧转换为 PyTorch 张量"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return tensor.to(DEVICE)

    def _calculate_similarity(self, ref_info, target_tensor):
        """GPU 加速的相似度计算"""
        # 特征相似度
        ref_feature = ref_info['feature'].to(DEVICE)
        with torch.no_grad():
            target_feature = self.feature_extractor(target_tensor.unsqueeze(0))
            target_feature = torch.nn.functional.adaptive_avg_pool2d(target_feature, (1, 1)).view(-1)
        
        feat_sim = torch.nn.functional.cosine_similarity(
            ref_feature.unsqueeze(0), target_feature.unsqueeze(0)
        ).item()
        feat_sim = (feat_sim + 1) / 2  # 归一化到 [0, 1]
        # feat_sim = 0.0

        # SSIM 计算
        ref_tensor = ref_info['tensor'].to(DEVICE)
        target_resized = torch.nn.functional.interpolate(
            target_tensor.unsqueeze(0),
            size=ref_tensor.shape[1:],
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # 转换为灰度
        gray_ref = kornia.color.rgb_to_grayscale(ref_tensor.unsqueeze(0))
        gray_target = kornia.color.rgb_to_grayscale(target_resized.unsqueeze(0))
        
        ssim_score = self.ssim(gray_ref, gray_target).item()
        
        return FEATURE_WEIGHT * feat_sim + SSIM_WEIGHT * ssim_score

    def compare_seconds(self, output_dir='output'):
        os.makedirs(output_dir, exist_ok=True)
        self.similarity_data = {}
        self.last_matched_frame = 0

        total_intervals = len(self.ref_frames_info)
        with tqdm(total=total_intervals, desc="视频比对进度") as pbar:
            for idx in range(total_intervals):
                current_time = idx * SEARCH_INTERVAL
                ref_info_set = self.ref_frames_info[idx]
                
                # 计算动态搜索范围
                start_frame = max(
                    int((current_time - SEARCH_WINDOW_SECONDS) * self.fps),
                    self.last_matched_frame
                )
                end_frame = min(
                    int((current_time + SEARCH_WINDOW_SECONDS) * self.fps),
                    self.total_aligned - 1,
                    start_frame + 50  # 限制最大搜索范围
                )
                
                if start_frame >= end_frame:
                    pbar.update(1)
                    continue

                # 批量处理目标帧
                target_frames = []
                aligned_cap = cv2.VideoCapture(self.aligned_video_path)
                aligned_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                for _ in range(end_frame - start_frame + 1):
                    ret, frame = aligned_cap.read()
                    if not ret: break
                    
                    processed = cv2.resize(frame, self.frame_size) if frame.shape[1] != self.frame_size[0] else frame
                    target_frames.append(self._frame_to_tensor(processed))
                
                aligned_cap.release()

                # GPU 批量计算
                max_similarity = 0.0
                best_match = None
                for frame_idx, target_tensor in enumerate(target_frames):
                    absolute_pos = start_frame + frame_idx
                    
                    for ref_info in ref_info_set:
                        if ref_info is None: continue
                        
                        similarity = self._calculate_similarity(ref_info, target_tensor)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_match = (ref_info['original_idx'], absolute_pos)

                # 记录结果
                if max_similarity > SIMILARITY_THRESHOLD and best_match:
                    self.last_matched_frame = best_match[1]
                    self._save_match_result(current_time, best_match, max_similarity, output_dir)

                pbar.update(1)

    def _save_match_result(self, current_time, best_match, similarity, output_dir):
        ref_idx, target_idx = best_match
        time_str = f"{current_time:.2f}".replace('.', '_')
        
        print(f"匹配成功: {current_time:.2f}s, 参考帧 {ref_idx}, 对齐帧 {target_idx}, 相似度 {similarity:.4f}")
        # 保存参考帧
        self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
        _, ref_frame = self.ref_cap.read()
        ref_path = os.path.join(output_dir, f"ref_{time_str}.jpg")
        cv2.imwrite(ref_path, ref_frame)
        
        # 保存目标帧
        aligned_cap = cv2.VideoCapture(self.aligned_video_path)
        aligned_cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        _, target_frame = aligned_cap.read()
        target_path = os.path.join(output_dir, f"target_{time_str}.jpg")
        cv2.imwrite(target_path, target_frame)
        aligned_cap.release()
        
        # 保存比对结果
        comp_path = os.path.join(output_dir, f"comp_{time_str}.jpg")
        cv2.imwrite(comp_path, np.hstack((ref_frame, target_frame)))
        
        frame_offset = target_idx - int(current_time * self.fps)
        self.similarity_data[current_time] = (
            similarity, ref_path, target_path, frame_offset
        )

    def generate_report(self, report_path='report.csv'):
        with open(report_path, 'w') as f:
            f.write("Time(sec),Similarity,RefPath,TargetPath,FrameOffset\n")
            for time in sorted(self.similarity_data.keys()):
                data = self.similarity_data[time]
                f.write(f"{time:.1f},{data[0]:.4f},{data[1]},{data[2]},{data[3]}\n")

if __name__ == "__main__":
    comparator = VideoComparator("reference.mp4", "aligned.mp4")
    comparator.compare_seconds("gpu_comparison_results")
    comparator.generate_report("gpu_report.csv")
