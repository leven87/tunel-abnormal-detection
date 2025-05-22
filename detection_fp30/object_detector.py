from ultralytics import YOLO
import cv2
import math
import torch
import csv, os
import numpy as np
from collections import defaultdict


class VideoObjectDetector:
    def __init__(self, model_name='best.pt', device=None):
        self.comparison_results = []
        self.model = YOLO(model_name)
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.class_names = self.model.names

        self.base_scale = 1.0
        self.line_thickness = int(4 * self.base_scale)
        self.font_scale = 1 * self.base_scale
        self.text_thickness = int(2 * self.base_scale)
        self.marker_size = int(5 * self.base_scale)
    
    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            results = self.model.track(
                frame,
                persist=True,
                verbose=False,
                conf=0.5,
                iou=0.5,
                device=self.device,
                tracker="bytetrack.yaml"
            )
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else []

            for idx, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                x1, y1, x2, y2 = map(int, box[:4])
                color = (0, 255, 0)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                label = f""
                label = f"ID:{track_ids[idx] if idx < len(track_ids) else 0}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.text_thickness)
                cv2.rectangle(frame,
                            (x1, y1 - int(35 * self.base_scale)),
                            (x1 + text_width, y1),
                            color,
                            -1)
                cv2.putText(frame,
                          label,
                          (x1, y1 - int(10 * self.base_scale)),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          self.font_scale,
                          (255, 255, 255),
                          self.text_thickness)
                cv2.circle(frame, (cx, cy), self.marker_size, (0, 0, 255), -1)
            
            out.write(frame)
        
        cap.release()
        out.release()

class EnhancedVideoObjectDetector(VideoObjectDetector):
    def __init__(self, model_name='best.pt', device=None):
        super().__init__(model_name, device)
        self.id_generator = 0  # 模拟跟踪器的内部ID计数器
        self.active_tracks = {}  # {track_id: (首次出现帧, 最后出现帧, 状态)}
        self.deleted_tracks = []  # 记录被删除的轨迹
        self.track_history = defaultdict(list)  # {track_id: [(frame_idx, center)]}
        self.max_age = 30
        self.user_id_generator = 1  # 用户自增ID
        self.track_id_to_user_id = {}  # 映射：跟踪器ID → 用户ID

        self.track_id_counter = defaultdict(int)  # 跟踪每个ID的出现次数
        self.min_show_frames = 4                  # 最小出现帧数阈值
        self.min_box_size = 20                    # 最小检测框边长（单位：像素        


    @staticmethod
    def calculate_iou(box1, box2):
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
        area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
        
        return area_inter / (area1 + area2 - area_inter + 1e-10)

    
    def process_video(self, input_path, output_path, save_detections=False, ref_detections=None):
        prev_boxes = {}
        self.report_data = []
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        self.detections = []
        frame_index = 0


        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 应用改进的跟踪参数
            results = self.model.track(
                frame,
                persist=True,
                verbose=False,
                conf=0.2,                    # 更低检测置信度
                iou=0.3,                     # 更宽松的NMS
                device=self.device,
                tracker='bytetrack.yaml'       # 使用自定义配置
            )
            
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else []

            current_detections = []
            current_ids = set()
            
            # for idx, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            #     x1, y1, x2, y2 = map(int, box[:4])        
            #     track_id = track_ids[idx] if idx < len(track_ids) else -1

            #     current_detections.append({
            #         # 'id': track_ids[idx] if idx < len(track_ids) else 0,
            #         'id': track_id,
            #         'class': int(cls),
            #         'center': ((x1+x2)//2, (y1+y2)//2),
            #         'box': (x1, y1, x2, y2),
            #         'conf': float(conf)
            #     })

            
            for idx, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                if idx >= len(track_ids) or track_ids[idx] == -1:
                    continue
                
                track_id = track_ids[idx]
                x1, y1, x2, y2 = map(int, box[:4])

                width, height = x2 - x1, y2 - y1                
                # 过滤条件1：检测框尺寸过小
                # if width < self.min_box_size or height < self.min_box_size:
                if (width * height) < 300:    
                    continue
                
                # 更新该ID的出现次数
                self.track_id_counter[track_id] += 1
                
                # 过滤条件2：出现帧数不足
                if self.track_id_counter[track_id] < self.min_show_frames:
                    continue

                # 若为新跟踪器ID，分配用户ID
                if track_id not in self.track_id_to_user_id:
                    self.track_id_to_user_id[track_id] = self.user_id_generator
                    self.user_id_generator += 1  # 确保连续增长
                
                user_id = self.track_id_to_user_id[track_id]
                
                current_detections.append({
                    'id': user_id,  # 使用连续的用户ID
                    'class': int(cls),
                    'center': ((x1+x2)//2, (y1+y2)//2),
                    'box': (x1, y1, x2, y2),
                    'conf': float(conf)
                })

            # 定期清理计数器（每隔60帧清理一次）
            if frame_index % 60 == 0:
                active_ids = set(track_ids)
                expired_ids = set(self.track_id_counter.keys()) - active_ids
                for tid in expired_ids:
                    del self.track_id_counter[tid]

            # 同步 ID 生成器：确保其始终为最大活跃 ID + 1
            all_active_ids = set(self.active_tracks.keys()).union(current_ids)
            if all_active_ids:
                max_track_id = max(all_active_ids)
                self.id_generator = max(max_track_id + 1, self.id_generator)

            # 更新活跃轨迹
            for track_id in current_ids:
                if track_id not in self.active_tracks:
                    print(f"跟踪器分配新ID: {track_id} | 内部生成器同步至: {self.id_generator}")
                    self.active_tracks[track_id] = {
                        'start_frame': frame_index,
                        'last_seen': frame_index,
                        'status': 'active'
                    }
                else:
                    self.active_tracks[track_id]['last_seen'] = frame_index



            # 记录当前帧所有活跃ID
            for obj in current_detections:
                track_id = obj['id']
                current_ids.add(track_id)
                
                # 新增ID注册逻辑
                if track_id not in self.active_tracks:
                    self.id_generator += 1
                    # 如果跟踪器内部有ID跳跃，这里会显示不一致
                    print(f"❗ 跟踪器分配新ID: {track_id} | 内部生成器预期下一个ID: {self.id_generator}")
                    self.active_tracks[track_id] = {
                        'start_frame': frame_index,
                        'last_seen': frame_index,
                        'status': 'active'
                    }
                else:
                    self.active_tracks[track_id]['last_seen'] = frame_index

            # 记录所有活跃和已删除的ID
            print(f"Frame {frame_index} 活跃ID: {list(current_ids)}")
            if self.deleted_tracks:
                print(f"已删除的ID历史: {self.deleted_tracks[-5:]}")  # 显示最近5个被删的ID

            for obj in current_detections:
                track_id = obj['id']
                center = obj['center']
                self.track_history[track_id].append((frame_index, center))
                # 绘制轨迹线（最近20帧）
                if len(self.track_history[track_id]) > 1:
                    points = np.array([p[1] for p in self.track_history[track_id][-20:]], np.int32)
                    cv2.polylines(frame, [points], isClosed=False, color=(255, 0, 255), thickness=2)

            if save_detections:
                self.detections.append(current_detections)

            if ref_detections is not None:
                ref_frame_dets = ref_detections[frame_index] if frame_index < len(ref_detections) else []
                
                matched_ref = set()
                matched_curr = set()
                iou_map = {}

                # IoU-based matching
                for r_idx, r_obj in enumerate(ref_frame_dets):
                    max_iou = 0.0
                    best_c_idx = -1
                    for c_idx, c_obj in enumerate(current_detections):
                        if c_obj['class'] == r_obj['class'] and c_idx not in matched_curr:
                            iou = self.calculate_iou(r_obj['box'], c_obj['box'])
                            if iou > max_iou:
                                max_iou = iou
                                best_c_idx = c_idx
                    if best_c_idx != -1 and max_iou >= 0.1:
                        matched_ref.add(r_idx)
                        matched_curr.add(best_c_idx)
                        iou_map[best_c_idx] = max_iou

                # Draw unmatched reference objects
                for r_idx, r_obj in enumerate(ref_frame_dets):
                    if r_idx not in matched_ref:
                        x1, y1, x2, y2 = r_obj['box']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), self.line_thickness)
                        cv2.circle(frame, r_obj['center'], self.marker_size, (0, 0, 255), -1)

                # Draw current detections
                for c_idx, c_obj in enumerate(current_detections):
                    color = (0, 165, 255)  # Default for new objects
                    if c_idx in iou_map:
                        iou = iou_map[c_idx]
                        color = (0, 255, 0) if iou >= 0.72 else (255, 192, 203)  # Green for matched, Pink for dislocated
                    
                    x1, y1, x2, y2 = c_obj['box']
                    cx, cy = c_obj['center']
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)
                    
                    label = f"ID:{c_obj['id']}"
                    # label = f""
                    if c_idx in iou_map:
                        label += f" IoU:{iou_map[c_idx]:.2f}"
                    
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                self.font_scale, self.text_thickness)
                    cv2.rectangle(frame, 
                                (x1, y1 - int(35*self.base_scale)),
                                (x1 + tw, y1),
                                color, -1)
                    
                    cv2.putText(frame, label,
                              (x1, y1 - int(10*self.base_scale)),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              self.font_scale,
                              (255, 255, 255),
                              self.text_thickness)
                    
                    cv2.circle(frame, (cx, cy), self.marker_size, (0, 0, 255), -1)

                # Update statistics
                frame_stats = {
                    'frame': frame_index,
                    'new': len(current_detections) - len(matched_curr),
                    'missing': len(ref_frame_dets) - len(matched_ref),
                    'matched': 0,
                    'dislocated': 0
                }

                for iou in iou_map.values():
                    if iou >= 0.72:
                        frame_stats['matched'] += 1
                    elif iou >= 0.1:
                        frame_stats['dislocated'] += 1
                
                self.report_data.append(frame_stats)

            else:
                for obj in current_detections:
                    x1, y1, x2, y2 = obj['box']
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)

            out.write(frame)
            frame_index += 1


        cap.release()
        out.release()

        if ref_detections is not None:
            report_path = os.path.join(os.path.dirname(output_path), 'detection_result.csv')
            self._generate_report(report_path, fps)

    def _generate_report(self, report_path: str, fps: float) -> None:
        with open(report_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Time(s)', 'New', 'Missing', 'Matched', 'Dislocated'])
            for row in self.report_data:
                writer.writerow([
                    row['frame'],
                    f"{row['frame']/fps:.2f}",
                    row['new'],
                    row['missing'],
                    row['matched'],
                    row['dislocated']
                ])