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
            for idx, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
                x1, y1, x2, y2 = map(int, box[:4])
                current_detections.append({
                    'id': track_ids[idx] if idx < len(track_ids) else 0,
                    'class': int(cls),
                    'center': ((x1+x2)//2, (y1+y2)//2),
                    'box': (x1, y1, x2, y2),
                    'conf': float(conf)
                })
            
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