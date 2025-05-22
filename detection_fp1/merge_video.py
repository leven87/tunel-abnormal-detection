import cv2
import numpy as np
import csv
import os

def merge_videos(vid1_path, vid2_path, output_path, csv_path='detection_result.csv'):
    csv_path = os.path.join(os.path.dirname(output_path), 'detection_result.csv')
    # 读取CSV数据
    csv_data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted_row = {
                'Total New': int(row['Total New']),
                'Total Missing': int(row['Total Missing']),
                'Total Matched': int(row['Total Matched']),
                'Total Dislocated': int(row['Total Dislocated'])
            }
            csv_data.append(converted_row)

    resolution_scale = 0.5
    cap1 = cv2.VideoCapture(vid1_path)
    cap2 = cv2.VideoCapture(vid2_path)

    # 获取视频参数
    width1 = int(cap1.get(3) * resolution_scale)
    height1 = int(cap1.get(4) * resolution_scale)
    width2 = int(cap2.get(3) * resolution_scale)
    height2 = int(cap2.get(4) * resolution_scale)

    merged_width = width1 + width2
    merged_height = max(height1, height2)
    fps = max(cap1.get(5), cap2.get(5))

    # 图例参数
    legend_height = 120
    total_height = merged_height + legend_height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (merged_width, total_height))

    # 图例配置（调整顺序并添加字段映射）
    legend_config = [
        {"color": (0, 165, 255), "text": "New", "field": "Total New"},
        {"color": (0, 0, 255),   "text": "Missing", "field": "Total Missing"},
        {"color": (0, 255, 0),   "text": "Matched", "field": "Total Matched"},
        {"color": (255, 192, 203), "text": "Dislocated", "field": "Total Dislocated"}
    ]

    frame_count = 0
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 and not ret2: 
            break

        # 处理缺失帧
        frame1 = frame1 if ret1 else np.zeros((height1, width1, 3), dtype=np.uint8)
        frame2 = frame2 if ret2 else np.zeros((height2, width2, 3), dtype=np.uint8)

        # 缩放帧
        frame1 = cv2.resize(frame1, (width1, merged_height), interpolation=cv2.INTER_CUBIC)
        frame2 = cv2.resize(frame2, (width2, merged_height), interpolation=cv2.INTER_CUBIC)
        
        # 合并视频
        merged = np.hstack((frame1, frame2))
        
        # 创建图例
        legend = np.zeros((legend_height, merged_width, 3), dtype=np.uint8)
        legend[:] = (40, 40, 40)
        
        # 绘制动态统计信息
        x_pos = 30
        box_size = 40
        font_scale = 1.0
        font_thickness = 2
        
        if frame_count < len(csv_data):
            current_stats = csv_data[frame_count]
        else:
            current_stats = csv_data[-1] if csv_data else {}

        for item in legend_config:
            # 获取统计值
            value = current_stats.get(item["field"], 0)
            label = f"{item['text']}: {value}"
            
            # 绘制颜色块
            cv2.rectangle(legend,
                        (x_pos, 30),
                        (x_pos + box_size, 30 + box_size),
                        item["color"], -1)
            
            # 计算文字位置
            (text_width, text_height), _ = cv2.getTextSize(label, 
                                                          cv2.FONT_HERSHEY_SIMPLEX,
                                                          font_scale,
                                                          font_thickness)
            
            # 文字垂直居中
            text_y = 30 + box_size//2 + text_height//2
            
            cv2.putText(legend, label,
                      (x_pos + box_size + 15, text_y),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      font_scale,
                      (255, 255, 255),
                      font_thickness)
            
            # 更新水平位置
            x_pos += box_size + text_width + 60

        # 合并画面
        final_frame = np.vstack((merged, legend))
        out.write(final_frame)
        frame_count += 1

    cap1.release()
    cap2.release()
    out.release()

# # 使用示例
# merge_videos("input1.mp4", "input2.mp4", "stats.csv", "output.mp4")
