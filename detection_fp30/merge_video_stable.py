import cv2
from tqdm import tqdm
import os, re
import numpy as np
import argparse


def merge_videos(vid1_path, vid2_path, output_path):
    # 新增分辨率倍数参数
    resolution_scale = 0.5  # 将整体分辨率放大2倍

    cap1 = cv2.VideoCapture(vid1_path)
    cap2 = cv2.VideoCapture(vid2_path)
    
    # 获取原始视频参数并放大
    width1 = int(cap1.get(3) * resolution_scale)
    height1 = int(cap1.get(4) * resolution_scale)
    width2 = int(cap2.get(3) * resolution_scale)
    height2 = int(cap2.get(4) * resolution_scale)
    
    merged_width = width1 + width2
    merged_height = max(height1, height2)
    fps = max(cap1.get(5), cap2.get(5))
    
    # 调整图例参数（放大高度和字体）
    legend_height = 100  # 原60改为100[1,3](@ref)
    total_height = merged_height + legend_height
    
    # 创建输出视频（使用更高分辨率）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (merged_width, total_height))
    
    # 图例颜色配置
    legend_config = [
        {"color": (0, 0, 255),   "text": "Missing"},
        {"color": (0, 165, 255), "text": "New "},
        {"color": (0, 255, 0),   "text": "Matched "},
        {"color": (255, 192, 203), "text": "Dislocated"}
    ]
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 and not ret2: break
        
        # 处理缺失帧
        if not ret1: 
            frame1 = np.zeros((height1, width1, 3), dtype=np.uint8)
        if not ret2: 
            frame2 = np.zeros((height2, width2, 3), dtype=np.uint8)
        
        # 修改缩放方式（使用高质量插值）
        frame1 = cv2.resize(frame1, (width1, merged_height), interpolation=cv2.INTER_CUBIC)
        frame2 = cv2.resize(frame2, (width2, merged_height), interpolation=cv2.INTER_CUBIC)
        
        # 横向拼接
        merged = np.hstack((frame1, frame2))
        
        # 创建图例区域
        legend = np.zeros((legend_height, merged_width, 3), dtype=np.uint8)
        legend[:] = (40, 40, 40)  # 深灰色背景
        
        # 绘制图例内容
        x_pos = 20
        box_size = 30  # 原30改为50[6](@ref)
        font_scale = 0.8  # 原0.8改为1.2[7](@ref)
        font_thickness = 1  # 保持原有
        text_offset = 20  # 原45改为80
        
        for item in legend_config:
            # 绘制颜色方块
            cv2.rectangle(legend, 
                        (x_pos, 15),
                        (x_pos + box_size, 15 + box_size),
                        item["color"], 
                        -1)
            
            # 添加文字说明
            cv2.putText(legend, 
                      item["text"], 
                      (x_pos + box_size + 15, 15 + box_size//2 + 10),  # 调整坐标
                      cv2.FONT_HERSHEY_DUPLEX,  # 改用更清晰字体[7](@ref)
                      font_scale,
                      (255, 255, 255),
                      font_thickness)
            
            x_pos += len(item["text"]) * 25 + 20  # 调整间距系数
        
        # 合并视频帧和图例
        final_frame = np.vstack((merged, legend))
        
        out.write(final_frame)
    
    cap1.release()
    cap2.release()
    out.release()