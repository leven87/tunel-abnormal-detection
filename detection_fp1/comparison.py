import cv2
from tqdm import tqdm
import os, re
import numpy as np
import argparse
from utils import convert_mp4v_to_h264
from object_detector import EnhancedVideoObjectDetector
from merge_video_stable import merge_videos

def natural_sort_key(s):
    # 提取文件名中的数字部分
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def images_to_video(image_dir, output_path, fps=1, type='ref'):
    if type == 'ref':
        img_files = sorted([f for f in os.listdir(image_dir) if f.endswith('_ref.jpg') ], key=natural_sort_key)
    else:
        img_files = sorted([f for f in os.listdir(image_dir) if f.endswith('_target.jpg') ],key=natural_sort_key)

    print(img_files)        
    frame = cv2.imread(os.path.join(image_dir, img_files[0]))
    h, w = frame.shape[:2]
    
    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 或 'h264' 
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
  
    for img_file in tqdm(img_files):
        img_path = os.path.join(image_dir, img_file)
        frame = cv2.imread(img_path)
        out.write(frame)
        
    out.release()


from align_video import AdvancedTrainAligner
from extract_align_img import VideoComparator
# from extract_align_img_bak import VideoComparator

def main():
    #  ffmpeg -i Trainingvideos_2.mp4 -vf "scale=1280:720" Trainingvideos_2_1280_720.mp4
    #  python comparison.py --ref Trainingvideos_2_1280_720.mp4 --test Test-2_1280_720.mp4

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Video alignment and comparison tool')
    parser.add_argument('--ref', required=True, help='Reference video file path')
    parser.add_argument('--test', required=True, help='Test video file path')
    args = parser.parse_args()

    # # Create output directory based on input filenames
    ref_name = os.path.splitext(os.path.basename(args.ref))[0]
    test_name = os.path.splitext(os.path.basename(args.test))[0]
    output_dir = f"compare_{ref_name}_vs_{test_name}"
    os.makedirs(output_dir, exist_ok=True)

    # # # # Align videos
    aligned_video = os.path.join(output_dir, f"aligned_{test_name}.mp4")
    aligner = AdvancedTrainAligner(args.ref, args.test)
    aligner.process(aligned_video)



    # # Extract and compare frames
    frames_dir = os.path.join(output_dir, "frames")
    comparator = VideoComparator(
        ref_video_path=args.ref,
        aligned_video_path=aligned_video        
    )
    # comparator.compare_seconds(method='feature', output_dir=frames_dir)
    comparator.compare_seconds(output_dir=frames_dir)
    comparator.generate_report(os.path.join(output_dir, f'comparison_report_{ref_name}_vs_{test_name}.csv'))
    # # # return 0

    # # Generate reference frames video
    ref_frames_video = os.path.join(output_dir, f"ref_frames_{ref_name}.mp4")
    # # ref_frames_video = os.path.join("Trainingvideos_2_1280_720.mp4")
    images_to_video(frames_dir, ref_frames_video, fps=1, type='ref')
    
    # # # # # Generate aligned frames video
    test_frames_video = os.path.join(output_dir, f"test_frames_{test_name}.mp4")
    # # test_frames_video = os.path.join(output_dir, f"aligned_Test-2_1280_720.mp4")
    images_to_video(frames_dir, test_frames_video, fps=1, type='align')


    # Process reference video with object detection
    ref_detected = os.path.join(output_dir, f"ref_detected_{ref_name}.mp4")
    ref_detector = EnhancedVideoObjectDetector()
    ref_detector.process_video(ref_frames_video, ref_detected, save_detections=True)


    # Process test video with object detection
    test_detected = os.path.join(output_dir, f"test_detected_{test_name}.mp4")
    aligned_detector = EnhancedVideoObjectDetector()
    aligned_detector.process_video(test_frames_video, test_detected, 
                                 ref_detections=ref_detector.detections)



    # Create final comparison video
    final_output_tmp = os.path.join(output_dir, f"final_comparison_{ref_name}_vs_{test_name}_tmp.mp4")
    final_output = os.path.join(output_dir, f"final_comparison_{ref_name}_vs_{test_name}.mp4")
    merge_videos(ref_detected, test_detected, final_output_tmp)
    convert_mp4v_to_h264(final_output_tmp, final_output)    

    print(f"Processing complete. Results saved in: {output_dir}")

if __name__ == "__main__":
    main()
    # # ffmpeg -i final_comparison.mp4 -vf "scale=1280:720" final_comparison_1280_720.mp4

