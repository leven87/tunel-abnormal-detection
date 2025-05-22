# Tunel detection
This is a project for tunel abnormal detection. Suppose there are two videos of the same tunnel, one is a reference video and the other is a test video. The videos are shot at the front of a train (the train speed may be inconsistent) to detect whether certain objects in the tunnel are added, missing, or dislocated.

## key-frame alignment and detection
`detection_fp1` implement  key frame alignment and detection.

### Quick start
Use below command：：
```sh
cd detection_fp1
python comparison.py --ref ../Trainingvideos_2_1280_720.mp4 --test ../Test-2_1280_720.mp4
```

## alignment and detection frame by frame
`detection_fp30` implement  alignment and detection frame by frame
### Demo
![demo](./result/final_comparison_Trainingvideos_2_1280_720_vs_Test-2_1280_720_good2.gif)

### Quick start
Use below command：
```sh
cd detection_fp30
python comparison.py --ref ../Trainingvideos_2_1280_720.mp4 --test ../Test-2_1280_720.mp4
```
