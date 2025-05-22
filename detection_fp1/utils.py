import subprocess

def convert_mp4v_to_h264(input_path, output_path):
    """
    将 MP4V 编码的视频转换为 H.264 编码的 MP4 文件
    :param input_path: 输入视频路径（MP4V 编码）
    :param output_path: 输出视频路径（H.264 编码）
    """
    command = [
        "ffmpeg",
        "-i", input_path,               # 输入文件
        "-vf", "scale=1200:720",         # 调整宽度为 1200 像素
        "-c:v", "libx264",              # 使用 H.264 编码
        "-pix_fmt", "yuv420p",          # 确保兼容性（Safari 需要）
        "-profile:v", "high",           # 提高兼容性
        # "-movflags", "+faststart",      # 支持流式播放
        # "-c:a", "aac",                  # 音频编码为 AAC
        # "-strict", "-2",                # 允许实验性 AAC 编码
        "-y",                           # 覆盖输出文件（可选）
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        print(f"转换成功！输出文件: {output_path}")
        subprocess.run(["rm", "-rf", input_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"转换失败: {e}")