import cv2
import os

# 输入视频文件路径
video_path = './video_get/5.avi'
# 输出目录，保存每一帧
output_dir = 'frames5/'

# 创建输出目录（如果没有的话）
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

frame_count = 0
saved_frame_count = 0

while True:
    # 读取每一帧
    ret, frame = cap.read()

    # 如果帧读取成功
    if not ret:
        break

    # 每3帧保存一次
    if frame_count % 3 == 0:
        # 保存每一帧为图像文件，格式为jpg
        frame_filename = os.path.join(output_dir, f"frame_{saved_frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_frame_count += 1

    frame_count += 1

# 释放视频捕获对象
cap.release()

print(f"Finished saving {saved_frame_count} frames.")

