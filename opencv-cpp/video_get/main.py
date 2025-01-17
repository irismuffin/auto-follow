import cv2

# 打开默认摄像头 (通常是设备的第一个摄像头)
cap = cv2.VideoCapture(8)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法访问摄像头")
    exit()

# 设置视频捕获的帧宽和帧高（可选）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 获取视频帧的宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建 VideoWriter 对象，用于保存视频文件
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

while True:
    # 从摄像头捕获一帧图像
    ret, frame = cap.read()

    # 如果帧读取成功，ret为True
    if not ret:
        print("无法读取帧")
        break

    # 将视频帧写入到文件
    out.write(frame)

    # 这里不再调用 cv2.imshow()，因为不需要实时显示视频帧

    # 等待用户按键，如果按下 'q' 键则退出
    # 由于不显示图像，这里的暂停检测键盘输入就不需要了
    # 退出条件可以根据需要设置，比如捕获一定数量的帧或其他方式

# 释放摄像头和 VideoWriter 对象
cap.release()
out.release()

# 不需要调用 cv2.destroyAllWindows() 因为没有打开任何窗口

