import cv2
import numpy as np

# 1. 读取 XML 格式的相机参数文件
camera_xml = "camera.xml"  # 替换为你的 XML 文件路径

# 使用 FileStorage 加载 XML 文件
fs = cv2.FileStorage(camera_xml, cv2.FILE_STORAGE_READ)

# 从 XML 文件中提取相机矩阵和畸变系数
camera_matrix = fs.getNode("cameraMatrix").mat()  # 读取 cameraMatrix
dist_coeffs = fs.getNode("distCoeffs").mat()  # 读取 distCoeffs

# 打印加载的相机矩阵和畸变系数
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# 2. 定义3D点（世界坐标系下）
object_points = np.array([
    [0, 0, 0],       # 点1
    [1, 0, 0],       # 点2
    [0, 1, 0],       # 点3
    [1, 1, 0],       # 点4
], dtype=np.float32)

# 3. 定义2D点（图像坐标系下）
image_points = np.array([
    [100, 200],      # 点1在图像中的坐标
    [200, 200],      # 点2在图像中的坐标
    [100, 300],      # 点3在图像中的坐标
    [200, 300],      # 点4在图像中的坐标
], dtype=np.float32)

# 4. 计算旋转向量和平移向量
retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

# 5. 输出旋转向量和平移向量
print("Rotation Vector:\n", rvec)
print("Translation Vector:\n", tvec)

# 6. 将旋转向量转换为旋转矩阵
rotation_matrix, _ = cv2.Rodrigues(rvec)
print("Rotation Matrix:\n", rotation_matrix)

# 7. 使用投影函数将3D点投影回2D图像
projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)

# 假设你有一张图像用于绘制投影点
image = np.zeros((480, 640, 3), dtype=np.uint8)  # 创建一张空白图像

# 绘制投影点
for point in projected_points:
    x, y = point[0]
    cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

# 显示图像
cv2.imshow('Projection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
