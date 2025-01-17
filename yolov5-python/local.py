import argparse
import cv2
import numpy as np
import onnxruntime as ort

# 类外定义类别映射关系，使用字典格式
CLASS_NAMES = {
    0: '0',  # 类别 0 名称
    1: '1',  # 类别 1 名称
    2: '2',
    3: '3',
    4: '4'  # 类别 4 名称
    # 可以添加更多类别...
}

import math

class Rotation:
    def __init__(self, x=0, y=0, z=0):
        """
        初始化旋转对象，默认坐标 (x, y, z) 为 (0, 0, 0)
        :param x: 初始坐标的 x 值
        :param y: 初始坐标的 y 值
        :param z: 初始坐标的 z 值
        """
        self.x = x
        self.y = y
        self.z = z

    def rotate_by_z(self, thetaz):
        """
        绕 Z 轴旋转
        :param thetaz: 绕 Z 轴旋转的角度，单位为度
        :return: 旋转后的 (outx, outy) 坐标
        """
        rz = math.radians(thetaz)  # 将角度转换为弧度
        outx = math.cos(rz) * self.x - math.sin(rz) * self.y
        outy = math.sin(rz) * self.x + math.cos(rz) * self.y
        return outx, outy

    def rotate_by_y(self, thetay):
        """
        绕 Y 轴旋转
        :param thetay: 绕 Y 轴旋转的角度，单位为度
        :return: 旋转后的 (outx, outz) 坐标
        """
        ry = math.radians(thetay)  # 将角度转换为弧度
        outx = math.cos(ry) * self.x + math.sin(ry) * self.z
        outz = math.cos(ry) * self.z - math.sin(ry) * self.x
        return outx, outz

    def rotate_by_x(self, thetax):
        """
        绕 X 轴旋转
        :param thetax: 绕 X 轴旋转的角度，单位为度
        :return: 旋转后的 (outy, outz) 坐标
        """
        rx = math.radians(thetax)  # 将角度转换为弧度
        outy = math.cos(rx) * self.y - math.sin(rx) * self.z
        outz = math.cos(rx) * self.z + math.sin(rx) * self.y
        return outy, outz

    def set_coordinates(self, x, y, z):
        """
        设置当前的坐标
        :param x: 新的 x 坐标
        :param y: 新的 y 坐标
        :param z: 新的 z 坐标
        """
        self.x = x
        self.y = y
        self.z = z

    def get_coordinates(self):
        """
        获取当前的坐标
        :return: 当前的 (x, y, z) 坐标
        """
        return self.x, self.y, self.z


class PoseEstimator:
    def __init__(self):
        pass
    
    def is_rotation_matrix(self, R):
        """
        判断一个矩阵是否为旋转矩阵
        """
        Rt = R.T
        should_be_identity = np.dot(Rt, R)
        identity = np.identity(3, dtype=R.dtype)
        return np.allclose(should_be_identity, identity)

    def rotation_matrix_to_euler_angles(self, R):
        """
        将旋转矩阵转换为欧拉角 (Roll, Pitch, Yaw)
        """
        assert self.is_rotation_matrix(R)

        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6  # 如果旋转矩阵接近奇异，表示存在万向节锁

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.atan2(-R[1, 2], R[1, 1])
            y = np.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def rvec_to_euler(self, rvec):
        """
        将旋转向量转换为欧拉角
        """
        # 旋转向量转为旋转矩阵
        rotM, _ = cv2.Rodrigues(rvec)
        
        # 将旋转矩阵转为欧拉角
        euler_angles = self.rotation_matrix_to_euler_angles(rotM)
        return euler_angles

class YOLO5:
    """YOLO5 目标检测模型类，用于处理推理和可视化。"""

    def __init__(self, onnx_model, confidence_thres, iou_thres,camera_xml):
        """
        初始化 YOLO5 类的实例。
        参数：
            onnx_model: ONNX 模型的路径。
            confidence_thres: 用于过滤检测结果的置信度阈值。
            iou_thres: 非极大值抑制（NMS）的 IoU（交并比）阈值。
        """
        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.session = ort.InferenceSession(onnx_model)
        self.camera_xml = camera_xml
        self.load_camera_params()

        # 获取模型的输入信息
        model_inputs = self.session.get_inputs()
        self.input_shape = model_inputs[0].shape
        self.input_width = 320 # 输入图像的宽度
        self.input_height = 320  # 输入图像的高度
        self.classes = CLASS_NAMES

        # 为每个类别生成一个颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))


    def load_camera_params(self):
        """加载相机内参和畸变系数"""
        fs = cv2.FileStorage(self.camera_xml, cv2.FILE_STORAGE_READ)
        self.camera_matrix = fs.getNode("cameraMatrix").mat()
        self.dist_coeffs = fs.getNode("distCoeffs").mat()

        print("Camera Matrix:\n", self.camera_matrix)
        print("Distortion Coefficients:\n", self.dist_coeffs)

    def preprocess(self, frame):
        """
        对输入帧图像进行预处理，以便进行推理。
        返回：处理后的图像数据。
        """
        # 获取输入图像的高度和宽度
        self.img_height, self.img_width = frame.shape[:2]

        # 将图像颜色空间从 BGR 转换为 RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 保持宽高比，进行 letterbox 填充, 使用模型要求的输入尺寸
        img, self.ratio, (self.dw, self.dh) = self.letterbox(img, new_shape=(self.input_width, self.input_height))

        # 通过除以 255.0 来归一化图像数据
        image_data = np.array(img) / 255.0

        # 将图像的通道维度移到第一维
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道优先

        # 扩展图像数据的维度，以匹配模型输入的形状
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        return image_data

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        """
        将图像进行 letterbox 填充，保持纵横比不变，并缩放到指定尺寸。
        """
        shape = img.shape[:2]  # 当前图像的宽高
        print(f"Original image shape: {shape}")

        print(f"Shape: {shape}, New Shape: {new_shape}")
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 选择宽高中最小的缩放比
        if not scaleup:  # 仅缩小，不放大
            r = min(r, 1.0)

        # 缩放后的未填充尺寸
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

        # 计算需要的填充
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算填充的尺寸
        dw /= 2  # padding 均分
        dh /= 2

        # 缩放图像
        if shape[::-1] != new_unpad:  # 如果当前图像尺寸不等于 new_unpad，则缩放
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # 为图像添加边框以达到目标尺寸
        top, bottom = int(round(dh)), int(round(dh))
        left, right = int(round(dw)), int(round(dw))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        print(f"Final letterboxed image shape: {img.shape}")

        return img, (r, r), (dw, dh)

    def postprocess(self, frame, output):
            """
            对模型输出进行后处理，以提取边界框、分数和类别 ID。
            参数： 
                frame: 输入视频帧图像。
                output: 模型的输出。
            返回： 
                frame: 绘制了检测结果的图像。
            """
            # 转置并压缩输出，以匹配预期形状
            outputs = np.transpose(np.squeeze(output[0]))
            rows = outputs.shape[0]
            boxes, scores, class_ids = [], [], []
            ratio = self.img_width / self.input_width, self.img_height / self.input_height

            for i in range(rows):
                classes_scores = outputs[i][4:]
                max_score = np.amax(classes_scores)
                if max_score >= self.confidence_thres:
                    class_id = np.argmax(classes_scores)
                    x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                    # 将框调整到原始图像尺寸，考虑缩放和填充
                    x -= self.dw  # 移除填充
                    y -= self.dh
                    x /= self.ratio[0]  # 缩放回原图
                    y /= self.ratio[1]
                    w /= self.ratio[0]
                    h /= self.ratio[1]
                    left = int(x - w / 2)
                    top = int(y - h / 2)
                    width = int(w)
                    height = int(h)

                    boxes.append([left, top, width, height])
                    scores.append(max_score)
                    class_ids.append(class_id)

            # 使用 cv2.dnn.NMSBoxes 来过滤重复的框（NMS）
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

            # indices 返回的是一个元组，我们需要提取其第一个元素并进行处理
            if len(indices) > 0:
                indices = indices.flatten()  # 提取索引并转换为一维数组

                for i in indices:
                    box = boxes[i]
                    score = scores[i]
                    class_id = class_ids[i]
                    self.draw_detections(frame, box, score, class_id)
                    self.estimate_pose(box, frame)
            
            return frame


    def draw_detections(self, img, box, score, class_id):
        """
        在输入图像上绘制检测到的边界框和标签。
        """
        # 提取边界框的坐标
        x1, y1, w, h = box

        # 获取类别对应的颜色
        color = self.color_palette[class_id]

        # 在图像上绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # 创建包含类别名和分数的标签文本
        label = f"{self.classes[class_id]}: {score:.2f}"

        # 计算标签文本的尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # 计算标签文本的位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # 绘制填充的矩形作为标签文本的背景
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)

        # 在图像上绘制标签文本
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def estimate_pose(self, box, img):
                """使用 `solvePnP` 估计目标物体的位姿"""
                object_points = np.array([
                    [-280 / 2,90 / 2, 0 ],       # 左上角
                    [-280 / 2, -90 / 2, 0],       # 左下角
                    [280 / 2, 90 / 2, 0],       # 右上角
                    [280 / 2, -90 / 2, 0],       # 右下角
                ], dtype=np.float32)
                print("object_points",object_points)

                # 假设这些框的2D点是从框的左上角和右下角获取的
                image_points = np.array([
                    [box[0], box[1]],         # 左上角
                    [box[0], box[1] + box[3]], # 左下角
                    [box[0] + box[2], box[1]], # 右上角
                    [box[0] + box[2], box[1] + box[3]], # 右下角
                ], dtype=np.float32)
                print("image_points",image_points)
        
                # 使用 `solvePnP` 估计位姿
                retval, rvec, tvec = cv2.solvePnP(object_points, image_points, self.camera_matrix, self.dist_coeffs)

                if retval:
                    # 显示旋转矩阵和平移向量
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    print("Rotation Matrix:\n", rotation_matrix)
                    print("Translation Vector:\n", tvec)

                    pose_estimator = PoseEstimator()
                    euler = pose_estimator.rvec_to_euler(rvec)

                    _theta_x = euler[0] * 180 / math.pi
                    _theta_y = euler[1] * 180 / math.pi
                    _theta_z = euler[2] * 180 / math.pi

                    tx, ty, tz = tvec[0, 0], tvec[1, 0], tvec[2, 0]
                    x, y, z = tx, ty, tz
                    rotation = Rotation(x, y, z)
                    # 绕 Z 轴旋转 90 度
                    x, y = rotation.rotate_by_z(_theta_z)
                    print(f"绕 Z 轴旋转后的坐标: x = {x}, y = {y}")
                    # 绕 Y 轴旋转 90 度
                    x, z = rotation.rotate_by_y(_theta_y)
                    print(f"绕 Y 轴旋转后的坐标: x = {x}, z = {z}")
                    # 绕 X 轴旋转 90 度
                    y, z = rotation.rotate_by_x(_theta_x)
                    print(f"绕 X 轴旋转后的坐标: y = {y}, z = {z}")

                    _Cx = x * -1;
                    _Cy = y * -1 - 400;
                    _Cz = z * -1;
                    print("x:",x," y:",y," z:",z)

                    # 可以根据需要绘制3D坐标轴等
                    self.draw_axis(img, rvec, tvec, _Cx, _Cy, _Cz)


    def draw_axis(self, img, rvec, tvec ,_Cx, _Cy, _Cz):
        """在图像中绘制3D坐标轴"""
        # 定义3D坐标轴的点（长度为1的X、Y、Z轴）
        axis = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100]])  # 3D轴的点
        
        # 将3D点投影到2D图像
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        
        # 图像中的原点位置
        origin = tuple(np.round(imgpts[0].ravel()).astype(int))  # 获取投影的第一个点（原点）
        print("imgpts",imgpts)
        print("origin",origin)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"_Cx: {_Cx:.2f}", (50, 50), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"_Cy: {_Cy:.2f}", (50, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"_Cz: {_Cz:.2f}", (50, 150), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # 绘制坐标轴线
        # 绘制 X 轴（红色）
        # cv2.line(img, origin, tuple(np.round(imgpts[0].ravel()).astype(int)), (255, 0, 0), 3)

        # 绘制 Y 轴（绿色）
        # cv2.line(img, origin, tuple(np.round(imgpts[1].ravel()).astype(int)), (0, 255, 0), 3)

        # 绘制 Z 轴（蓝色）
        # cv2.line(img, origin, tuple(np.round(imgpts[2].ravel()).astype(int)), (0, 0, 255), 3)



    def process_video(self, video_path):
        """实时处理视频流"""
        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 预处理当前帧
            img_data = self.preprocess(frame)

            # 运行 ONNX 推理
            outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_data})

            # 后处理并绘制检测框
            frame_with_boxes = self.postprocess(frame, outputs)

            # 显示实时结果
            cv2.imshow("Real-time Detection", frame_with_boxes)

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 创建参数解析器以处理命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov5_320.onnx", help="输入你的 ONNX 模型路径。")
    parser.add_argument("--video", type=str, default="./video.avi", help="输入视频文件的路径。")
    parser.add_argument("--conf-thres", type=float, default=0.6, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--camera_xml", type=str, default="./camera.xml", help="camera_xml")
    args = parser.parse_args()

    # 使用指定的参数创建 YOLO5 类的实例
    detection = YOLO5(args.model, args.conf_thres, args.iou_thres,args.camera_xml)

    # 执行视频流目标检测
    detection.process_video(args.video)
