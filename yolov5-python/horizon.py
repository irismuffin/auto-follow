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


class YOLO5:
    """YOLO5 目标检测模型类，用于处理推理和可视化。"""

    def __init__(self, onnx_model, confidence_thres, iou_thres):
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

        # 获取模型的输入信息
        model_inputs = self.session.get_inputs()
        self.input_shape = model_inputs[0].shape
        self.input_width = 320 # 输入图像的宽度
        self.input_height = 320  # 输入图像的高度
        self.classes = CLASS_NAMES

        # 为每个类别生成一个颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

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
    parser.add_argument("--conf-thres", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU 阈值")
    args = parser.parse_args()

    # 使用指定的参数创建 YOLO5 类的实例
    detection = YOLO5(args.model, args.conf_thres, args.iou_thres)

    # 执行视频流目标检测
    detection.process_video(args.video)
