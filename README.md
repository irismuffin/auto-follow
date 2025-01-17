# 智能跟随辅助系统设计课程设计相关资料
本项目中包含“智能跟随辅助系统设计”中的相关工程源文件和实验报告。
## 目录结构
auto-follow/  
├─opencv-cpp/                        # 通过opencv的视觉图像处理识别车牌  
├─video_source_and_frame/            # OV5647摄像头采集的车牌视频以及图像帧  
├─yolov5/                            # yolov5的训练代码，自定义内容在mydata/目录下  
└─yolov5-python/                     # 通过Pytorch和onnx_runtime部署onnx模型  
└─智能机器人设计实验报告_第一组.pdf    # 本课程设计的详细实验报告，其中包含硬件系统结构和程序流程  
## 项目环境
操作系统：Windows10/11、Ubuntu 18.04+  
开发工具：Keil MDK 5.25、cmake 3.8、opencv 3.4.0、python3.8、CUDA11  
调试工具：J-Linkv5.40  
开发平台：地平线旭日X3派
