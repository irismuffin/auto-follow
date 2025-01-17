#include <opencv2/opencv.hpp>

int main() {
    // 使用摄像头 ID 1
    cv::VideoCapture cap(8);  // 打开ID为1的摄像头
    if (!cap.isOpened()) {
        std::cerr << "Error: Camera with ID 8 not found!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;  // 从摄像头捕获一帧
        if (frame.empty()) {
            std::cerr << "Error: Could not grab frame!" << std::endl;
            break;
        }

        cv::imshow("Camera 8", frame);  // 显示捕获的帧
        if (cv::waitKey(1) == 27) {  // 按Esc键退出
            break;
        }
    }

    cap.release();  // 释放摄像头
    cv::destroyAllWindows();  // 关闭所有窗口
    return 0;
}

