#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>

int main() {
    // 设置保存图片的目录
    std::string save_dir = "/home/sunrise/Desktop/auto-follow/calibration/test_cam/pics";
    std::string xml_file = "pics_list.xml";
    int num_images = 10; // 设置需要捕获的图片数量
/*
    // 创建保存图片的目录（使用 sys/stat.h）
    if (mkdir(save_dir.c_str(), 0777) == -1 && errno != EEXIST) {
        std::cerr << "Error: Failed to create directory: " << save_dir << std::endl;
        return -1;
    }
*/
    // 打开摄像头设备（ID 8）
    cv::VideoCapture capture(8); // 也可以使用 "/dev/video8" 来指定设备路径
    if (!capture.isOpened()) {
        std::cerr << "Error: Could not open camera device /dev/video8" << std::endl;
        return -1;
    }
					 //
    // 创建 XML 文件，保存图片列表
    std::ofstream xml_output(xml_file);
    if (!xml_output.is_open()) {
        std::cerr << "Error: Could not open XML file for writing" << std::endl;
        return -1;
    }

    // 写入 XML 文件头
    xml_output << "<?xml version=\"1.0\"?>\n";
    xml_output << "<opencv_storage>\n";
    xml_output << "  <images>\n";

    cv::Mat frame;
    int skip_frames = 20;  // 要舍弃的帧数 
    int delay_ms = 30;
    for (int i = 1; i <= num_images + skip_frames; ++i) {
        if (i <= skip_frames) {
      	  continue;  // 跳过这10帧
	}
	    // 捕获一帧
        cv::waitKey(delay_ms);  // 延时30毫秒
        capture >> frame;
        if (frame.empty()) {
            std::cerr << "Error: Failed to capture image" << std::endl;
            continue;
        }

        // 保存图片到文件夹
        std::string image_path = save_dir + "/image_" + std::to_string(i-skip_frames) + ".jpg";
        if (cv::imwrite(image_path, frame)) {
            std::cout << "Captured image: " << image_path << std::endl;

            // 将图片路径写入 XML 文件
            xml_output << "    <image>" << image_path << "</image>\n";
        } else {
            std::cerr << "Error: Failed to save image " << i << std::endl;
        }
    }

    // 结束 XML 文件
    xml_output << "  </images>\n";
    xml_output << "</opencv_storage>\n";

    // 关闭文件
    xml_output.close();

    std::cout << "Capture complete. " << num_images << " images saved to '" << save_dir << "' and picture list saved to '" << xml_file << "'." << std::endl;

    return 0;
}

