#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    // 读取 XML 配置文件
    FileStorage fs("default.xml", FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Could not open the configuration file!" << endl;
        return -1;
    }else{
        cerr << "FileSource open successfuly" << endl;
    // 读取配置信息
    int boardWidth, boardHeight, inputType, inputFlip, inputDelay, nrFramesToUse;
    bool fixAspectRatio, assumeZeroTangentialDistortion, fixPrincipalPoint;
    string outputFileName;
    bool showUndistortedImage;
    string inputImageList;

    // 从 XML 文件中读取参数
    fs["Settings.BoardSize_Width"] >> boardWidth;
    fs["Settings.BoardSize_Height"] >> boardHeight;
    fs["Settings.Input"] >> inputImageList;  // 获取图像路径列表文件路径
    fs["Settings.Input_FlipAroundHorizontalAxis"] >> inputFlip;
    fs["Settings.Input_Delay"] >> inputDelay;
    fs["Settings.Calibrate_NrOfFrameToUse"] >> nrFramesToUse;
    fs["Settings.Calibrate_FixAspectRatio"] >> fixAspectRatio;
    fs["Settings.Calibrate_AssumeZeroTangentialDistortion"] >> assumeZeroTangentialDistortion;
    fs["Settings.Calibrate_FixPrincipalPointAtTheCenter"] >> fixPrincipalPoint;
    fs["Settings.Write_outputFileName"] >> outputFileName;
    fs["Settings.Show_UndistortedImage"] >> showUndistortedImage;

    // 初始化相机内参和畸变系数
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    Mat distCoeffs = Mat::zeros(5, 1, CV_64F);

    // 检查是否需要翻转图像
    bool flipVertical = inputFlip == 1;

    // 图像列表读取
    vector<string> imagePaths;
    FileStorage fsImages(inputImageList, FileStorage::READ);
    if (!fsImages.isOpened()) {
        cerr << "Could not open the image list file!" << endl;
        return -1;
    }
    FileNode imagesNode = fsImages["images"];
    for (FileNodeIterator it = imagesNode.begin(); it != imagesNode.end(); ++it) {
        imagePaths.push_back((string)(*it));
    }

    // 校准参数
    vector<vector<Point2f>> imagePoints;
    Size boardSize(boardWidth, boardHeight);

    // 校准图像尺寸
    Size imageSize;

    // 循环读取图像并进行角点检测
    for (size_t i = 0; i < imagePaths.size() && imagePoints.size() < nrFramesToUse; i++) {
        Mat img = imread(imagePaths[i]);

        if (img.empty()) {
            cerr << "Failed to load image: " << imagePaths[i] << endl;
            continue;
        }

        // 如果需要，翻转图像
        if (flipVertical) {
            flip(img, img, 0);
        }

        imageSize = img.size(); // 更新图像尺寸

        // 检测棋盘格角点
        vector<Point2f> pointBuf;
        bool found = findChessboardCorners(img, boardSize, pointBuf,
                                           CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

        if (found) {
            // 提高角点精度
            Mat gray;
            cvtColor(img, gray, COLOR_BGR2GRAY);
            cornerSubPix(gray, pointBuf, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

            // 保存角点
            imagePoints.push_back(pointBuf);

            // 在图像上绘制角点
            drawChessboardCorners(img, boardSize, Mat(pointBuf), found);

            // 显示图像
            imshow("Calibration", img);
            waitKey(inputDelay); // 延迟
        }

        // 如果收集足够的图像，退出
        if (imagePoints.size() >= (size_t)nrFramesToUse) {
            break;
        }
    }

    // 相机标定
    if (imagePoints.size() >= (size_t)nrFramesToUse) {
        vector<vector<Point3f>> objectPoints;
        vector<Point3f> obj;
        for (int i = 0; i < boardHeight; i++) {
            for (int j = 0; j < boardWidth; j++) {
                obj.push_back(Point3f(j * 20.0f, i * 20.0f, 0.0f));  // 假设每个棋盘格的方格大小为 20 单位
            }
        }

        for (size_t i = 0; i < imagePoints.size(); i++) {
            objectPoints.push_back(obj);
        }

        // 标定相机
        calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, noArray(), noArray(),
                        (fixAspectRatio ? CALIB_FIX_ASPECT_RATIO : 0) |
                        (assumeZeroTangentialDistortion ? CALIB_ZERO_TANGENT_DIST : 0) |
                        (fixPrincipalPoint ? CALIB_FIX_PRINCIPAL_POINT : 0));

        // 保存标定结果
        FileStorage fsOut(outputFileName, FileStorage::WRITE);
        fsOut << "cameraMatrix" << cameraMatrix;
        fsOut << "distCoeffs" << distCoeffs;
        fsOut.release();
        cout << "Calibration finished and saved to " << outputFileName << endl;
    } else {
        cerr << "Not enough frames for calibration!" << endl;
        return -1;
    }

    return 0;
}

