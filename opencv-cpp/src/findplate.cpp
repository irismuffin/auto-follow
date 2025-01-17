#include "findplate.h"

using namespace std;
using namespace cv;
bool FindLicense::initial()
{
    //加载相机参数
    FileStorage fs("./para/camera.xml", FileStorage::READ);
    if (!fs.isOpened())
    {
        cout << "findplate : can't open xml file" << endl;
        return false;
    }
    fs["cameraMatrix"] >> cameraMatrix;
    fs["distCoeffs"] >> distCoeffs;
    fs.release();
    cout << "cameraMatrix" << cameraMatrix << endl
         << "distCoeffs" << distCoeffs << endl;

    //设置世界坐标
    _Point3D.push_back(Point3f(-280 / 2, 0, 90 / 2)); //x,y,z
    _Point3D.push_back(Point3f(-280 / 2, 0, -90 / 2));
    _Point3D.push_back(Point3f(280 / 2, 0, -90 / 2));
    _Point3D.push_back(Point3f(280 / 2, 0, 90 / 2));

//    cout << "load the svm_data.xml..." << endl;
//    svm.load("../para/SVM_DATA.xml");
    cout << "end" << endl;
    return true;
}

bool FindLicense::getlicense(cv::Mat &src, cv::Point3d &point, cv::Point3d &theta)
{
    time0 = static_cast<double>(getTickCount());
    this->get_gray(src);
    this->get_license();
    double distance = this->GetUltraSonic();//cm
    cout << "[info] distance :" << distance << "cm" << endl;
    if(distance <= 20.00)
	    return false;
    if (!this->solve_pnp())
        return false;
    point.x = _Cx;
    point.y = _Cy;
    point.z = _Cz;
    theta.x = _theta_x;
    theta.y = _theta_y;
    theta.z = _theta_z;
    return true;
}

void FindLicense::get_gray(cv::Mat &src)
{
	//cout << "[info]get_gray" << endl; 
    srcImage = src.clone();
    cv::Mat hsvImage;
    cvtColor(srcImage, hsvImage, COLOR_BGR2HSV);
//imwrite("./pics/hsvImage.jpg",hsvImage);
    //    vector<Mat> hsvSplit;
    //    split(hsvImage, hsvSplit);
    //    equalizeHist(hsvSplit[2], hsvSplit[2]);
    //    merge(hsvSplit, hsvImage);
    //    int iLowH = 67, iLowS = 14, iLowV = 0, iHighH = 135, iHighS = 255, iHighV = 255;
    //    inRange(hsvImage, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), binaryImage); //Threshold the image																							   //消除噪声点
    //inRange(hsvImage, Scalar(110,70, 0), Scalar(130, 255, 100), binaryImage);
inRange(hsvImage, Scalar(100,43, 46), Scalar(124, 255, 255), binaryImage);
    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
//    morphologyEx(binaryImage, binaryImage, MORPH_OPEN, element); //开操作
threshold(binaryImage, binaryImage, 127, 255, cv::THRESH_BINARY);//二值化
morphologyEx(binaryImage, binaryImage, MORPH_CLOSE, element);//闭操作
	//imwrite("./pics/binaryImage.jpg",binaryImage);
    //imshow("binaryImage", binaryImage);
}

void FindLicense::get_license()
{
//	cout << "[info]get_license" << endl; 
    license_rects.clear();
    roi_mats.clear();
    select_count.clear();
    //    contours.clear();
    std::vector<cv::RotatedRect> selected_rect;
    std::vector<int> count_contours;
	std::vector<std::vector<cv::Point>> contours;

	Mat hierarchy;
    //findContours(binaryImage, count_contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
//cout << "channels: " << binaryImage.channels() << "depth: " << binaryImage.depth() << endl;
    findContours(binaryImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//cout << "[info]findContours" << endl; 
//imwrite("./pics/hierarchy.jpg",hierarchy);
    Mat contours_show = Mat(binaryImage.rows,binaryImage.cols,CV_8UC3,Scalar(0,0,0));
//imwrite("./pics/contours_show_og.jpg",contours_show);
//cout << "[info]contours nums : " << contours.size() << endl; 
Mat single;
single = contours_show.clone();
    for (size_t i = 0; i < contours.size(); i++)
    {
        drawContours(contours_show, contours, i, Scalar(0,0,255), 1); //绘制轮廓
	
        //最小包围矩形
        RotatedRect rect = minAreaRect(contours[i]); //最小包围矩形
        Point2f vertices[4];
        rect.points(vertices);
        //顶点是否在边界
        int isborder = 0;
        int distance = 10;
        for (int i = 0; i < 4; i++)
        {
            if (vertices[i].x <= distance || (srcImage.cols - vertices[i].x) <= distance
                || vertices[i].y <= distance ||  (srcImage.rows - vertices[i].y <= distance))
                    isborder++;
        }
        if (isborder > 1)
            continue;
        //通过面积、长宽比排除
//cout << "[info]rect_detect "<< i << endl;
        double rect_area = rect.size.area(); //计算面积
        double rect_height = rect.size.height;
        double rect_width = rect.size.width;
        double ratio = rect_width / rect_height;    //计算宽高比
//cout << "rect_area : "<< rect_area << endl;
//cout << "rect_height : "<< rect_height << endl;
//cout << "rect_width : "<< rect_width << endl;
//cout << "ratio : "<< ratio << endl;
        ratio = ratio < 1 ? 1 / ratio : ratio;      //保证比值大于1
        double standard = 3.15;                     // 286 / 90;	//标准车牌宽高比 约3.15
        double error = 0.6;                         //TODO error
        //if (rect_area > 800 && rect_area <= 200000) //TODO rect_area
	if (rect_area > 800 && rect_area <= 800000)
        {
            if (ratio >= standard * (1 - error) && ratio <= standard * (1 + error))
            {
                if (fabs(rect.angle) < 30 && rect_width > rect_height || fabs(rect.angle) > 60 && rect_width < rect_height)
                {
                    //drawContours(contours_show, contours, i, Scalar(255,0,0), 1); //绘制轮廓
/*
	drawContours(single, contours, i, Scalar(255,0,0), 1); 
   	std::ostringstream oss;
  	oss << "./pics/single_" << std::setw(2) << std::setfill('0') << i << ".jpg";
   	std::string path = oss.str();
	imwrite(path,hierarchy);
*/
                    selected_rect.push_back(rect); //将符合长宽比要求的矩形区域筛选出来
                    count_contours.push_back(i);
                    //                    for (int i = 0; i < 4; i++) {
                    //                        line(binaryImage, vertices[i], vertices[(i + 1) % 4], Scalar(255), 2);
                    //                    }
                    //                    cout<<"ratio= "<<ratio<<" "<<"rect_area= "<<rect_area<<endl;
                }
            }
        }
    }
//imwrite("./pics/contours_show.jpg",contours_show);
    //imshow("contours_show", contours_show);
    if (selected_rect.empty())
    {
        return;
    }

    //通过矩形内轮廓数及长度排除
    else
    {
//cout << "[info]ROI" << endl; 
        Mat ROIsrc;
        Point2f vertices[4];
        for (size_t i = 0; i < selected_rect.size(); i++)
        {
            selected_rect[i].points(vertices);
            int rect_width = (int)min(selected_rect[i].size.height, selected_rect[i].size.width); //当前候选区宽度
            //cout << "rect_width" << rect_width << endl;
            Size rect_size = selected_rect[i].size;
            if (rect_size.width < rect_size.height)
            {
                swap(rect_size.width, rect_size.height);
            }
            getRectSubPix(srcImage, rect_size, selected_rect[i].center, ROIsrc);
//imwrite("./pics/ROIsrc.jpg", ROIsrc);
            //            imshow("ROIsrc", ROIsrc);

            //对ROI区域进行检测
            Mat ROIgray;
            cvtColor(ROIsrc, ROIgray, COLOR_BGR2GRAY);
            threshold(ROIgray, ROIgray, 0, 255, THRESH_OTSU);
            vector<vector<Point>> contours;
            findContours(ROIgray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            for (size_t i = 0; i<contours.size(); i++)
             {
                 drawContours(ROIgray, contours, i, Scalar(255), 1); // 绘制轮廓
             }
//imwrite("./pics/ROIgray.jpg", ROIgray);
            // imshow("ROIgray", ROIgray);
            int fitcount = 0;
            for (size_t i = 0; i != contours.size(); i++)
            {
                if (contours[i].size() > rect_width * 0.5)
                    fitcount++;
            }
//cout << "[info]fticount : " << fitcount << endl;
            if (fitcount >= 5)
            {
                Point2f draw_point[4];
                selected_rect[i].points(draw_point);
                for(int i =0;i<4;i++)
                    line(srcImage,draw_point[i],draw_point[(i+1)%4],Scalar(0,255,0),2);
//imwrite("./pics/srcImage.jpg", srcImage);
                //imshow("src&target",srcImage);
                license_rects.push_back(selected_rect[i]);
                roi_mats.push_back(ROIsrc);
                select_count.push_back(count_contours[i]); //记录下轮廓的位置,方便后面进行找四个顶点　
                //string adfile = "../write/";
                //time_t nowtime;
                //nowtime = time(NULL); //获取日历时间
                //imwrite(adfile + to_string(nowtime) + to_string(i) + ".jpg", ROIsrc);
            }
        }
    }
    plate_points.clear();
    if(!license_rects.empty())
    {
        int area_max = license_rects[0].size.width*license_rects[0].size.height;
        int index = 0;
        for(int i =1;i<license_rects.size();i++)
        {
            if(area_max < license_rects[i].size.width*license_rects[i].size.height)
            {
                index = i;
                area_max = license_rects[i].size.width*license_rects[i].size.height;
            }
        }
        Point2f temp[4];
        license_rects[index].points(temp);
        if(license_rects[index].size.width < license_rects[index].size.height)
        {
            plate_points.push_back(temp[1]);
            plate_points.push_back(temp[2]);
            plate_points.push_back(temp[3]);
            plate_points.push_back(temp[0]);
        }
        else
        {
            plate_points.push_back(temp[0]);
            plate_points.push_back(temp[1]);
            plate_points.push_back(temp[2]);
            plate_points.push_back(temp[3]);
        }
    }
}



bool FindLicense::solve_pnp()
{
    //cout<<"[info]solve_pnp"<<endl;
    if (plate_points.size() != 4)
    {
        //        cout << "plate_points.size() != 4" << endl;
        return false;
    }
    //    if(isChanged== false){
    //        cout<<"不需要更改"<<endl;
    //        return true;
    //    }
    //    cout<<"需要更改"<<endl;
    //    cout<<"四点: "<<" ("<<plate_points[0].x<<","<<plate_points[0].y<<")"<<
    //        " ("<<plate_points[1].x<<","<<plate_points[1].y<<")"<<
    //        " ("<<plate_points[2].x<<","<<plate_points[2].y<<")"<<
    //        " ("<<plate_points[3].x<<","<<plate_points[3].y<<")"<<endl;
    //特征点图像坐标：
    vector<cv::Point2d> Points2D;
    Points2D.push_back(plate_points[0]); //P1  左下 单位是像素
    Points2D.push_back(plate_points[1]); //P2  左上
    Points2D.push_back(plate_points[2]); //P3  右上
    Points2D.push_back(plate_points[3]); //P4  右下
#ifdef DEBUG
    draw_text(srcImage, plate_points[0], 1, Scalar(200, 100, 100));
    draw_text(srcImage, plate_points[1], 2, Scalar(200, 100, 100));
    draw_text(srcImage, plate_points[2], 3, Scalar(200, 100, 100));
    draw_text(srcImage, plate_points[3], 4, Scalar(200, 100, 100));
#endif
//imwrite("./pics/Points2D.jpg", srcImage);
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F); //旋转矩阵
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F); //平移矩阵
    solvePnP(_Point3D, plate_points, cameraMatrix, distCoeffs, rvec, tvec, false, SOLVEPNP_ITERATIVE);
//cout<<"[info]func -- solvePNP"<<endl;
    vector<Point3d> center_point;
    //center_point.push_back(Point3f(-280 / 2.0, 10, 90 / 2.0)); //TODO 修改参数
    center_point.push_back(Point3f(0, 10, 0));
    vector<Point2d> image_po;
    projectPoints(center_point, rvec, tvec, cameraMatrix, distCoeffs, image_po);
//cout<<"[info]func -- projectPoints"<<endl;
    //#ifdef DEBUG
        line(srcImage, plate_points[0], image_po[0], Scalar(80, 100, 85), 3, 8);

    //    cout <<" line!!" << endl;

    //#endif
    cv::Vec3f elur = rvec2elur(rvec);

    _theta_x = elur[0] * 180 / CV_PI;
    _theta_y = elur[1] * 180 / CV_PI;
    _theta_z = elur[2] * 180 / CV_PI;

    double tx = tvec.ptr<double>(0)[0];
    double ty = tvec.ptr<double>(0)[1];
    double tz = tvec.ptr<double>(0)[2];
    double x = tx, y = ty, z = tz;

    codeRotateByZ(x, y, -1 * _theta_z, x, y);
    codeRotateByY(x, z, -1 * _theta_y, x, z);
    codeRotateByX(y, z, -1 * _theta_x, y, z);
    _Cx = x * -1;
    _Cy = y * -1 - 400;
    _Cz = z * -1;

    time0 = ((double)(getTickCount()) - time0) / getTickFrequency();
    ostringstream centerText, distanceText, angleText, timeText;
    centerText << "Vehicle is " << abs(_Cx) << " mm";
    if (_Cx < 0)
    {
        centerText << " left";
    }
    else
    {
        centerText << " right";
    }
    distanceText << "The distance is " << abs(_Cy) << " mm";
    angleText << "The angle is " << _theta_y << " du";
    timeText << "Time is " << time0 * 1000 << " ms";
    int font_face = FONT_HERSHEY_COMPLEX;
    putText(srcImage, centerText.str(), Point(25, 25), font_face, 0.7, Scalar(255, 0, 0), 1, CV_AA);
    putText(srcImage, distanceText.str(), Point(25, 75), font_face, 0.7, Scalar(255, 0, 0), 1, CV_AA);
    putText(srcImage, angleText.str(), Point(25, 125), font_face, 0.7, Scalar(255, 0, 0), 1, CV_AA);
    putText(srcImage, timeText.str(), Point(25, 175), font_face, 0.7, Scalar(255, 0, 0), 1, CV_AA);

    //    draw_text(srcImage,Point(20,10),int(_Cx));
    //    draw_text(srcImage,Point(20,30),int(_Cy));
    //    draw_text(srcImage,Point(20,50),int(_Cz));
    //
    //    draw_text(srcImage,Point(80,10),int(_theta_x),Scalar(56,180,34));
    //    draw_text(srcImage,Point(80,30),int(_theta_y),Scalar(56,180,34));
    //    draw_text(srcImage,Point(80,50),int(_theta_z),Scalar(56,180,34));
	//imwrite("./pics/result.jpg", srcImage);
	//imshow("result",srcImage);
	//cout<<"[info]solve_pnp end"<<endl;
    	//imshow("result", srcImage);
    return true;
}
void FindLicense::draw_text(Mat img, Point p, float num, Scalar scalar)
{
    ostringstream ss;
    ss << num;
    int font_face = FONT_HERSHEY_COMPLEX;
    double font_scale = 0.6;
    int thickness = 2;
    int baseline;
    Size text_size = getTextSize(ss.str(), font_face, font_scale, thickness, &baseline);

    Point origin;
    origin.x = p.x;
    origin.y = p.y + text_size.height;
    putText(img, ss.str(), origin, font_face, font_scale, scalar, thickness, 8, 0);
}
double FindLicense::GetUltraSonic()
{
// 调用 Python 脚本并获取结果
    string command = "python3 ultrasonic.py";  // 假设 Python 脚本为 script.py
    string dis1 = "";
    string dis2 = "";
    string result = "";
    char buffer[128];
    FILE* pipe = popen(command.c_str(), "r");
    
    if (!pipe) {
        std::cerr << "Failed to run command" << std::endl;
        return 1;
    }
    
    // 获取 Python 脚本的输出
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    
    fclose(pipe);

	istringstream stream(result); // 使用istringstream处理输入字符串
    	stream >> dis1 >> dis2;
    // cout << "dis1: " << dis1 << endl;
    // cout << "dis2: " << dis2 << endl;
    // 进一步处理输出的数据，假设是一个数字
    double result_dis1 = stod(dis1); // 假设 Python 输出的是一个数值
    double result_dis2 = stod(dis2); // 假设 Python 输出的是一个数值
    // cout << "Converted result: " << result_dis1 << result_dis2<< endl;
    return min(result_dis1,result_dis2);
}
