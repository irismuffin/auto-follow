#include "autodrive.h"

Autodrive::Autodrive(){
    if(!park.initial()){
        cout<<"park initial fail"<<endl;
        exit(-1);
    }

#ifdef SERIAL
    if(!serial.open_port()) {
        cout << "open port fail" << endl;
        exit(-1);
    }else{
    	cout << "open port" << endl;
    }
    FILE* pipe = popen("bash ./serial/check_serial.sh", "r");
    //const char* scriptPath = "./serial/check_serial.sh";

    // 调用 system 函数执行脚本
    //int result = system(scriptPath);
#else
    cout << "fail serial" << endl;
#endif // !end SERIAL
#ifdef LOCAL
    cout << "local debug init" << endl;
#else
    if(!capture.open(8,CAP_V4L2)){
        //if(!capture.open(1)) {
        if(!capture.isOpened()) {
            cout << "capture open fail" << endl;
            exit(-1);
        }
    }
#endif
    waitKey(200);
    cout<<"initial OK!!!"<<endl;
}

void Autodrive::run() {
       doPark();
}


void Autodrive::doPark() {
    int flag=0;
    if(!plate.initial())  // check if we succeeded
        return;
#ifdef LOCAL
	srcImage = imread("./pics/plate.jpg");
	//imshow("origin",srcImage);
	cout << "[info]get test pic" << endl;
	plate.getlicense(srcImage,point,theta);
	flag = 0;
        int x, z;
        x = static_cast<int>(point.x / 10);
        z = static_cast<int>(point.y / 10);
	cout<< "x = "<<x<<" y = "<<z<<endl;

#else
    while (waitKey(1)!=27) {
        capture>>srcImage;
	//imwrite("./pics/origin.jpg",srcImage);
	//imshow("origin",srcImage);
#ifdef TASK_DEBUG
		flag=0;
            int x, y, w;
            x = 30;
            y = 30;
	    w = 20;
	    cout<<"x = "<<x<<endl;
            cout<<"y = "<<y<<endl;
            cout<<"w= "<<w<<endl;
            serial.send_AP(w,-y,x);
                

#else
        if (!plate.getlicense(srcImage,point,theta)) {
            flag++;
            if(flag>=5){
		cout<<"no target"<<endl;
                 serial.send_AP(0,0,0);
                 serial.send_AP(0,0,0);
            }
            continue;
        }
        else {
		flag=0;
            int x, y;
	    double w;
            x = static_cast<int>(point.x / 10)*2;
            y = static_cast<int>(point.y / 10);
	    w = 0;
            cout<<"x = "<<x<<endl;
            cout<<"y = "<<y<<endl;
            cout<<"theta.y = "<<theta.y<<endl;
            cout << "abs w:" << abs(theta.y);
            if (abs(theta.y) < 3) {
                serial.send_AP(w+20,-y,-x);
            }
            else{
                //serial.send_AP(-theta.y-45,-y,x);
                serial.send_AP(w+20,-y,-x);
                cout << "send:" << -theta.y-45 << endl;
            }
            //cout << "park done !!!" << endl;
            //serial.send_AP(0,0,0);
            //state = Nothing;
            //return 1;

/*
	    flag = 0;
            int x, z;
            x = static_cast<int>(point.x / 10);
            z = static_cast<int>(point.y / 10);
	    cout<< "x = "<<x<<" z = "<<z<<endl;
            if (abs(theta.y) < 3) {
                serial.send_AP(0,x,z);
                serial.send_AP(0,x,z);
            }
            else{
                serial.send_AP(theta.y,x,z);
                serial.send_AP(theta.y,x,z);
            }
*/
        }
#endif
    }
#endif
}

void Autodrive::doNothing() {
    while(true){
        std::cout<<"waiting for command"<<std::endl;
        int n=serial.receive();
        if(n>0 && (serial.buf[0]=='c')){
            state=CPark;
            break;
        }
    }
}
