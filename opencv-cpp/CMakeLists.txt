cmake_minimum_required(VERSION 2.8)
project(auto-follow)

set(CMAKE_CXX_STANDARD 11)

#set(OpenCV_DIR /home/leung/Downloads/opencv-3.4.0/build)
find_package( OpenCV REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(auto-follow src/main.cpp src/findplate.cpp src/findplate.h
        src/common/rvec2elur.h src/common/rvec2elur.cpp
        src/common/codeRotate.h src/common/rotate.cpp src/park/Park.h
        src/park/Park.cpp serial/Serial.h serial/Serial.cpp src/auto_drive/autodrive.h src/auto_drive/autodrive.cpp)

target_link_libraries(auto-follow ${OpenCV_LIBS} )
