cmake_minimum_required(VERSION 2.8)
project(camera_calibration)

set(CMAKE_CXX_STANDARD 11)

# set(OpenCV_DIR "<your opencv install path>")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES camera_calibration.cpp)

add_executable(camera_calibration ${SOURCE_FILES})
target_link_libraries(camera_calibration ${OpenCV_LIBS})
