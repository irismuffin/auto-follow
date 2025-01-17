cmake_minimum_required(VERSION 2.8)
project(test_cam)

set(CMAKE_CXX_STANDARD 11)

# set(OpenCV_DIR "<your opencv install path>")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE_FILES test_cam.cpp)

add_executable(test_cam ${SOURCE_FILES})
target_link_libraries(test_cam ${OpenCV_LIBS})
