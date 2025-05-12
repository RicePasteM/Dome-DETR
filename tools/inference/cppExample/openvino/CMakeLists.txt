cmake_minimum_required(VERSION 3.5.1)

project(
    openvinoExample
    LANGUAGES CXX
    VERSION 1.0.0
)

set(CMAKE_CXX_STANDARD 17)

if (MSVC)
    add_compile_options(-nologo)
    add_definitions(-DNOMINMAX)
else()
    add_compile_options(-Wall)
endif()

find_package(OpenVINO CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED core dnn imgcodecs imgproc)

add_executable(
    openvinoExample
    openvinoExample.cpp
)
target_link_libraries(
    openvinoExample
    openvino::runtime
    opencv_core
    opencv_dnn
    opencv_imgcodecs
)
