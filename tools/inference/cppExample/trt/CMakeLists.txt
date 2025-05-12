cmake_minimum_required(VERSION 3.5.1)

project(
    trtExample
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

find_package(OpenCV CONFIG REQUIRED core dnn imgcodecs imgproc)
find_package(CUDA REQUIRED)

set(tensorrt_include_DIR $ENV{Tensorrt_DIR}/include)
set(tensorrt_lib_DIR $ENV{Tensorrt_DIR}/lib)

file(GLOB tensorrt_libs
    "${tensorrt_lib_DIR}/*.lib"
)
message("Tensorrt include: ${tensorrt_include_DIR}")
message("Tensorrt lib: ${tensorrt_lib_DIR}")

add_executable(
    trtExample
    trtExample.cpp
)
target_link_libraries(
    ${tensorrt_libs}
    ${CUDA_LIBRARIES}
    opencv_core
    opencv_dnn
    opencv_imgcodecs
)
target_include_directories(
    trtExample PUBLIC
    ${tensorrt_include_DIR}
    ${CUDA_INCLUDE_DIRS}
)
