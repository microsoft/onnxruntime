# usage:
# cd build/
# cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Debug
# cmake --build ./
cmake_minimum_required(VERSION 3.26)
project(TestOutTreeEp)
add_executable(TestOutTreeEp test.cpp)

target_include_directories(TestOutTreeEp PUBLIC "../../include/onnxruntime")
#target_link_libraries(TestOutTreeEp PUBLIC "/home/leca/code/onnxruntime/build/Linux/Debug/libonnxruntime.so")
target_link_libraries(TestOutTreeEp PUBLIC "/home/leca/code/onnxruntime/build/tensorrt/Debug/libonnxruntime.so")
