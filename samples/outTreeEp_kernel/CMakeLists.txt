# usage:
# cd build/
# cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Debug
# cmake --build ./
cmake_minimum_required(VERSION 3.26)
project(kernelEp VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
add_library(kernelEp SHARED kernel_ep.cc)
target_include_directories(kernelEp PUBLIC "../../include/onnxruntime")

# looks we need this in Win as in Windows you cannot build shared library with undefined symbol
#target_link_libraries(outTreeEp PUBLIC "/home/leca/code/onnxruntime/build/Linux/Debug/libonnxruntime.so")
