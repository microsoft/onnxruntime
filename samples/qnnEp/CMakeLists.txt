# usage:
# cd build/
# cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Debug
# cmake --build ./
cmake_minimum_required(VERSION 3.26)
project(QnnEp VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
if (MSVC)
else()
    set(QNN_ARCH_ABI aarch64-android)
endif()

add_definitions(-DONNX_NAMESPACE=onnx)
add_definitions(-DONNX_ML)
file(GLOB_RECURSE qnn_src "./*.cc")
#message(STATUS "qnn_src=${qnn_src}")
add_library(QnnEp SHARED ${qnn_src})
target_include_directories(QnnEp PUBLIC "../../include/onnxruntime"
                                        "/home/leca/qnn-v2.25.0.240728/include/QNN"
                                        "../../build/Linux/Debug/_deps/gsl-src/include"
                                        "../../build/Linux/Debug/_deps/onnx-src"
                                        "../../build/Linux/Debug/_deps/onnx-build"
                                        "../../build/Linux/Debug/_deps/protobuf-src/src")

# looks we need libonnxruntime.so in Win as in Windows you cannot build shared library with undefined symbol
target_link_libraries(QnnEp PUBLIC #"/home/leca/code/onnxruntime/build/Linux/Debug/libonnxruntime.so"
                                    "/home/leca/code/onnxruntime/build/Linux/Debug/_deps/onnx-build/libonnx.a"
                                    "/home/leca/code/onnxruntime/build/Linux/Debug/_deps/onnx-build/libonnx_proto.a"
                                    "/home/leca/code/onnxruntime/build/Linux/Debug/_deps/protobuf-build/libprotobufd.a"
                                    "/home/leca/code/onnxruntime/build/Linux/Debug/_deps/protobuf-build/libprotocd.a")
