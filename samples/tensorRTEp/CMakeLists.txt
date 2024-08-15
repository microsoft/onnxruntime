# usage:
# cd build/
# cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc (see the result of "nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits")
# cmake --build ./
cmake_minimum_required(VERSION 3.26)
project(TensorRTEp VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)
file(TO_CMAKE_PATH CUDAToolkit_ROOT "/usr/local/cuda")
find_package(CUDAToolkit REQUIRED)

#add_definitions(-DONNX_NAMESPACE=onnx)
#add_definitions(-DONNX_ML)
add_library(TensorRTEp SHARED tensorrt_execution_provider.cc)
target_include_directories(TensorRTEp PUBLIC "../../include/onnxruntime"
                                             "/usr/local/cuda/include")
#                                        "/home/leca/qnn-v2.25.0.240728/include/QNN"
#                                        "../../build/Linux/Debug/_deps/gsl-src/include"
#                                        "../../build/Linux/Debug/_deps/onnx-src"
#                                        "../../build/Linux/Debug/_deps/onnx-build"
#                                        "../../build/Linux/Debug/_deps/protobuf-src/src")
#
## looks we need libonnxruntime.so in Win as in Windows you cannot build shared library with undefined symbol
#target_link_libraries(TensorRTEp PUBLIC #"/home/leca/code/onnxruntime/build/Linux/Debug/libonnxruntime.so"
#                                    "/home/leca/code/onnxruntime/build/Linux/Debug/_deps/onnx-build/libonnx.a"
#                                    "/home/leca/code/onnxruntime/build/Linux/Debug/_deps/onnx-build/libonnx_proto.a"
#                                    "/home/leca/code/onnxruntime/build/Linux/Debug/_deps/protobuf-build/libprotobufd.a"
#                                    "/home/leca/code/onnxruntime/build/Linux/Debug/_deps/protobuf-build/libprotocd.a")
