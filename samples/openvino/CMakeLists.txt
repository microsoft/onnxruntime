# usage:
# cd build/
# cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Debug -DOPENVINO_HOME="C:/Program Files (x86)/Intel/openvino_2024.4.0/runtime"
# cmake --build .
cmake_minimum_required(VERSION 3.26)
project(TensorRTEp VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenVINO REQUIRED COMPONENTS Runtime ONNX)
list(APPEND OPENVINO_LIB_LIST openvino::frontend::onnx openvino::runtime)

file(GLOB openvino_src "./*.cc" "./ov_versions/*.cc")
add_library(OpenVINOEp SHARED ${openvino_src})
target_include_directories(OpenVINOEp PUBLIC "../../include/onnxruntime"
                                             ${OPENVINO_HOME}/include
)
target_link_libraries(OpenVINOEp PUBLIC "C:/Users/leca/source/onnxruntime/build/Windows/Debug/Debug/onnxruntime.lib"
                                        ${OPENVINO_LIB_LIST}
)
