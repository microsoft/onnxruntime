cmake_minimum_required(VERSION 3.26)
project(custom_openvino VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
add_definitions(-DOPENVINO_2023_1=1)
add_definitions(-DONNX_NAMESPACE=onnx)
add_definitions(-DONNX_ML)

file(GLOB_RECURSE custom_openvino_src CONFIGURE_DEPENDS "*.h" "*.cc")

find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)
add_library(custom_openvino SHARED ${custom_openvino_src})
target_include_directories(custom_openvino PUBLIC "../../include/onnxruntime" $ENV{INTEL_OPENVINO_DIR}/../../include/ "../../build/openvino/Debug/_deps/onnx-build" "../../build/openvino/Debug/_deps/protobuf-src/src" "../../build/openvino/Debug/_deps/onnx-src")
#target_include_directories(custom_openvino PUBLIC "../../include/onnxruntime" "/bert_ort/leca/condaenv/pe/include/openvino")

target_link_libraries(custom_openvino PUBLIC "/bert_ort/leca/code/onnxruntime/build/openvino/Debug/_deps/onnx-build/libonnx.a")
