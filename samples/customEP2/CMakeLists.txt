cmake_minimum_required(VERSION 3.26)
project(customep2 VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
add_library(customep2 SHARED custom_ep2.cc)
set(CUSTOM_PUBLIC_HEADER custom_ep2.h)
#set_target_properties(customep2 PROPERTIES PUBLIC_HEADER custom_ep2.h)
set_target_properties(customep2 PROPERTIES PUBLIC_HEADER "${CUSTOM_PUBLIC_HEADER}")

target_include_directories(customep2 PUBLIC "../../include/onnxruntime")
#target_include_directories(customep PUBLIC "/bert_ort/leca/code/onnxruntime2/onnxruntime")

target_link_libraries(customep2 PUBLIC "/home/leca/code/onnxruntime/build/Linux/Debug/libonnxruntime.so")

