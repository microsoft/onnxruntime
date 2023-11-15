cmake_minimum_required(VERSION 3.26)
project(customep2 VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
add_library(customep2 SHARED custom_ep2.cc)
set(CUSTOM_PUBLIC_HEADER custom_ep2.h)

#set_target_properties(customep2 PROPERTIES PUBLIC_HEADER custom_ep2.h)
set_target_properties(customep2 PROPERTIES PUBLIC_HEADER "${CUSTOM_PUBLIC_HEADER}")

IF (WIN32)

target_include_directories(customep2 PUBLIC "/onnxruntime/include/onnxruntime")
target_link_directories(customep2 PUBLIC "/onnxruntime/build/windows/debug/debug/")
target_link_libraries(customep2 PUBLIC onnxruntime)

ELSE()

target_include_directories(customep2 PUBLIC "../../include/onnxruntime")
target_link_libraries(customep2 PUBLIC "/bert_ort/leca/code/onnxruntime/build/Linux/Debug/libonnxruntime.so") # seems relative path does not work

ENDIF()
