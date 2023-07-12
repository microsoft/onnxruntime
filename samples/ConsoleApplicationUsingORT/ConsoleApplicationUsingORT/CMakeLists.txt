cmake_minimum_required(VERSION 3.26)
project(ConsoleUsingOrt)
add_executable(ConsoleUsingOrt ConsoleApplicationUsingORT.cpp)

# OK!
#target_include_directories(ConsoleUsingOrt PUBLIC "/bert_ort/leca/code/cmake/MathFunctions")
#target_link_libraries(ConsoleUsingOrt PUBLIC "/bert_ort/leca/code/cmake/MathFunctions/build/libmathLib.so")

target_include_directories(ConsoleUsingOrt PUBLIC "/bert_ort/leca/code/onnxruntime2/include/onnxruntime")
target_link_libraries(ConsoleUsingOrt PUBLIC "/bert_ort/leca/code/onnxruntime2/build/Linux/Debug/libonnxruntime.so")
