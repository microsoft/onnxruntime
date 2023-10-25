cmake_minimum_required(VERSION 3.26)
project(ConsoleUsingExternalEP)
add_executable(ConsoleUsingExternalEP test.cpp)

target_include_directories(ConsoleUsingExternalEP PUBLIC "/bert_ort/leca/code/onnxruntime2/include/onnxruntime")
target_link_libraries(ConsoleUsingExternalEP PUBLIC "/bert_ort/leca/code/onnxruntime2/build/Linux/Debug/libonnxruntime.so")

