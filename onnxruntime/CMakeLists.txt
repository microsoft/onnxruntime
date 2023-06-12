cmake_minimum_required(VERSION 3.26.0)
project(onnxruntime)
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../include/onnxruntime)
add_subdirectory(wasm)
add_subdirectory(core)
