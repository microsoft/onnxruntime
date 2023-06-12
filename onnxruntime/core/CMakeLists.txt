cmake_minimum_required(VERSION 3.26)
project(onnxruntime_core_provider_js)
set(CMAKE_CXX_STANDARD 17)
add_subdirectory(providers/js)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

