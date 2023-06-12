cmake_minimum_required(VERSION 3.26)
project(onnxruntime_core_provider_js)
set(CMAKE_CXX_STANDARD 17)
file(GLOB_RECURSE SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        )
add_library(${PROJECT_NAME} STATIC ${SOURCES})
