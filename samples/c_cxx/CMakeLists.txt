# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)

# Project
project(onnxruntime_samples C CXX)
if (WIN32)
 string(APPEND CMAKE_CXX_FLAGS " /W4")
else()
 string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra")
 string(APPEND CMAKE_C_FLAGS " -Wall -Wextra")
endif()

message("${CMAKE_SYSTEM_INCLUDE_PATH}")
find_package(JPEG)
find_package(PNG)


add_definitions(-DUSE_CUDA)

#TODO: remove this part(need to talk to Ryan and find a solution for this)
if(UNIX)
 include_directories("/usr/local/include/onnxruntime")
else()
 include_directories("C:\\Program Files (x86)\\onnxruntime\\include" "C:\\Program Files (x86)\\onnxruntime\\include\\onnxruntime" "C:\\Program Files (x86)\\onnxruntime\\include\\onnxruntime\\core\\session")
 link_directories("C:\\Program Files (x86)\\onnxruntime\\lib")
endif()


add_subdirectory(imagenet)
if(PNG_FOUND)
  add_subdirectory(fns_candy_style_transfer)
endif()
add_subdirectory(MNIST)
