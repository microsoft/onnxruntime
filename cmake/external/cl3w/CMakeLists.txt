# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.12)
project(cl3w)

add_library(cl3w STATIC "src/cl3w.c" "include/cl3w.h")
target_include_directories(cl3w PUBLIC "include" "OpenCL-Headers")
target_compile_definitions(cl3w PUBLIC "CL3W_ENABLE" "CL_TARGET_OPENCL_VERSION=120")
set_target_properties(cl3w PROPERTIES POSITION_INDEPENDENT_CODE ON)
