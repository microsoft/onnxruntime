# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)

add_executable(mnist MNIST.cpp)

target_link_options(mnist PRIVATE "/SUBSYSTEM:WINDOWS")