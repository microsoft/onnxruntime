# Copyright(C) Xilinx Inc.
# Licensed under the MIT License

include(ExternalProject)

ExternalProject_Add(oct
                    GIT_REPOSITORY https://github.com/RandySheriffH/octopus.git
                    GIT_TAG busyspinII
                    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/_deps/oct-src
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND ""
                    INSTALL_COMMAND "")

include_directories(${CMAKE_CURRENT_BINARY_DIR}/_deps/oct-src/include)
