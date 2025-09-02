# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: MIT

##
## Cross-compilation toolchain for linux-oe-aarch64-gcc11.2. This is the same toolchain
## used by QAIRT/QNN.
##
## This comes with advice from: https://mcilloni.ovh/2021/02/09/cxx-cross-clang/
##

if(LINUX_AARCH64_TOOLCHAIN_INCLUDED)
  return()
endif()
set(LINUX_AARCH64_TOOLCHAIN_INCLUDED true)

# the target triple
set(LINUX_TOOLCHAIN_TARGET "aarch64-oe-linux")

# the root of the cross-compilation toolchain, passed in via the environment since
# toolchain files are sometimes evaluated without the project's CMakeCache.txt
set(LINUX_TOOLCHAIN_ROOT "$ENV{ORT_BUILD_LINUX_TOOLCHAIN_ROOT}")
if (NOT IS_DIRECTORY "${LINUX_TOOLCHAIN_ROOT}")
  message(FATAL_ERROR "LINUX_TOOLCHAIN_ROOT ${LINUX_TOOLCHAIN_ROOT} is not a directory.")
endif()

cmake_path(SET LINUX_TOOLCHAIN_ROOT NORMALIZE ${LINUX_TOOLCHAIN_ROOT})

# the platform name in the sysroot
set(LINUX_TOOLCHAIN_PLATFORM "armv8a-oe-linux")

set(LINUX_TOOLCHAIN_SYSROOT "${LINUX_TOOLCHAIN_ROOT}/sysroots/${LINUX_TOOLCHAIN_PLATFORM}")

if(APPLE)
  set(LLVM_ROOT "/opt/homebrew/Cellar/llvm@16/16.0.6_1/bin")
else()
  set(LLVM_ROOT "/usr/lib/llvm-16/bin")
endif()

##
## setup cross-compilation
##

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_ASM_COMPILER_TARGET ${LINUX_TOOLCHAIN_TARGET})

set(CMAKE_C_COMPILER "${LLVM_ROOT}/clang")
set(CMAKE_C_COMPILER_TARGET ${LINUX_TOOLCHAIN_TARGET})

set(CMAKE_CXX_COMPILER "${LLVM_ROOT}/clang++")
set(CMAKE_CXX_COMPILER_TARGET ${LINUX_TOOLCHAIN_TARGET})

set(CMAKE_LINKER "${LLVM_ROOT}/ld.lld")
set(CMAKE_LINKER_TYPE LLD)

set(CMAKE_AR "${LLVM_ROOT}/llvm-ar")
set(CMAKE_RANLIB "${LLVM_ROOT}/llvm-ranlib")

set(CMAKE_SYSROOT "${LINUX_TOOLCHAIN_SYSROOT}")

# these variables tell CMake to avoid using any binary it finds in
# the sysroot, while picking headers and libraries exclusively from it
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

message(STATUS "Using linux toolchain from ${LINUX_TOOLCHAIN_ROOT}")
message(STATUS "System root: ${CMAKE_SYSROOT}")
