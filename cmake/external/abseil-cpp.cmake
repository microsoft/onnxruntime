# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(FetchContent)

# Pass to build
set(ABSL_PROPAGATE_CXX_STD 1)
set(BUILD_TESTING 0)

if(Patch_FOUND)
  set(ABSL_PATCH_COMMAND ${Patch_EXECUTABLE} --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/abseil/Fix_Nvidia_Build_Break.patch)
else()
  set(ABSL_PATCH_COMMAND git apply --ignore-space-change --ignore-whitespace ${PROJECT_SOURCE_DIR}/patches/abseil/Fix_Nvidia_Build_Break.patch)
endif()

FetchContent_Declare(
    abseil_cpp
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/abseil-cpp"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/abseil-cpp"
    GIT_REPOSITORY https://github.com/jstoecker/abseil-cpp.git
    GIT_TAG        c4ae5fa3144c8f04359c9344a104453c2315314e
)

FetchContent_MakeAvailable(abseil_cpp)
FetchContent_GetProperties(abseil_cpp SOURCE_DIR)




