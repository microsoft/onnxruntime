# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(FetchContent)

set(abseil_URL https://github.com/abseil/abseil-cpp.git)
set(abseil_TAG 9336be04a242237cd41a525bedfcf3be1bb55377)

# Pass to build
set(ABSL_PROPAGATE_CXX_STD 1)
set(BUILD_TESTING 0)

FetchContent_Declare(
    abseil_cpp
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/abseil-cpp"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/external/abseil-cpp"
    GIT_REPOSITORY ${abseil_URL}
    GIT_TAG ${abseil_TAG}
)

FetchContent_MakeAvailable(abseil_cpp)
FetchContent_GetProperties(abseil_cpp SOURCE_DIR)

# Patching separately because repeated builds would fail the whole FetchContent_* command
# instead of just patching
execute_process(COMMAND  git apply --ignore-space-change --ignore-whitespace ${PROJECT_SOURCE_DIR}/patches/abseil/Fix_Nvidia_Build_Break.patch
                    WORKING_DIRECTORY "${abseil_cpp_SOURCE_DIR}"
                   )

include_directories("${abseil_cpp_SOURCE_DIR}")

list(APPEND onnxruntime_EXTERNAL_LIBRARIES absl::inlined_vector absl::flat_hash_set absl::flat_hash_map absl::base absl::throw_delegate absl::raw_hash_set absl::hash absl::city absl::low_level_hash absl::raw_logging_internal)
list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES absl::inlined_vector absl::flat_hash_set absl::flat_hash_map absl::base absl::throw_delegate absl::raw_hash_set absl::hash absl::city absl::low_level_hash absl::raw_logging_internal)

