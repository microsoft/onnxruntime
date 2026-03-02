# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.28)

project(onnxruntime_sample CXX)

set(CMAKE_CXX_STANDARD 20)

foreach(VAR IN ITEMS ORT_LIBRARY_DIR ORT_HEADER_DIR)
    if (NOT DEFINED ${VAR})
        message(FATAL_ERROR "Required variable ${VAR} is not set. "
            "Set ORT_LIBRARY_DIR to the ONNX Runtime lib directory and "
            "ORT_HEADER_DIR to the ONNX Runtime include directory.")
    endif()
endforeach()

# Resolve to absolute paths
get_filename_component(ORT_LIBRARY_DIR "${ORT_LIBRARY_DIR}" ABSOLUTE)
get_filename_component(ORT_HEADER_DIR "${ORT_HEADER_DIR}" ABSOLUTE)

#
# onnxruntime_sample_program
#
block()
add_executable(onnxruntime_sample_program)

target_sources(onnxruntime_sample_program PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)

target_include_directories(onnxruntime_sample_program PRIVATE ${ORT_HEADER_DIR})

target_link_directories(onnxruntime_sample_program PRIVATE ${ORT_LIBRARY_DIR})
target_link_libraries(onnxruntime_sample_program PRIVATE onnxruntime)

# Copy ONNX Runtime shared libraries next to the executable.
# Collect shared library files from the ORT library directory based on platform.
if (WIN32)
    file(GLOB ORT_SHARED_LIBS "${ORT_LIBRARY_DIR}/*.dll")
elseif (APPLE)
    file(GLOB ORT_SHARED_LIBS "${ORT_LIBRARY_DIR}/*.dylib")
else()
    file(GLOB ORT_SHARED_LIBS "${ORT_LIBRARY_DIR}/*.so" "${ORT_LIBRARY_DIR}/*.so.*")
endif()

add_custom_command(TARGET onnxruntime_sample_program POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ORT_SHARED_LIBS}
        $<TARGET_FILE_DIR:onnxruntime_sample_program>
)
endblock()
