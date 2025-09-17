cmake_minimum_required(VERSION 3.16)
project(ONNXRuntimeSample)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(onnxruntimesetup)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable for sample.cpp (file-based)
add_executable(sample sample.cpp)
target_link_libraries(sample PRIVATE onnxruntime_interface)

# Add executable for sample_buffer.cpp (buffer-based)
add_executable(sample_buffer sample_buffer.cpp)
target_link_libraries(sample_buffer PRIVATE onnxruntime_interface)
