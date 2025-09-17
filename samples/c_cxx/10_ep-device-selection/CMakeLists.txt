cmake_minimum_required(VERSION 3.20)
project(winai-samples)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(onnxruntimesetup)

add_executable(ep-device-selection
    main.cpp
    utils.cpp
    lodepng/lodepng.cpp
    )

set_target_properties(ep-device-selection PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
    )
target_link_libraries(ep-device-selection PRIVATE
  onnxruntime_interface
)
target_include_directories(ep-device-selection PRIVATE
    lode_png
)

set(DEPTH_ANYTHING_FP16_ONNX "depth_anything_v2_torch.float16_240.onnx")

copy_file_to_bin_dir(${DEPTH_ANYTHING_FP16_ONNX})
copy_file_to_bin_dir(Input.png)

target_compile_definitions(ep-device-selection
    PRIVATE -DMODEL_FILE="${DEPTH_ANYTHING_FP16_ONNX}")
set_target_properties(ep-device-selection
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${RUNTIME_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY "${RUNTIME_DIRECTORY}"
    RUNTIME_OUTPUT_DIRECTORY "${RUNTIME_DIRECTORY}"
)
