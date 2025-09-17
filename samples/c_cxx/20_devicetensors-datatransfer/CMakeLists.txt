cmake_minimum_required(VERSION 3.20)
project(winai-samples)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(onnxruntimesetup)

add_executable(devicetensors-datatransfer
    main.cpp
    lodepng/lodepng.cpp
    utils.cpp
    )

set_target_properties(devicetensors-datatransfer PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
    )
target_link_libraries(devicetensors-datatransfer PRIVATE
  onnxruntime_interface
)
target_include_directories(devicetensors-datatransfer PRIVATE
    lode_png
)

set(ONNX "candy.onnx")

copy_file_to_bin_dir(${ONNX})
copy_file_to_bin_dir(Input.png)

set_target_properties(devicetensors-datatransfer
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${RUNTIME_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY "${RUNTIME_DIRECTORY}"
    RUNTIME_OUTPUT_DIRECTORY "${RUNTIME_DIRECTORY}"
)
