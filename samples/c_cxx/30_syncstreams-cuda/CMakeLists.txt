cmake_minimum_required(VERSION 3.20)
project(winai-samples)
find_package(CUDAToolkit REQUIRED)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(onnxruntimesetup)

add_executable(syncstreams_cuda
    main.cpp
    lodepng/lodepng.cpp
    utils.cpp
    )

set_target_properties(syncstreams_cuda PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
    )
target_link_libraries(syncstreams_cuda PRIVATE
  onnxruntime_interface
  CUDA::cudart_static
)
target_include_directories(syncstreams_cuda PRIVATE
    lode_png
)

set(ONNX "candy.onnx")

copy_file_to_bin_dir(${ONNX})
copy_file_to_bin_dir(Input.png)

set_target_properties(syncstreams_cuda
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${RUNTIME_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY "${RUNTIME_DIRECTORY}"
    RUNTIME_OUTPUT_DIRECTORY "${RUNTIME_DIRECTORY}"
)
