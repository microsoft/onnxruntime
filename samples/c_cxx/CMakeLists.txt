cmake_minimum_required(VERSION 3.20)
project(onnxruntime_c_cxx_samples)


if (MSVC)
    # warning level 4
    add_compile_options(/W4)
else()
    # additional warnings
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_subdirectory(10_ep-device-selection)
add_subdirectory(20_devicetensors-datatransfer)
add_subdirectory(30_syncstreams-cuda)
add_subdirectory(40_ep-context)
