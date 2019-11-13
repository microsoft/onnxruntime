# Minimum CMake required
cmake_minimum_required(VERSION 3.10)

include(${ML_BUILD_TOOLS_DIR}/cppwinrt_command.cmake)

winml_add_library(winml_test_common STATIC
    Std.h
    DllLoad.cpp
    filehelpers.cpp
    filehelpers.h
    protobufhelpers.cpp
    protobufhelpers.h
    squeezenetvalidator.h
    squeezenetvalidator.cpp
    JsonHelpers.h
    JsonHelpers.cpp)

# Add packages
target_add_packages(winml_test_common Taef.Native)
target_add_packages(winml_test_common MLRuntime.rapidjson)

# Compiler options
# <none>

# Compiler definitions
target_compile_definitions(winml_test_common PRIVATE ONNX_NAMESPACE=onnx)
target_compile_definitions(winml_test_common PRIVATE ONNX_ML)
target_compile_definitions(winml_test_common PRIVATE LOTUS_LOG_THRESHOLD=2)
target_compile_definitions(winml_test_common PRIVATE LOTUS_ENABLE_STDERR_LOGGING)

# This is needed for lotus
include_directories(${CMAKE_BINARY_DIR}/engine/lotus)                             # windows machine learning component headers
include_directories(${CMAKE_BINARY_DIR}/engine/lotus/winml_api)                   # windows machine learning component headers
include_directories(${CMAKE_BINARY_DIR}/engine/lotus/winml/sdk/cppwinrt/include)  # sdk cppwinrt headers
include_directories(${ML_LOTUS_DIR}/onnxruntime)
include_directories(${ML_LOTUS_DIR}/include/onnxruntime)
include_directories(${ML_LOTUS_DIR}/cmake/external/onnx)
include_directories(${ML_LOTUS_DIR}/cmake/external/gsl/include)
include_directories(${ML_LOTUS_DIR}/cmake/external/protobuf/src)
include_directories(${ML_LOTUS_DIR}/winml/api)
include_directories(${ML_LOTUS_DIR}/winml/lib/common/inc)
include_directories(${CMAKE_SOURCE_DIR}/engine/eigen)
include_directories(${ML_LOTUS_BINARY_DIR})
include_directories(${ML_LOTUS_BINARY_DIR}/external/date/include)
include_directories(${ML_LOTUS_BINARY_DIR}/external/gsl/include)
include_directories(${ML_LOTUS_BINARY_DIR}/external/onnx)

# Properties
set_target_properties(winml_test_common
    PROPERTIES
    FOLDER
    "WinML")

# Add deps
add_dependencies(winml_test_common onnx)
add_dependencies(winml_test_common DirectML)

# Link libraries
target_link_libraries(winml_test_common PRIVATE wil)

set(filesToCopy
    testdata/squeezenet/model.onnx
    testdata/squeezenet/test_data_0_input.pb
    testdata/squeezenet/test_data_0_output.pb
)
foreach (file ${filesToCopy})
    file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/winml/${CMAKE_CONFIGURATION_TYPES})
endforeach(file)

set (postBuildFilesToCopy
    ${CMAKE_BINARY_DIR}/bin/DirectML.dll
)
foreach (file ${postBuildFilesToCopy})
    add_custom_command(TARGET winml_test_common POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${file} ${CMAKE_BINARY_DIR}/bin/winml/${CMAKE_CONFIGURATION_TYPES})
endforeach(file)

install(DIRECTORY testdata/squeezenet/ DESTINATION bin)