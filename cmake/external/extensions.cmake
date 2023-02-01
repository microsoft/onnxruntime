# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

message(STATUS "[onnxruntime-extensions] Building onnxruntime-extensions: ${onnxruntime_EXTENSIONS_PATH}")

# add compile definition to enable custom operators in onnxruntime-extensions
add_compile_definitions(ENABLE_EXTENSION_CUSTOM_OPS)

# set options for onnxruntime-extensions
set(OCOS_ENABLE_CTEST OFF CACHE INTERNAL "")
set(OCOS_ENABLE_STATIC_LIB ON CACHE INTERNAL "")
set(OCOS_ENABLE_SPM_TOKENIZER OFF CACHE INTERNAL "")

# disable exceptions
if (onnxruntime_DISABLE_EXCEPTIONS)
  set(OCOS_ENABLE_CPP_EXCEPTIONS OFF CACHE INTERNAL "")
endif()

# customize operators used
if (onnxruntime_REDUCED_OPS_BUILD)
  set(OCOS_ENABLE_SELECTED_OPLIST ON CACHE INTERNAL "")
endif()

# when onnxruntime-extensions is not a subdirectory of onnxruntime,
# output binary directory must be explicitly specified.
# and the output binary path is the same as CMake FetchContent pattern
add_subdirectory(${onnxruntime_EXTENSIONS_PATH} ${CMAKE_BINARY_DIR}/_deps/extensions-subbuild EXCLUDE_FROM_ALL)

# target library or executable are defined in CMakeLists.txt of onnxruntime-extensions
target_include_directories(ocos_operators PRIVATE ${RE2_INCLUDE_DIR} ${json_SOURCE_DIR}/include)
target_include_directories(ortcustomops PUBLIC ${onnxruntime_EXTENSIONS_PATH}/includes)
