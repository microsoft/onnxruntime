# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# QNN EP Static Library Build
# This file contains only the static library build configuration to avoid
# NodeAttributes redefinition conflicts with shared builds

add_compile_definitions(USE_QNN=1)

# Force static library build mode
if(NOT onnxruntime_BUILD_QNN_EP_STATIC_LIB)
  message(FATAL_ERROR "onnxruntime_providers_qnn_static.cmake is for static library builds only. Use onnxruntime_providers_qnn_abi.cmake for shared builds.")
endif()

add_compile_definitions(BUILD_QNN_EP_STATIC_LIB=1)

message(STATUS "QNN EP: Using STATIC library build - qnn/ directory")

# Collect QNN static library source files
file(GLOB_RECURSE
     onnxruntime_providers_qnn_ep_srcs CONFIGURE_DEPENDS
     "${ONNXRUNTIME_ROOT}/core/providers/qnn/*.h"
     "${ONNXRUNTIME_ROOT}/core/providers/qnn/*.cc"
)

# Debug: Print source files count
list(LENGTH onnxruntime_providers_qnn_ep_srcs num_sources)
message(STATUS "QNN EP Static: Found ${num_sources} source files")
if(num_sources GREATER 0)
  list(GET onnxruntime_providers_qnn_ep_srcs 0 first_source)
  message(STATUS "QNN EP Static: First source file: ${first_source}")
endif()

set(example_plugin_ep_utils_srcs "${ONNXRUNTIME_ROOT}/test/autoep/library/example_plugin_ep_utils.cc")

function(extract_qnn_sdk_version_from_yaml QNN_SDK_YAML_FILE QNN_VERSION_OUTPUT)
  file(READ "${QNN_SDK_YAML_FILE}" QNN_SDK_YAML_CONTENT)
  # Match a line of text like "version: 1.33.2"
  string(REGEX MATCH "(^|\n|\r)version: ([0-9]+\\.[0-9]+\\.[0-9]+)" QNN_VERSION_MATCH "${QNN_SDK_YAML_CONTENT}")
  if(QNN_VERSION_MATCH)
    set(${QNN_VERSION_OUTPUT} "${CMAKE_MATCH_2}" PARENT_SCOPE)
    message(STATUS "Extracted QNN SDK version ${CMAKE_MATCH_2} from ${QNN_SDK_YAML_FILE}")
  else()
    message(WARNING "Failed to extract QNN SDK version from ${QNN_SDK_YAML_FILE}")
  endif()
endfunction()

if(NOT QNN_SDK_VERSION)
  if(EXISTS "${onnxruntime_QNN_HOME}/sdk.yaml")
    extract_qnn_sdk_version_from_yaml("${onnxruntime_QNN_HOME}/sdk.yaml" QNN_SDK_VERSION)
  else()
    message(WARNING "Cannot open sdk.yaml to extract QNN SDK version")
  endif()
endif()
message(STATUS "QNN SDK version ${QNN_SDK_VERSION}")

#
# Build QNN EP as a static library
#
set(onnxruntime_providers_qnn_srcs ${onnxruntime_providers_qnn_ep_srcs} ${example_plugin_ep_utils_srcs})
source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_qnn_ep_srcs})
source_group(TREE ${ONNXRUNTIME_ROOT}/test FILES ${example_plugin_ep_utils_srcs})
onnxruntime_add_static_library(onnxruntime_providers_qnn ${onnxruntime_providers_qnn_srcs})
onnxruntime_add_include_to_target(onnxruntime_providers_qnn onnxruntime_common onnxruntime_framework onnx
                                                            onnx_proto protobuf::libprotobuf-lite
                                                            flatbuffers::flatbuffers Boost::mp11
                                                            nlohmann_json::nlohmann_json)
add_dependencies(onnxruntime_providers_qnn onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
set_target_properties(onnxruntime_providers_qnn PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(onnxruntime_providers_qnn PROPERTIES FOLDER "ONNXRuntime")
target_include_directories(onnxruntime_providers_qnn PRIVATE ${ONNXRUNTIME_ROOT}
                                                             ${onnxruntime_QNN_HOME}/include/QNN
                                                             ${onnxruntime_QNN_HOME}/include)
set_target_properties(onnxruntime_providers_qnn PROPERTIES LINKER_LANGUAGE CXX)

# ignore the warning unknown-pragmas on "pragma region"
if(NOT MSVC)
  target_compile_options(onnxruntime_providers_qnn PRIVATE "-Wno-unknown-pragmas")
endif()

set(onnxruntime_providers_qnn_target onnxruntime_providers_qnn)

if (MSVC OR ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  add_custom_command(
    TARGET ${onnxruntime_providers_qnn_target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${QNN_LIB_FILES} $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_target}>
    )
endif()
if (EXISTS "${onnxruntime_QNN_HOME}/Qualcomm AI Hub Proprietary License.pdf")
  add_custom_command(
    TARGET ${onnxruntime_providers_qnn_target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "${onnxruntime_QNN_HOME}/Qualcomm AI Hub Proprietary License.pdf" $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_target}>
    )
endif()

if (NOT onnxruntime_BUILD_SHARED_LIB)
  install(TARGETS onnxruntime_providers_qnn EXPORT ${PROJECT_NAME}Targets
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()