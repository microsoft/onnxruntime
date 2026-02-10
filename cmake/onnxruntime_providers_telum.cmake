# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Telum Execution Provider CMake Configuration

if(NOT onnxruntime_USE_TELUM)
  return()
endif()

add_definitions(-DUSE_TELUM=1)

# Find zDNN library
if(NOT DEFINED ZDNN_ROOT)
  if(DEFINED ENV{ZDNN_ROOT})
    set(ZDNN_ROOT $ENV{ZDNN_ROOT})
  else()
    message(FATAL_ERROR "ZDNN_ROOT not set. Please set ZDNN_ROOT to zDNN installation directory.")
  endif()
endif()

# Find zDNN library and headers
find_library(ZDNN_LIB
  NAMES zdnn libzdnn
  PATHS ${ZDNN_ROOT}/lib ${ZDNN_ROOT}/zdnn
  NO_DEFAULT_PATH
  REQUIRED
)

find_path(ZDNN_INCLUDE
  NAMES zdnn.h
  PATHS ${ZDNN_ROOT}/include ${ZDNN_ROOT}/zdnn
  NO_DEFAULT_PATH
  REQUIRED
)

if(NOT ZDNN_LIB)
  message(FATAL_ERROR "zDNN library not found. Please check ZDNN_ROOT: ${ZDNN_ROOT}")
endif()

if(NOT ZDNN_INCLUDE)
  message(FATAL_ERROR "zDNN headers not found. Please check ZDNN_ROOT: ${ZDNN_ROOT}")
endif()

message(STATUS "Found zDNN library: ${ZDNN_LIB}")
message(STATUS "Found zDNN headers: ${ZDNN_INCLUDE}")

# Verify we're on s390x architecture
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "s390x")
  message(WARNING "Telum EP is designed for s390x architecture. Current: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Collect Telum EP source files
file(GLOB_RECURSE onnxruntime_providers_telum_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/telum/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/telum/*.cc"
)

# Create Telum EP library
source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_telum_cc_srcs})

onnxruntime_add_static_library(onnxruntime_providers_telum ${onnxruntime_providers_telum_cc_srcs})

# Set include directories
target_include_directories(onnxruntime_providers_telum PRIVATE
  ${ONNXRUNTIME_ROOT}
  ${ZDNN_INCLUDE}
  ${CMAKE_CURRENT_BINARY_DIR}
)

# Link zDNN library
target_link_libraries(onnxruntime_providers_telum PRIVATE
  ${ZDNN_LIB}
  onnxruntime_common
  onnxruntime_framework
  onnx
  onnx_proto
  ${PROTOBUF_LIB}
)

# Set compile options
target_compile_options(onnxruntime_providers_telum PRIVATE
  -march=z16
  -mtune=z16
)

# Add to main onnxruntime library
set_target_properties(onnxruntime_providers_telum PROPERTIES FOLDER "ONNXRuntime")
set_target_properties(onnxruntime_providers_telum PROPERTIES LINKER_LANGUAGE CXX)

# Install targets
if(NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_EXTENDED_MINIMAL_BUILD)
  install(TARGETS onnxruntime_providers_telum
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()
