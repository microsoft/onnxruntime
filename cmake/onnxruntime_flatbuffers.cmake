# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB onnxruntime_flatbuffers_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/flatbuffers/*.h"
    "${ONNXRUNTIME_ROOT}/core/flatbuffers/*.cc"
    )

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_flatbuffers_srcs})

onnxruntime_add_static_library(onnxruntime_flatbuffers ${onnxruntime_flatbuffers_srcs})
onnxruntime_add_include_to_target(onnxruntime_flatbuffers onnx flatbuffers)
if(onnxruntime_ENABLE_INSTRUMENT)
  target_compile_definitions(onnxruntime_flatbuffers PUBLIC ONNXRUNTIME_ENABLE_INSTRUMENT)
endif()
target_include_directories(onnxruntime_flatbuffers PRIVATE ${ONNXRUNTIME_ROOT})
add_dependencies(onnxruntime_flatbuffers ${onnxruntime_EXTERNAL_DEPENDENCIES})
set_target_properties(onnxruntime_flatbuffers PROPERTIES FOLDER "ONNXRuntime")

# Add dependency so the flatbuffers compiler is built if enabled
if (FLATBUFFERS_BUILD_FLATC)
  add_dependencies(onnxruntime_flatbuffers flatc)
endif()
if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_flatbuffers
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if (GDK_PLATFORM)
  # cstdlib only defines std::getenv when _CRT_USE_WINAPI_FAMILY_DESKTOP_APP is defined, which
  # is probably an oversight for GDK/Xbox builds (::getenv exists and works).
  file(WRITE ${CMAKE_BINARY_DIR}/gdk_cstdlib_wrapper.h [[
#pragma once
#ifdef __cplusplus
#include <cstdlib>
namespace std { using ::getenv; }
#endif
]])
  target_compile_options(flatbuffers PRIVATE /FI${CMAKE_BINARY_DIR}/gdk_cstdlib_wrapper.h)
  target_compile_options(flatc PRIVATE /FI${CMAKE_BINARY_DIR}/gdk_cstdlib_wrapper.h)
endif()