# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(onnxruntime_common_src_patterns
    "${ONNXRUNTIME_INCLUDE_DIR}/core/common/*.h"
    "${ONNXRUNTIME_INCLUDE_DIR}/core/common/logging/*.h"
    "${ONNXRUNTIME_INCLUDE_DIR}/core/platform/*.h"
    "${ONNXRUNTIME_ROOT}/core/common/*.h"
    "${ONNXRUNTIME_ROOT}/core/common/*.cc"
    "${ONNXRUNTIME_ROOT}/core/common/logging/*.h"
    "${ONNXRUNTIME_ROOT}/core/common/logging/*.cc"
    "${ONNXRUNTIME_ROOT}/core/common/logging/sinks/*.h"
    "${ONNXRUNTIME_ROOT}/core/common/logging/sinks/*.cc"
    "${ONNXRUNTIME_ROOT}/core/inc/*.h"
    "${ONNXRUNTIME_ROOT}/core/platform/env.h"
    "${ONNXRUNTIME_ROOT}/core/platform/env.cc"
    "${ONNXRUNTIME_ROOT}/core/platform/env_time.h"
    "${ONNXRUNTIME_ROOT}/core/platform/env_time.cc"
)

if(WIN32)
    list(APPEND onnxruntime_common_src_patterns
         "${ONNXRUNTIME_ROOT}/core/platform/windows/*.h"
         "${ONNXRUNTIME_ROOT}/core/platform/windows/*.cc"
         "${ONNXRUNTIME_ROOT}/core/platform/windows/logging/*.h"
         "${ONNXRUNTIME_ROOT}/core/platform/windows/logging/*.cc"
    )
else()
    list(APPEND onnxruntime_common_src_patterns
         "${ONNXRUNTIME_ROOT}/core/platform/posix/*.h"
         "${ONNXRUNTIME_ROOT}/core/platform/posix/*.cc"
    )
endif()

file(GLOB onnxruntime_common_src ${onnxruntime_common_src_patterns})

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_common_src})

add_library(onnxruntime_common ${onnxruntime_common_src})

onnxruntime_add_include_to_target(onnxruntime_common gsl date_interface)
target_include_directories(onnxruntime_common PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT}
        PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/external/nsync/public")
if(onnxruntime_USE_NSYNC)
    target_compile_definitions(onnxruntime_common PUBLIC USE_NSYNC)
endif()
if(onnxruntime_USE_EIGEN_THREADPOOL)
    target_include_directories(onnxruntime_common PRIVATE ${eigen_INCLUDE_DIRS})
    target_compile_definitions(onnxruntime_common PUBLIC USE_EIGEN_THREADPOOL)
    add_dependencies(onnxruntime_common ${onnxruntime_EXTERNAL_DEPENDENCIES})
endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/common  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core)
set_target_properties(onnxruntime_common PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(onnxruntime_common PROPERTIES FOLDER "ONNXRuntime")

if(WIN32)
    # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include.
    set_target_properties(onnxruntime_common PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/EnableVisualStudioCodeAnalysis.props)
endif()
