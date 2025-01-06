# Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
# Licensed under the MIT License.

set(JS_ROOT ${REPO_ROOT}/js)
set(JS_COMMON_ROOT ${JS_ROOT}/common)
set(JS_NODE_ROOT ${JS_ROOT}/node)

find_program(NPM_CLI
  NAMES "npm.cmd" "npm"
  DOC "NPM command line client"
  REQUIRED
)

# verify Node.js and NPM
execute_process(COMMAND node --version
    WORKING_DIRECTORY ${JS_NODE_ROOT}
    OUTPUT_VARIABLE node_version
    RESULT_VARIABLE had_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(had_error)
    message(FATAL_ERROR "Failed to find Node.js: " ${had_error})
endif()
execute_process(COMMAND ${NPM_CLI} --version
    WORKING_DIRECTORY ${JS_NODE_ROOT}
    OUTPUT_VARIABLE npm_version
    RESULT_VARIABLE had_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(had_error)
    message(FATAL_ERROR "Failed to find NPM: " ${had_error})
endif()

# setup ARCH
if (APPLE)
    if (CMAKE_OSX_ARCHITECTURES_LEN GREATER 1)
        message(FATAL_ERROR "CMake.js does not support multi-architecture for macOS")
    endif()
    if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
        set(NODEJS_BINDING_ARCH arm64)
    else()
        set(NODEJS_BINDING_ARCH x64)
    endif()
elseif (WIN32)
    if (NOT MSVC)
        message(FATAL_ERROR "Only support MSVC for building Node.js binding on Windows.")
    endif()
    if(onnxruntime_target_platform STREQUAL "ARM64")
        set(NODEJS_BINDING_ARCH arm64)
    elseif(onnxruntime_target_platform STREQUAL "x64")
        set(NODEJS_BINDING_ARCH x64)
    else()
        message(FATAL_ERROR "Unsupported target platform for Node.js binding:" ${onnxruntime_target_platform})
    endif()
else()
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(NODEJS_BINDING_ARCH arm64)
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(NODEJS_BINDING_ARCH x64)
    else()
        message(FATAL_ERROR "Unsupported target platform for Node.js binding:" ${onnxruntime_target_platform})
    endif()
endif()

# a list of DLLs that the Node.js binding depends on
set(NODEJS_DLL_DEPS)

# setup providers
if (onnxruntime_USE_CUDA)
    set(NODEJS_BINDING_USE_CUDA "--use_cuda")
endif()
if (onnxruntime_USE_DML)
    set(NODEJS_BINDING_USE_DML "--use_dml")
    list(APPEND NODEJS_DLL_DEPS "$<TARGET_FILE_DIR:onnxruntime>/DirectML.dll")
endif()
if (onnxruntime_USE_WEBGPU)
    set(NODEJS_BINDING_USE_WEBGPU "--use_webgpu")
    if (WIN32 AND onnxruntime_ENABLE_DAWN_BACKEND_D3D12)
        list(APPEND NODEJS_DLL_DEPS "$<TARGET_FILE_DIR:dxcompiler>/dxil.dll")
        list(APPEND NODEJS_DLL_DEPS "$<TARGET_FILE_DIR:dxcompiler>/dxcompiler.dll")
    endif()
    if (onnxruntime_BUILD_DAWN_MONOLITHIC_LIBRARY)
        list(APPEND NODEJS_DLL_DEPS "$<TARGET_FILE:dawn::webgpu_dawn>")
    endif()
endif()
if (onnxruntime_USE_TENSORRT)
    set(NODEJS_BINDING_USE_TENSORRT "--use_tensorrt")
endif()
if (onnxruntime_USE_COREML)
    set(NODEJS_BINDING_USE_COREML "--use_coreml")
endif()
if (onnxruntime_USE_QNN)
    set(NODEJS_BINDING_USE_QNN "--use_qnn")
endif()

if(NOT onnxruntime_ENABLE_STATIC_ANALYSIS)
# add custom target
add_custom_target(js_npm_ci ALL
    COMMAND ${NPM_CLI} ci
    WORKING_DIRECTORY ${JS_ROOT}
    COMMENT "NPM install on /js")

add_custom_target(js_common_npm_ci ALL
    COMMAND ${NPM_CLI} ci
    WORKING_DIRECTORY ${JS_COMMON_ROOT}
    COMMENT "NPM install on /js/common")

add_custom_target(nodejs_binding_wrapper ALL
    COMMAND ${NPM_CLI} ci
    COMMAND ${NPM_CLI} run build -- "--onnxruntime-build-dir=${CMAKE_CURRENT_BINARY_DIR}"
        --config=${CMAKE_BUILD_TYPE}
        "--onnxruntime-generator=${CMAKE_GENERATOR}"
        "--dll_deps=${NODEJS_DLL_DEPS}"
        --arch=${NODEJS_BINDING_ARCH} ${NODEJS_BINDING_USE_CUDA} ${NODEJS_BINDING_USE_DML} ${NODEJS_BINDING_USE_WEBGPU}
        ${NODEJS_BINDING_USE_TENSORRT} ${NODEJS_BINDING_USE_COREML} ${NODEJS_BINDING_USE_QNN}
    WORKING_DIRECTORY ${JS_NODE_ROOT}
    COMMENT "Using cmake-js to build OnnxRuntime Node.js binding")

add_dependencies(js_common_npm_ci js_npm_ci)
add_dependencies(nodejs_binding_wrapper js_common_npm_ci)
add_dependencies(nodejs_binding_wrapper onnxruntime)
endif()
