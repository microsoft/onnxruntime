# Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
# Licensed under the MIT License.

set(JS_ROOT ${REPO_ROOT}/js)
set(JS_COMMON_ROOT ${JS_ROOT}/common)
set(JS_NODE_ROOT ${JS_ROOT}/node)
if (WIN32)
    set(NPM_CLI cmd /c npm)
else()
    set(NPM_CLI npm)
endif()

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
if (APPLE AND CMAKE_OSX_ARCHITECTURES_LEN GREATER 1)
    message(FATAL_ERROR "CMake.js does not support multi-architecture for macOS")
endif()
if (APPLE AND CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    set(NODEJS_BINDING_ARCH arm64)
# elseif()
else()
    set(NODEJS_BINDING_ARCH x64)
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
    COMMAND ${NPM_CLI} run build -- --onnxruntime-build-dir=${CMAKE_CURRENT_BINARY_DIR} --config=${CMAKE_BUILD_TYPE} --arch=${NODEJS_BINDING_ARCH}
    WORKING_DIRECTORY ${JS_NODE_ROOT}
    COMMENT "Using cmake-js to build OnnxRuntime Node.js binding")
add_dependencies(js_common_npm_ci js_npm_ci)
add_dependencies(nodejs_binding_wrapper js_common_npm_ci)
add_dependencies(nodejs_binding_wrapper onnxruntime)
endif()