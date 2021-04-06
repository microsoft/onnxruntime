# Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
# Licensed under the MIT License.

set(NODEJS_BINDING_ROOT ${REPO_ROOT}/nodejs)
if (WIN32)
    set(NPM_CLI cmd /c npm)
else()
    set(NPM_CLI npm)
endif()

# verify Node.js and NPM
execute_process(COMMAND node --version
    WORKING_DIRECTORY ${NODEJS_BINDING_ROOT}
    OUTPUT_VARIABLE node_version
    RESULT_VARIABLE had_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(had_error)
    message(FATAL_ERROR "Failed to find Node.js: " ${had_error})
endif()
execute_process(COMMAND ${NPM_CLI} --version
    WORKING_DIRECTORY ${NODEJS_BINDING_ROOT}
    OUTPUT_VARIABLE npm_version
    RESULT_VARIABLE had_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(had_error)
    message(FATAL_ERROR "Failed to find NPM: " ${had_error})
endif()

if(NOT onnxruntime_ENABLE_STATIC_ANALYSIS)
# add custom target
add_custom_target(nodejs_binding_wrapper ALL
    COMMAND ${NPM_CLI} ci --ort-skip-build
    COMMAND ${NPM_CLI} run build -- --onnxruntime-build-dir=${CMAKE_CURRENT_BINARY_DIR} --config=${CMAKE_BUILD_TYPE}
    WORKING_DIRECTORY ${NODEJS_BINDING_ROOT}
    COMMENT "Using cmake-js to build OnnxRuntime Node.js binding")
add_dependencies(nodejs_binding_wrapper onnxruntime)
endif()