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

if(WIN32 AND NODEJS_BINDING_ARCH STREQUAL "arm64")
    # NOTE:
    # Node.js for Windows ARM64 is not available in official release list.
    # We uses node.lib for Node.js ARM64 v16.2.0 unofficial build from:
    # https://unofficial-builds.nodejs.org/download/release/v16.2.0/win-arm64/
    # The file should be downloaded and saved to ${CMAKE_CURRENT_BINARY_DIR}
    #
    # once Node.js for Windows ARM64 is in official release list and cmake-js supports Windows ARM64,
    # we can remove this section.
    include(FetchContent)
    FetchContent_Declare(
        nodejs_win_arm64_node_lib
        URL  https://unofficial-builds.nodejs.org/download/release/v16.2.0/win-arm64/node.lib
        URL_HASH SHA256=decbf5c6f7e048af34add43da5eaa75b309340acc17b2670b8289a47dd751401
        DOWNLOAD_NO_EXTRACT TRUE
    )
    FetchContent_MakeAvailable(nodejs_win_arm64_node_lib)
    FetchContent_Declare(
        nodejs_win_arm64_headers
        URL  https://unofficial-builds.nodejs.org/download/release/v16.2.0/node-v16.2.0-headers.tar.gz
        URL_HASH SHA256=4f6a1e877c65ce31217b1c67f4fd90b9f9f4d3051e5ffabf151f43dd6edf7a34
    )
    FetchContent_MakeAvailable(nodejs_win_arm64_headers)

    add_custom_target(nodejs_binding_wrapper ALL
        COMMAND ${NPM_CLI} ci
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${JS_NODE_ROOT}/build/
        COMMAND ${CMAKE_COMMAND} -E make_directory ${JS_NODE_ROOT}/build/
        COMMAND cd ${JS_NODE_ROOT} && ${CMAKE_COMMAND} ${JS_NODE_ROOT} --no-warn-unused-cli -G"Visual Studio 16 2019" -A"ARM64" -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="${JS_NODE_ROOT}/build/" -DCMAKE_JS_INC="${nodejs_win_arm64_headers_SOURCE_DIR}/include/node" -DCMAKE_JS_SRC="${JS_NODE_ROOT}/node_modules/cmake-js/lib/cpp/win_delay_load_hook.cc" -DNODE_RUNTIME="node" -DNODE_RUNTIMEVERSION="16.2.0" -DNODE_ARCH="arm64" -DCMAKE_JS_LIB="${nodejs_win_arm64_node_lib_SOURCE_DIR}/node.lib" -Dnapi_build_version="3" -DCMAKE_BUILD_TYPE="RelWithDebInfo" -DONNXRUNTIME_BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}" -DCMAKE_SHARED_LINKER_FLAGS="/DELAYLOAD:NODE.EXE" -B ./build
        COMMAND ${CMAKE_COMMAND} --build ${JS_NODE_ROOT}/build/ --config RelWithDebInfo
        WORKING_DIRECTORY ${JS_NODE_ROOT}
        COMMENT "Using custom script to build OnnxRuntime Node.js binding")
else()
    add_custom_target(nodejs_binding_wrapper ALL
        COMMAND ${NPM_CLI} ci
        COMMAND ${NPM_CLI} run build -- --onnxruntime-build-dir=${CMAKE_CURRENT_BINARY_DIR} --config=${CMAKE_BUILD_TYPE} --arch=${NODEJS_BINDING_ARCH}
        WORKING_DIRECTORY ${JS_NODE_ROOT}
        COMMENT "Using cmake-js to build OnnxRuntime Node.js binding")
endif()
add_dependencies(js_common_npm_ci js_npm_ci)
add_dependencies(nodejs_binding_wrapper js_common_npm_ci)
add_dependencies(nodejs_binding_wrapper onnxruntime)
endif()