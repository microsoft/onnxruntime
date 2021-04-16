# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if (${CMAKE_VERSION} VERSION_LESS "3.16")
    message(FATAL_ERROR "CMake 3.16+ is required when building the Objective-C API.")
endif()

check_language(OBJC)
if (CMAKE_OBJC_COMPILER)
    enable_language(OBJC)
else()
    message(FATAL_ERROR "Objective-C is not supported.")
endif()

check_language(OBJCXX)
if (CMAKE_OBJCXX_COMPILER)
    enable_language(OBJCXX)
else()
    message(FATAL_ERROR "Objective-C++ is not supported.")
endif()

SET(CMAKE_XCODE_ATTRIBUTE_CLANG_ENABLE_MODULES "YES")
SET(CMAKE_XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC "YES")

# onnxruntime_objc
file(GLOB onnxruntime_objc_headers
    "${REPO_ROOT}/objc/include/onnxruntime/*.h")

file(GLOB onnxruntime_objc_srcs
    "${REPO_ROOT}/objc/src/onnxruntime/*.h"
    "${REPO_ROOT}/objc/src/onnxruntime/*.m"
    "${REPO_ROOT}/objc/src/onnxruntime/*.mm")

source_group(TREE "${REPO_ROOT}/objc"
    FILES ${onnxruntime_objc_headers} ${onnxruntime_objc_srcs})

add_library(onnxruntime_objc SHARED ${onnxruntime_objc_headers} ${onnxruntime_objc_srcs})

target_include_directories(onnxruntime_objc
    PUBLIC
        "${REPO_ROOT}/objc/include"
    PRIVATE
        "${OPTIONAL_LITE_INCLUDE_DIR}"
        "${REPO_ROOT}/objc/src")

target_link_libraries(onnxruntime_objc PUBLIC onnxruntime)

set_target_properties(onnxruntime_objc PROPERTIES
    FRAMEWORK TRUE
    VERSION "1.0.0"
    SOVERSION "1.0.0"
    FRAMEWORK_VERSION "A"
    PUBLIC_HEADER "${onnxruntime_objc_headers}"
    FOLDER "ONNXRuntime")

if (onnxruntime_BUILD_UNIT_TESTS)
    find_package(XCTest REQUIRED)

    # onnxruntime_test_objc
    file(GLOB onnxruntime_objc_test_srcs
        "${REPO_ROOT}/objc/test/onnxruntime/*.h"
        "${REPO_ROOT}/objc/test/onnxruntime/*.m"
        "${REPO_ROOT}/objc/test/onnxruntime/*.mm")

    source_group(TREE "${REPO_ROOT}/objc"
        FILES ${onnxruntime_objc_test_srcs})

    xctest_add_bundle(onnxruntime_objc_test onnxruntime_objc
        ${onnxruntime_objc_test_srcs})

    set_target_properties(onnxruntime_objc_test PROPERTIES FOLDER "ONNXRuntimeTest")

    add_custom_command(TARGET onnxruntime_objc_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${REPO_ROOT}/objc/test/testdata"
            "$<TARGET_BUNDLE_CONTENT_DIR:onnxruntime_objc_test>/Resources/testdata")

    xctest_add_test(XCTest.onnxruntime_objc_test onnxruntime_objc_test)

    set_property(TEST XCTest.onnxruntime_objc_test APPEND PROPERTY
        ENVIRONMENT DYLD_LIBRARY_PATH=$<TARGET_FILE_DIR:onnxruntime>)
endif()
