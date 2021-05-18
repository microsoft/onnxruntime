# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT APPLE)
    message(FATAL_ERROR "The Objective-C API must be built on an Apple platform.")
endif()

if(CMAKE_VERSION VERSION_LESS "3.18")
    message(FATAL_ERROR "The Objective-C API requires CMake 3.18+.")
endif()

if(NOT onnxruntime_BUILD_SHARED_LIB)
    message(FATAL_ERROR "The Objective-C API requires onnxruntime_BUILD_SHARED_LIB to be enabled.")
endif()

check_language(OBJC)
if(CMAKE_OBJC_COMPILER)
    enable_language(OBJC)
else()
    message(FATAL_ERROR "Objective-C is not supported.")
endif()

check_language(OBJCXX)
if(CMAKE_OBJCXX_COMPILER)
    enable_language(OBJCXX)
else()
    message(FATAL_ERROR "Objective-C++ is not supported.")
endif()

add_compile_options(
    "$<$<COMPILE_LANGUAGE:OBJC,OBJCXX>:-Wall>"
    "$<$<COMPILE_LANGUAGE:OBJC,OBJCXX>:-Wextra>")
if(onnxruntime_DEV_MODE)
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:OBJC,OBJCXX>:-Werror>")
endif()

set(OBJC_ROOT "${REPO_ROOT}/objectivec")

set(OBJC_ARC_COMPILE_OPTIONS "-fobjc-arc" "-fobjc-arc-exceptions")

# onnxruntime_objc target

# these headers are the public interface
# explicitly list them here so it is easy to see what is included
set(onnxruntime_objc_headers
    "${OBJC_ROOT}/include/onnxruntime.h"
    "${OBJC_ROOT}/include/ort_enums.h"
    "${OBJC_ROOT}/include/ort_env.h"
    "${OBJC_ROOT}/include/ort_session.h"
    "${OBJC_ROOT}/include/ort_value.h"
    )

file(GLOB onnxruntime_objc_srcs
    "${OBJC_ROOT}/src/*.h"
    "${OBJC_ROOT}/src/*.m"
    "${OBJC_ROOT}/src/*.mm")

# files common to implementation and test targets
set(onnxruntime_objc_common_srcs
    "${OBJC_ROOT}/common/assert_arc_enabled.mm")

source_group(TREE "${OBJC_ROOT}" FILES
    ${onnxruntime_objc_headers}
    ${onnxruntime_objc_srcs}
    ${onnxruntime_objc_common_srcs})

add_library(onnxruntime_objc SHARED
    ${onnxruntime_objc_headers}
    ${onnxruntime_objc_srcs}
    ${onnxruntime_objc_common_srcs})

target_include_directories(onnxruntime_objc
    PUBLIC
        "${OBJC_ROOT}/include"
    PRIVATE
        "${ONNXRUNTIME_ROOT}"
        "${OBJC_ROOT}")

find_library(FOUNDATION_LIB Foundation REQUIRED)

target_link_libraries(onnxruntime_objc
    PRIVATE
        onnxruntime
        safeint_interface
        ${FOUNDATION_LIB})

target_compile_options(onnxruntime_objc PRIVATE ${OBJC_ARC_COMPILE_OPTIONS})

set_target_properties(onnxruntime_objc PROPERTIES
    FRAMEWORK TRUE
    VERSION "1.0.0"
    SOVERSION "1.0.0"
    FRAMEWORK_VERSION "A"
    PUBLIC_HEADER "${onnxruntime_objc_headers}"
    FOLDER "ONNXRuntime"
    CXX_STANDARD 17 # TODO remove when everything else moves to 17
    )

target_link_options(onnxruntime_objc PRIVATE "-Wl,-headerpad_max_install_names")

add_custom_command(TARGET onnxruntime_objc POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_BUNDLE_CONTENT_DIR:onnxruntime_objc>/Libraries"
    COMMAND ${CMAKE_COMMAND} -E copy
        "$<TARGET_FILE:onnxruntime>"
        "$<TARGET_BUNDLE_CONTENT_DIR:onnxruntime_objc>/Libraries"
    COMMAND install_name_tool
        -change "@rpath/$<TARGET_FILE_NAME:onnxruntime>"
                "@rpath/$<TARGET_NAME:onnxruntime_objc>.framework/Libraries/$<TARGET_FILE_NAME:onnxruntime>"
        "$<TARGET_FILE:onnxruntime_objc>")

install(TARGETS onnxruntime_objc
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})

if(onnxruntime_BUILD_UNIT_TESTS)
    find_package(XCTest REQUIRED)

    # onnxruntime_objc_test target

    file(GLOB onnxruntime_objc_test_srcs
        "${OBJC_ROOT}/test/*.h"
        "${OBJC_ROOT}/test/*.m"
        "${OBJC_ROOT}/test/*.mm")

    source_group(TREE "${OBJC_ROOT}" FILES ${onnxruntime_objc_test_srcs})

    xctest_add_bundle(onnxruntime_objc_test onnxruntime_objc
        ${onnxruntime_objc_headers}
        ${onnxruntime_objc_test_srcs}
        ${onnxruntime_objc_common_srcs})

    target_include_directories(onnxruntime_objc_test
        PRIVATE
            "${OBJC_ROOT}")

    target_compile_options(onnxruntime_objc_test PRIVATE ${OBJC_ARC_COMPILE_OPTIONS})

    set_target_properties(onnxruntime_objc_test PROPERTIES
        FOLDER "ONNXRuntimeTest"
        XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO")

    add_custom_command(TARGET onnxruntime_objc_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${OBJC_ROOT}/test/testdata"
            "$<TARGET_BUNDLE_CONTENT_DIR:onnxruntime_objc_test>/Resources")

    xctest_add_test(XCTest.onnxruntime_objc_test onnxruntime_objc_test)

    set_property(TEST XCTest.onnxruntime_objc_test APPEND PROPERTY
        ENVIRONMENT "DYLD_LIBRARY_PATH=$<TARGET_FILE_DIR:onnxruntime>")
endif()
