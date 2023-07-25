# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT APPLE)
    message(FATAL_ERROR "The Objective-C API must be built on an Apple platform.")
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

set(OBJC_ROOT "${REPO_ROOT}/objectivec")

# onnxruntime_objc target

# these headers are the public interface
file(GLOB onnxruntime_objc_headers CONFIGURE_DEPENDS
    "${OBJC_ROOT}/include/*.h")

file(GLOB onnxruntime_objc_srcs CONFIGURE_DEPENDS
    "${OBJC_ROOT}/*.h"
    "${OBJC_ROOT}/*.m"
    "${OBJC_ROOT}/*.mm")

if(NOT onnxruntime_ENABLE_TRAINING_APIS)
    list(REMOVE_ITEM onnxruntime_objc_headers
        "${OBJC_ROOT}/include/ort_checkpoint.h"
        "${OBJC_ROOT}/include/ort_training_session.h")

    list(REMOVE_ITEM onnxruntime_objc_srcs
        "${OBJC_ROOT}/ort_checkpoint_internal.h"
        "${OBJC_ROOT}/ort_checkpoint.mm"
        "${OBJC_ROOT}/ort_training_session_internal.h"
        "${OBJC_ROOT}/ort_training_session.mm")
endif()


source_group(TREE "${OBJC_ROOT}" FILES
    ${onnxruntime_objc_headers}
    ${onnxruntime_objc_srcs})

onnxruntime_add_shared_library(onnxruntime_objc
    ${onnxruntime_objc_headers}
    ${onnxruntime_objc_srcs})

target_include_directories(onnxruntime_objc
    PUBLIC
        "${OBJC_ROOT}/include"
    PRIVATE
        "${ONNXRUNTIME_INCLUDE_DIR}/core/session"
        "${OBJC_ROOT}")

if(onnxruntime_USE_COREML)
    target_include_directories(onnxruntime_objc
        PRIVATE
            "${ONNXRUNTIME_INCLUDE_DIR}/core/providers/coreml")
endif()

if (onnxruntime_ENABLE_TRAINING_APIS)
    target_include_directories(onnxruntime_objc
        PRIVATE
            "${ORTTRAINING_SOURCE_DIR}/training_api/include/")

endif()

find_library(FOUNDATION_LIB Foundation REQUIRED)

target_link_libraries(onnxruntime_objc
    PRIVATE
        onnxruntime
        ${FOUNDATION_LIB})

set_target_properties(onnxruntime_objc PROPERTIES
    FRAMEWORK TRUE
    VERSION "1.0.0"
    SOVERSION "1.0.0"
    FRAMEWORK_VERSION "A"
    PUBLIC_HEADER "${onnxruntime_objc_headers}"
    FOLDER "ONNXRuntime"
    )

set_property(TARGET onnxruntime_objc APPEND PROPERTY COMPILE_OPTIONS "-fvisibility=default")

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

    file(GLOB onnxruntime_objc_test_srcs CONFIGURE_DEPENDS
        "${OBJC_ROOT}/test/*.h"
        "${OBJC_ROOT}/test/*.m"
        "${OBJC_ROOT}/test/*.mm")

    if(NOT onnxruntime_ENABLE_TRAINING_APIS)
        list(REMOVE_ITEM onnxruntime_objc_test_srcs
            "${OBJC_ROOT}/test/ort_checkpoint_test.mm"
            "${OBJC_ROOT}/test/ort_training_session_test.mm"
            "${OBJC_ROOT}/test/ort_training_utils_test.mm")

    endif()

    source_group(TREE "${OBJC_ROOT}" FILES ${onnxruntime_objc_test_srcs})

    xctest_add_bundle(onnxruntime_objc_test onnxruntime_objc
        ${onnxruntime_objc_headers}
        ${onnxruntime_objc_test_srcs})

    onnxruntime_configure_target(onnxruntime_objc_test)

    target_include_directories(onnxruntime_objc_test
        PRIVATE
            "${OBJC_ROOT}")

    set_target_properties(onnxruntime_objc_test PROPERTIES
        FOLDER "ONNXRuntimeTest"
        XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO")

    add_custom_command(TARGET onnxruntime_objc_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${OBJC_ROOT}/test/testdata"
            "${ONNXRUNTIME_ROOT}/test/testdata/training_api"
            "$<TARGET_BUNDLE_CONTENT_DIR:onnxruntime_objc_test>/Resources")

    xctest_add_test(XCTest.onnxruntime_objc_test onnxruntime_objc_test)

    set_property(TEST XCTest.onnxruntime_objc_test APPEND PROPERTY
        ENVIRONMENT "DYLD_LIBRARY_PATH=$<TARGET_FILE_DIR:onnxruntime>")
endif()
