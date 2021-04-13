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

find_library(FOUNDATION_LIBRARY Foundation)

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
    #MACOSX_FRAMEWORK_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/FrameworkExample/Info.plist
    PUBLIC_HEADER "${onnxruntime_objc_headers}")

if (onnxruntime_BUILD_UNIT_TESTS)
    find_package(XCTest REQUIRED)

    # onnxruntime_test_objc
    file(GLOB onnxruntime_objc_test_srcs
        "${REPO_ROOT}/objc/test/*.h"
        "${REPO_ROOT}/objc/test/*.m"
        "${REPO_ROOT}/objc/test/*.mm")

    xctest_add_bundle(onnxruntime_objc_test onnxruntime_objc
        ${onnxruntime_objc_test_srcs}
        #FrameworkExampleTests/Info.plist
        )
      
    # set_target_properties(FrameworkExampleTests PROPERTIES
    #     MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/FrameworkExampleTests/Info.plist
    #     )
      
    xctest_add_test(XCTest.onnxruntime_objc_test onnxruntime_objc_test)
      
endif()
