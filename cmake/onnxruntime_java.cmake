# Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
# Licensed under the MIT License.

#set(CMAKE_VERBOSE_MAKEFILE on)

# Setup Java compilation
include(FindJava)
find_package(Java REQUIRED)
find_package(JNI REQUIRED)
include(UseJava)
include_directories(${JNI_INCLUDE_DIRS})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")

set(JAVA_ROOT ${REPO_ROOT}/java)
set(CMAKE_JAVA_COMPILE_FLAGS "-source" "1.8" "-target" "1.8" "-encoding" "UTF-8")
if (onnxruntime_RUN_ONNX_TESTS)
  set(JAVA_DEPENDS onnxruntime ${test_data_target})
else()
  set(JAVA_DEPENDS onnxruntime)
endif()

# Specify the Java source files
file(GLOB onnxruntime4j_src 
    "${REPO_ROOT}/java/src/main/java/ai/onnxruntime/*.java"
    )

# Specify the native sources (without the generated headers)
file(GLOB onnxruntime4j_native_src 
    "${REPO_ROOT}/java/src/main/native/*.c"
    "${REPO_ROOT}/java/src/main/native/*.h"
    "${REPO_ROOT}/include/onnxruntime/core/session/*.h"
    )
include_directories(${REPO_ROOT}/java/build/jni-headers)
# Build the JNI library
add_library(onnxruntime4j_jni SHARED ${onnxruntime4j_native_src})
onnxruntime_add_include_to_target(onnxruntime4j_jni onnxruntime_session)
target_include_directories(onnxruntime4j_jni PRIVATE ${REPO_ROOT}/include ${REPO_ROOT}/java/src/main/native)
target_link_libraries(onnxruntime4j_jni PUBLIC ${JNI_LIBRARIES} onnxruntime)

set(JAVA_PACKAGE_DIR ai/onnxruntime/native/)
set(JAVA_NATIVE_LIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/java/native-lib)
set(JAVA_NATIVE_JNILIB_DIR ${CMAKE_CURRENT_BINARY_DIR}/java/native-jnilib)
file(MAKE_DIRECTORY ${JAVA_NATIVE_LIB_DIR}/${JAVA_PACKAGE_DIR})
file(MAKE_DIRECTORY ${JAVA_NATIVE_JNILIB_DIR}/${JAVA_PACKAGE_DIR})
add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime> ${JAVA_NATIVE_LIB_DIR}/${JAVA_PACKAGE_DIR})
add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime4j_jni> ${JAVA_NATIVE_JNILIB_DIR}/${JAVA_PACKAGE_DIR})
add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ./gradlew runBuild -DcmakeBuildDir=${CMAKE_CURRENT_BINARY_DIR} WORKING_DIRECTORY ${REPO_ROOT}/java)