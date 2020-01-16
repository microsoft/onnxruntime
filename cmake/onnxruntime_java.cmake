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
if (onnxruntime_RUN_ONNX_TESTS)
  set(JAVA_DEPENDS onnxruntime ${test_data_target})
else()
  set(JAVA_DEPENDS onnxruntime)
endif()

# Specify the Java source files
file(GLOB_RECURSE onnxruntime4j_src "${JAVA_ROOT}/src/main/java/ai/onnxruntime/*.java")
add_custom_target(onnxruntime4j SOURCES ${onnxruntime4j_src})

# Specify the native sources
file(GLOB onnxruntime4j_native_src 
    "${JAVA_ROOT}/src/main/native/*.c"
    "${JAVA_ROOT}/src/main/native/*.h"
    "${REPO_ROOT}/include/onnxruntime/core/session/*.h"
    )
# Build the JNI library
add_library(onnxruntime4j_jni SHARED ${onnxruntime4j_native_src})
add_dependencies(onnxruntime4j_jni onnxruntime4j)
onnxruntime_add_include_to_target(onnxruntime4j_jni onnxruntime_session)
target_include_directories(onnxruntime4j_jni PRIVATE ${REPO_ROOT}/include ${JAVA_ROOT}/src/main/native)
target_link_libraries(onnxruntime4j_jni PUBLIC ${JNI_LIBRARIES} onnxruntime)

# expose native libraries to the gradle build process
set(JAVA_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/java)
file(MAKE_DIRECTORY ${JAVA_OUTPUT_DIR}/build)
set(JAVA_PACKAGE_DIR ai/onnxruntime/native/)
set(JAVA_NATIVE_LIB_DIR ${JAVA_OUTPUT_DIR}/native-lib)
set(JAVA_NATIVE_JNI_DIR ${JAVA_OUTPUT_DIR}/native-jni)
set(JAVA_PACKAGE_LIB_DIR ${JAVA_NATIVE_LIB_DIR}/${JAVA_PACKAGE_DIR})
set(JAVA_PACKAGE_JNI_DIR ${JAVA_NATIVE_JNI_DIR}/${JAVA_PACKAGE_DIR})
file(MAKE_DIRECTORY ${JAVA_PACKAGE_LIB_DIR})
file(MAKE_DIRECTORY ${JAVA_PACKAGE_JNI_DIR})
add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:onnxruntime> ${JAVA_PACKAGE_LIB_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime>)
add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:onnxruntime4j_jni> ${JAVA_PACKAGE_JNI_DIR}/$<TARGET_LINKER_FILE_NAME:onnxruntime4j_jni>)
# run the build process (this copies the results back into CMAKE_CURRENT_BINARY_DIR)
add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ./gradlew cmakeBuild -DcmakeBuildDir=${CMAKE_CURRENT_BINARY_DIR} WORKING_DIRECTORY ${JAVA_ROOT})