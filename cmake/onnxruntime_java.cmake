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

# Build the jar and generate the native headers
add_jar(onnxruntime4j SOURCES ${onnxruntime4j_src} VERSION ${ORT_VERSION} GENERATE_NATIVE_HEADERS onnxruntime4j_generated DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/java-headers/)

# Specify the native sources (without the generated headers)
file(GLOB onnxruntime4j_native_src 
    "${REPO_ROOT}/java/src/main/native/*.c"
    "${REPO_ROOT}/java/src/main/native/*.h"
    "${REPO_ROOT}/include/onnxruntime/core/session/*.h"
    )

# Build the JNI library
add_library(onnxruntime4j_jni SHARED ${onnxruntime4j_native_src} ${onnxruntime4j_generated})
onnxruntime_add_include_to_target(onnxruntime4j_jni onnxruntime_session)
target_include_directories(onnxruntime4j_jni PRIVATE ${REPO_ROOT}/include ${REPO_ROOT}/java/src/main/native)
target_link_libraries(onnxruntime4j_jni PUBLIC ${JNI_LIBRARIES} onnxruntime onnxruntime4j_generated)

# Copy the library and jni library over to gradle into the proper os-arch directory
if(APPLE)
	set(GRADLE_CLASSIFIER_OS macosx)
endif()
if(WIN32)
	set(GRADLE_CLASSIFIER_OS windows)
endif()
if(UNIX AND NOT APPLE)
	set(GRADLE_CLASSIFIER_OS linux)
endif()

set(GRADLE_SUBPROJECT cpu)
if (onnxruntime_USE_CUDA)
	set(GRADLE_SUBPROJECT gpu)
endif()

if(GRADLE_CLASSIFIER_OS)
	set(GRADLE_NATIVE_DIR build/jni-output/${GRADLE_CLASSIFIER_OS}-${CMAKE_SYSTEM_PROCESSOR}/ai/onnxruntime/native/)
	set(GRADLE_NATIVE_JNILIB_DIR ${REPO_ROOT}/java/${GRADLE_NATIVE_DIR})
	set(GRADLE_NATIVE_LIB_DIR ${REPO_ROOT}/java/runtime-${GRADLE_SUBPROJECT}/${GRADLE_NATIVE_DIR})
	add_custom_target(onnxruntime4j_gradle ALL DEPENDS onnxruntime4j_gradle_lib onnxruntime4j_gradle_jnilib)
	file(MAKE_DIRECTORY ${GRADLE_NATIVE_JNILIB_DIR})
	file(MAKE_DIRECTORY ${GRADLE_NATIVE_LIB_DIR})
	add_custom_command(OUTPUT onnxruntime4j_gradle_jnilib COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_LINKER_FILE_NAME:onnxruntime4j_jni> ${GRADLE_NATIVE_JNILIB_DIR}$<TARGET_LINKER_FILE_NAME:onnxruntime4j_jni> DEPENDS onnxruntime4j_jni)
	add_custom_command(OUTPUT onnxruntime4j_gradle_lib COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_LINKER_FILE_NAME:onnxruntime> ${GRADLE_NATIVE_LIB_DIR}$<TARGET_LINKER_FILE_NAME:onnxruntime> DEPENDS onnxruntime)
endif()
