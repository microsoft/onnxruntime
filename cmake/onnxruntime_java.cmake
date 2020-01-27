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
set(JAVA_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/java)
if (onnxruntime_RUN_ONNX_TESTS)
  set(JAVA_DEPENDS onnxruntime ${test_data_target})
else()
  set(JAVA_DEPENDS onnxruntime)
endif()

# use the gradle wrapper if it exists
if(EXISTS "${JAVA_ROOT}/gradlew")
    set(GRADLE_EXECUTABLE "${JAVA_ROOT}/gradlew")
else()
    # fall back to gradle on our PATH
    find_program(GRADLE_EXECUTABLE gradle)
    if(NOT GRADLE_EXECUTABLE)
        message(SEND_ERROR "Gradle installation not found")
    endif()
endif()
message(STATUS "Using gradle: ${GRADLE_EXECUTABLE}")

# Specify the Java source files
file(GLOB_RECURSE onnxruntime4j_gradle_files "${JAVA_ROOT}/*.gradle")
file(GLOB_RECURSE onnxruntime4j_src "${JAVA_ROOT}/src/main/java/ai/onnxruntime/*.java")
set(JAVA_OUTPUT_JAR ${JAVA_ROOT}/build/libs/onnxruntime.jar)
# this jar is solely used to signalling mechanism for dependency management in CMake
# if any of the Java sources change, the jar (and generated headers) will be regenerated and the onnxruntime4j_jni target will be rebuilt
add_custom_command(OUTPUT ${JAVA_OUTPUT_JAR} COMMAND ${GRADLE_EXECUTABLE} clean jar WORKING_DIRECTORY ${JAVA_ROOT} DEPENDS ${onnxruntime4j_gradle_files} ${onnxruntime4j_src})
add_custom_target(onnxruntime4j DEPENDS ${JAVA_OUTPUT_JAR})
set_source_files_properties(${JAVA_OUTPUT_JAR} PROPERTIES GENERATED TRUE)
set_property(TARGET onnxruntime4j APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${JAVA_OUTPUT_DIR}")

# Specify the native sources
file(GLOB onnxruntime4j_native_src 
    "${JAVA_ROOT}/src/main/native/*.c"
    "${JAVA_ROOT}/src/main/native/*.h"
    "${REPO_ROOT}/include/onnxruntime/core/session/*.h"
    )
# Build the JNI library
add_library(onnxruntime4j_jni SHARED ${onnxruntime4j_native_src})
# depend on java sources. if they change, the JNI should recompile
add_dependencies(onnxruntime4j_jni onnxruntime4j)
onnxruntime_add_include_to_target(onnxruntime4j_jni onnxruntime_session)
# the JNI headers are generated in the onnxruntime4j target
target_include_directories(onnxruntime4j_jni PRIVATE ${REPO_ROOT}/include ${JAVA_ROOT}/build/headers)
target_link_libraries(onnxruntime4j_jni PUBLIC ${JNI_LIBRARIES} onnxruntime)

# expose native libraries to the gradle build process
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
add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${GRADLE_EXECUTABLE} cmakeBuild -DcmakeBuildDir=${CMAKE_CURRENT_BINARY_DIR} WORKING_DIRECTORY ${JAVA_ROOT})