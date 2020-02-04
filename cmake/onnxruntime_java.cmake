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
set(onnxruntime4j_src
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/MapInfo.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/NodeInfo.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/OnnxRuntime.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/OnnxJavaType.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/OnnxMap.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/OnnxSequence.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/OnnxTensor.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/OnnxValue.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/OrtAllocator.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/OrtEnvironment.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/OrtException.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/OrtSession.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/OrtUtil.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/package-info.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/SequenceInfo.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/TensorInfo.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/ValueInfo.java
        )

# Build the jar and generate the native headers
add_jar(onnxruntime4j SOURCES ${onnxruntime4j_src} VERSION ${ORT_VERSION} GENERATE_NATIVE_HEADERS onnxruntime4j_generated DESTINATION ${REPO_ROOT}/java/src/main/native/)

# Specify the native sources (without the generated headers)
file(GLOB onnxruntime4j_native_src 
    "${REPO_ROOT}/java/src/main/native/*.c"
    "${REPO_ROOT}/java/src/main/native/OrtJniUtil.h"
    "${REPO_ROOT}/include/onnxruntime/core/session/*.h"
    )

# Build the JNI library
add_library(onnxruntime4j_jni SHARED ${onnxruntime4j_native_src} ${onnxruntime4j_generated})
onnxruntime_add_include_to_target(onnxruntime4j_jni onnxruntime_session)
target_include_directories(onnxruntime4j_jni PRIVATE ${REPO_ROOT}/include ${REPO_ROOT}/java/src/main/native)
target_link_libraries(onnxruntime4j_jni PUBLIC onnxruntime onnxruntime4j_generated)

# Now the jar, jni binary and shared lib binary have been built, now to build the jar with the binaries added.

# This blob creates the new jar name
get_property(onnxruntime_jar_name TARGET onnxruntime4j PROPERTY JAR_FILE)
get_filename_component(onnxruntime_jar_abs ${onnxruntime_jar_name} ABSOLUTE)
get_filename_component(jar_path ${onnxruntime_jar_abs} DIRECTORY)
set(onnxruntime_jar_binaries_name "${jar_path}/onnxruntime4j-${ORT_VERSION}-with-binaries.jar")
set(onnxruntime_jar_binaries_platform "$<SHELL_PATH:${onnxruntime_jar_binaries_name}>")

# Copy the current jar
add_custom_command(TARGET onnxruntime4j_jni PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_jar_name}
        ${onnxruntime_jar_binaries_platform})

# Make a temp directory to store the binaries
add_custom_command(TARGET onnxruntime4j_jni POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/java-libs/lib")

# Copy the binaries
add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:onnxruntime4j_jni>" ${CMAKE_CURRENT_BINARY_DIR}/java-libs/lib/)

if (WIN32) 
add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:onnxruntime>" ${CMAKE_CURRENT_BINARY_DIR}/java-libs/lib/)
# Update the with-binaries jar so it includes the binaries
add_custom_command(
            TARGET onnxruntime4j_jni POST_BUILD
            COMMAND ${Java_JAR_EXECUTABLE} -uf ${onnxruntime_jar_binaries_platform} -C ${CMAKE_CURRENT_BINARY_DIR}/java-libs lib/$<TARGET_FILE_NAME:onnxruntime4j_jni> -C ${CMAKE_CURRENT_BINARY_DIR}/java-libs lib/$<TARGET_FILE_NAME:onnxruntime>
            DEPENDS onnxruntime4j
            COMMENT "Rebuilding Java archive ${_JAVA_TARGET_OUTPUT_NAME}"
            VERBATIM
        )
else ()
add_custom_command(TARGET onnxruntime4j_jni POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_LINKER_FILE:onnxruntime>" ${CMAKE_CURRENT_BINARY_DIR}/java-libs/lib/)
# Update the with-binaries jar so it includes the binaries
add_custom_command(
            TARGET onnxruntime4j_jni POST_BUILD
            COMMAND ${Java_JAR_EXECUTABLE} -uf ${onnxruntime_jar_binaries_platform} -C ${CMAKE_CURRENT_BINARY_DIR}/java-libs lib/$<TARGET_FILE_NAME:onnxruntime4j_jni> -C ${CMAKE_CURRENT_BINARY_DIR}/java-libs lib/$<TARGET_LINKER_FILE_NAME:onnxruntime>
            DEPENDS onnxruntime4j
            COMMENT "Rebuilding Java archive ${_JAVA_TARGET_OUTPUT_NAME}"
            VERBATIM
        )
endif()

create_javadoc(onnxruntime4j_javadoc
           FILES ${onnxruntime4j_src}
           DOCTITLE "Onnx Runtime Java API"
           WINDOWTITLE "OnnxRuntime-Java-API"
           AUTHOR FALSE
           USE TRUE
           VERSION FALSE
           )
