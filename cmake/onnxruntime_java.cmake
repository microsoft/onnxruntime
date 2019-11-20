# Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
# Licensed under the MIT License.

set(CMAKE_VERBOSE_MAKEFILE on)

include(FindJava)
find_package(Java REQUIRED)
find_package(JNI REQUIRED)
include(UseJava)
include_directories(${JNI_INCLUDE_DIRS})
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")

set (JAVA_ROOT ${REPO_ROOT}/java)
set(CMAKE_JAVA_COMPILE_FLAGS "-source" "1.8" "-target" "1.8")
if (onnxruntime_RUN_ONNX_TESTS)
    set (JAVA_DEPENDS onnxruntime ${test_data_target})
else()
  set (JAVA_DEPENDS onnxruntime)
endif()

set(onnxruntime4j_src
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/MapInfo.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/NodeInfo.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/ONNX.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/ONNXAllocator.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/ONNXEnvironment.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/ONNXException.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/ONNXJavaType.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/ONNXMap.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/ONNXSequence.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/ONNXSession.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/ONNXTensor.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/ONNXUtil.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/ONNXValue.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/SequenceInfo.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/TensorInfo.java
        ${REPO_ROOT}/java/src/main/java/ai/onnxruntime/ValueInfo.java
        )

add_jar(onnxruntime4j SOURCES ${onnxruntime4j_src} VERSION ${ORT_VERSION} GENERATE_NATIVE_HEADERS onnxruntime4j_generated DESTINATION ${REPO_ROOT}/java/src/main/native/)

file(GLOB onnxruntime4j_native_src 
    "${REPO_ROOT}/java/src/main/native/*.c"
    "${REPO_ROOT}/java/src/main/native/ONNXUtil.h"
    "${REPO_ROOT}/include/onnxruntime/core/session/*.h"
    )

add_library(onnxruntime4j_jni SHARED 
    ${onnxruntime4j_native_src}
    ${onnxruntime4j_generated}
    )

onnxruntime_add_include_to_target(onnxruntime4j_jni onnxruntime_session)

target_include_directories(onnxruntime4j_jni PRIVATE ${REPO_ROOT}/include ${REPO_ROOT}/java/src/main/native)

target_link_libraries(onnxruntime4j_jni PUBLIC 
    ${JNI_LIBRARIES}
    onnxruntime
    onnxruntime4j_generated
    )

get_property(onnxruntime_jar_name TARGET onnxruntime4j PROPERTY JAR_FILE)

# Now the jar, jni binary and shared lib binary have been built, but the jar does not contain the necessary binaries.

add_custom_target(TARGET onnxruntime4j_jni POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/java-libs/lib)

add_custom_command(
        TARGET onnxruntime4j_jni POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                "$<TARGET_FILE:onnxruntime4j_jni>"
                ${CMAKE_CURRENT_BINARY_DIR}/java-libs/lib/)

add_custom_command(
        TARGET onnxruntime4j_jni POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        "$<TARGET_LINKER_FILE:onnxruntime>"
                ${CMAKE_CURRENT_BINARY_DIR}/java-libs/lib/)

add_custom_command(
            TARGET onnxruntime4j_jni POST_BUILD
            OUTPUT ${_JAVA_JAR_OUTPUT_PATH}
            COMMAND ${Java_JAR_EXECUTABLE} -uf ${onnxruntime_jar_name} -C ${CMAKE_CURRENT_BINARY_DIR}/java-libs lib/$<TARGET_FILE_NAME:onnxruntime4j_jni> -C ${CMAKE_CURRENT_BINARY_DIR}/java-libs lib/$<TARGET_LINKER_FILE_NAME:onnxruntime>
            DEPENDS onnxruntime4j
            COMMENT "Rebuilding Java archive ${_JAVA_TARGET_OUTPUT_NAME}"
            VERBATIM
        )

