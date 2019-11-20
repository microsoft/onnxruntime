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

if (onnxruntime_USE_CUDA)
  STRING(APPEND JAVA_PREPROCESSOR_DEFINES "USE_CUDA,")
endif()

if (onnxruntime_USE_MKLDNN)
  STRING(APPEND JAVA_PREPROCESSOR_DEFINES "USE_MKLDNN,")
endif()

if (onnxruntime_USE_TENSORRT)
  STRING(APPEND JAVA_PREPROCESSOR_DEFINES "USE_TENSORRT,")
endif()

if (onnxruntime_USE_OPENVINO)
  STRING(APPEND JAVA_PREPROCESSOR_DEFINES "USE_OPENVINO,")
endif()

if (onnxruntime_USE_NGRAPH)
  STRING(APPEND JAVA_PREPROCESSOR_DEFINES "USE_NGRAPH,")
endif()

if (onnxruntime_USE_NUPHAR)
  STRING(APPEND JAVA_PREPROCESSOR_DEFINES "USE_NUPHAR,")
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

add_jar(onnxruntime4j SOURCES ${onnxruntime4j_src} GENERATE_NATIVE_HEADERS onnxruntime4j_generated DESTINATION ${REPO_ROOT}/java/src/main/native/)

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
