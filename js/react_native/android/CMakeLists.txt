project(OnnxruntimeJSI)
cmake_minimum_required(VERSION 3.9.0)

set (PACKAGE_NAME "onnxruntime-react-native")
set (BUILD_DIR ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

if (ORT_ENABLE_EXTENSIONS)
        add_definitions(-DORT_ENABLE_EXTENSIONS)
        find_library(onnxruntime-extensions-lib ortextensions)
endif()

file(TO_CMAKE_PATH "${NODE_MODULES_DIR}/react-native/ReactCommon/jsi/jsi/jsi.cpp" libPath)

include_directories(
        "${NODE_MODULES_DIR}/react-native/React"
        "${NODE_MODULES_DIR}/react-native/React/Base"
        "${NODE_MODULES_DIR}/react-native/ReactCommon"
        "${NODE_MODULES_DIR}/react-native/ReactCommon/jsi"
        "${NODE_MODULES_DIR}/react-native/ReactCommon/callinvoker"
        ../cpp
)

add_library(onnxruntimejsi
        SHARED
        ${libPath}
        src/main/cpp/cpp-adapter.cpp
        ../cpp/JsiMain.cpp
        ../cpp/InferenceSessionHostObject.cpp
        ../cpp/JsiUtils.cpp
        ../cpp/SessionUtils.cpp
        ../cpp/TensorUtils.cpp
)

# Configure C++ 17
set_target_properties(
        onnxruntimejsi PROPERTIES
        CXX_STANDARD 17
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
)

find_library(log-lib log)
find_library(onnxruntime-lib onnxruntime)
find_library(fbjni-lib fbjni)

target_link_libraries(
        onnxruntimejsi
        ${onnxruntime-lib}
        ${onnxruntime-extensions-lib}
        ${fbjni-lib}
        ${log-lib}          # <-- Logcat logger
        android             # <-- Android JNI core
)
