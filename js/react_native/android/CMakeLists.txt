project(OnnxruntimeJSIHelper)
cmake_minimum_required(VERSION 3.9.0)

set (PACKAGE_NAME "onnxruntime-react-native")
set (BUILD_DIR ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

file(TO_CMAKE_PATH "${NODE_MODULES_DIR}/react-native/ReactCommon/jsi/jsi/jsi.cpp" libPath)

include_directories(
        "${NODE_MODULES_DIR}/react-native/React"
        "${NODE_MODULES_DIR}/react-native/React/Base"
        "${NODE_MODULES_DIR}/react-native/ReactCommon/jsi"
)

add_library(onnxruntimejsihelper
        SHARED
        ${libPath}
        src/main/cpp/cpp-adapter.cpp
)

# Configure C++ 17
set_target_properties(
        onnxruntimejsihelper PROPERTIES
        CXX_STANDARD 17
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
)

find_library(log-lib log)

target_link_libraries(
        onnxruntimejsihelper
        ${log-lib}          # <-- Logcat logger
        android             # <-- Android JNI core
)
