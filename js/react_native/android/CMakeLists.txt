project(OnnxruntimeJSIHelper)
cmake_minimum_required(VERSION 3.9.0)

set (PACKAGE_NAME "onnxruntime-react-native")
set (BUILD_DIR ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

add_library(onnxruntimejsihelper
        SHARED
        src/main/cpp/cpp-adapter.cpp
)

# Configure C++ 17
set_target_properties(
        onnxruntimejsihelper PROPERTIES
        CXX_STANDARD 17
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
)


find_package(ReactAndroid REQUIRED CONFIG)
find_library(log-lib log)

target_link_libraries(
        onnxruntimejsihelper
        ${log-lib}          # <-- Logcat logger
        ReactAndroid::jsi   # <-- JSI
        android             # <-- Android JNI core
)