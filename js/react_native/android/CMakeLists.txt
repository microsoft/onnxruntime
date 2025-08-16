project(OnnxruntimeJSI)
cmake_minimum_required(VERSION 3.9.0)

set(PACKAGE_NAME "onnxruntime-react-native")
set(BUILD_DIR ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

option(ORT_EXTENSIONS_ENABLED "Enable Ort Extensions" NO)
option(USE_NNAPI "Use NNAPI" YES)
option(USE_QNN "Use QNN" NO)

file(GLOB libfbjni_link_DIRS "${BUILD_DIR}/fbjni-*.aar/jni/${ANDROID_ABI}")
file(GLOB libfbjni_include_DIRS "${BUILD_DIR}/fbjni-*-headers.jar/")

file(GLOB onnxruntime_include_DIRS
     "${BUILD_DIR}/onnxruntime-android-*.aar/headers")
file(GLOB onnxruntime_link_DIRS
     "${BUILD_DIR}/onnxruntime-android-*.aar/jni/${ANDROID_ABI}/")

if(ORT_EXTENSIONS_ENABLED)
  add_definitions(-DORT_ENABLE_EXTENSIONS)
endif()

if(USE_QNN)
  add_definitions(-DUSE_QNN)
endif()

if(USE_NNAPI)
  add_definitions(-DUSE_NNAPI)
endif()

file(TO_CMAKE_PATH
     "${NODE_MODULES_DIR}/react-native/ReactCommon/jsi/jsi/jsi.cpp" libPath)

find_package(fbjni REQUIRED CONFIG)
find_package(ReactAndroid REQUIRED CONFIG)

find_library(
  onnxruntime-lib onnxruntime
  PATHS ${onnxruntime_link_DIRS}
  NO_CMAKE_FIND_ROOT_PATH)

include_directories(
  "${NODE_MODULES_DIR}/react-native/React"
  "${NODE_MODULES_DIR}/react-native/React/Base"
  "${NODE_MODULES_DIR}/react-native/ReactCommon"
  "${NODE_MODULES_DIR}/react-native/ReactCommon/jsi"
  "${NODE_MODULES_DIR}/react-native/ReactCommon/callinvoker"
  "${NODE_MODULES_DIR}/react-native/ReactAndroid/src/main/java/com/facebook/react/turbomodule/core/jni"
  ../cpp
  ${onnxruntime_include_DIRS}
  ${libfbjni_include_DIRS})

add_library(
  onnxruntimejsi SHARED
  ${libPath}
  src/main/cpp/cpp-adapter.cpp
  ../cpp/JsiMain.cpp
  ../cpp/InferenceSessionHostObject.cpp
  ../cpp/JsiUtils.cpp
  ../cpp/SessionUtils.cpp
  ../cpp/TensorUtils.cpp)

# Configure C++ 17
set_target_properties(
  onnxruntimejsi
  PROPERTIES CXX_STANDARD 17
             CXX_EXTENSIONS OFF
             POSITION_INDEPENDENT_CODE ON)

find_library(log-lib log)

target_link_libraries(
  onnxruntimejsi ${onnxruntime-lib} fbjni::fbjni
  ReactAndroid::turbomodulejsijni ${log-lib} # <-- Logcat logger
  android # <-- Android JNI core
)
