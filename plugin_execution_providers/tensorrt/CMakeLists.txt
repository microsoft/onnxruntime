# usage:
# cd build/
# cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DORT_HOME=/home/lochi/repos/ort -DTENSORRT_HOME=/home/lochi/tensorrt/TensorRT-10.3.0.26 (see the result of "nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits")
# cmake --build ./ --config Debug
cmake_minimum_required(VERSION 3.26)
project(TensorRTEp VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
enable_language(CUDA)
file(TO_CMAKE_PATH CUDAToolkit_ROOT "/usr/local/cuda")
find_package(CUDAToolkit REQUIRED)

add_definitions(-DONNX_NAMESPACE=onnx)
add_definitions(-DONNX_ML)
add_definitions(-DNV_TENSORRT_MAJOR=10)
add_definitions(-DNOMINMAX)
file(GLOB tensorrt_src "./*.cc" "./utils/*.cc" "./cuda/unary_elementwise_ops_impl.cu")
add_library(TensorRTEp SHARED ${tensorrt_src})

if (NOT ORT_HOME)
  message(FATAL_ERROR "Please specify ORT_HOME, e.g. -DORT_HOME=/path/to/ort/")
endif()

if (NOT TENSORRT_HOME)
  message(FATAL_ERROR "Please specify TENSORRT_HOME, e.g. -DTENSORRT_HOME=/path/to/trt/")
endif()

# Use release mode if not specified
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# There is an known issue when running "Debug build" TRT EP plugin with "Release build" TRT builtin parser on Windows.
if (WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
  MESSAGE(FATAL_ERROR "[Note] There is an known issue when running \"Debug build\" TRT EP plugin with \"Release build\" TRT built-in parser on Windows. Please use release mode to build TRT EP plugin.")
endif()

if (WIN32)
  set(PLATFORM "Windows")
  set(ORT_LIB "${ORT_HOME}/build/${PLATFORM}/${CMAKE_BUILD_TYPE}/${CMAKE_BUILD_TYPE}/onnxruntime.lib")
  set(DEPS_PATH "${ORT_HOME}/build/${PLATFORM}/${CMAKE_BUILD_TYPE}/_deps")
  set(TRT_LIBS "${TENSORRT_HOME}/lib/nvinfer_10.lib" 
               "${TENSORRT_HOME}/lib/nvinfer_plugin_10.lib"
               "${TENSORRT_HOME}/lib/nvonnxparser_10.lib")
  set(DEPS_LIBS "${DEPS_PATH}/flatbuffers-build/${CMAKE_BUILD_TYPE}/flatbuffers.lib"
               "${DEPS_PATH}/onnx-build/${CMAKE_BUILD_TYPE}/onnx.lib"
               "${DEPS_PATH}/onnx-build/${CMAKE_BUILD_TYPE}/onnx_proto.lib")
  
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DEPS_LIBS ${DEPS_LIBS}
                 "${DEPS_PATH}/protobuf-build/${CMAKE_BUILD_TYPE}/libprotobufd.lib"
                 "${DEPS_PATH}/protobuf-build/${CMAKE_BUILD_TYPE}/libprotocd.lib")
  else()
    set(DEPS_LIBS ${DEPS_LIBS} 
                 "${DEPS_PATH}/protobuf-build/${CMAKE_BUILD_TYPE}/libprotobuf.lib"
                 "${DEPS_PATH}/protobuf-build/${CMAKE_BUILD_TYPE}/libprotoc.lib")
  endif()
else()
  set(PLATFORM "Linux")
  set(ORT_LIB "${ORT_HOME}/build/${PLATFORM}/${CMAKE_BUILD_TYPE}/libonnxruntime.so")
  set(DEPS_PATH "${ORT_HOME}/build/${PLATFORM}/${CMAKE_BUILD_TYPE}/_deps")
  set(TRT_LIBS "${TENSORRT_HOME}/lib/libnvinfer.so"
               "${TENSORRT_HOME}/lib/libnvinfer_plugin.so"
               "${TENSORRT_HOME}/lib/libnvonnxparser.so")
  set(DEPS_LIBS "${DEPS_PATH}/flatbuffers-build/libflatbuffers.a"
               "${DEPS_PATH}/onnx-build/libonnx.a"
               "${DEPS_PATH}/onnx-build/libonnx_proto.a")
  
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DEPS_LIBS ${DEPS_LIBS}
                 "${DEPS_PATH}/protobuf-build/libprotobufd.a"
                 "${DEPS_PATH}/protobuf-build/libprotocd.a")
  else()
    set(DEPS_LIBS ${DEPS_LIBS}
                 "${DEPS_PATH}/protobuf-build/libprotobuf.a"
                 "${DEPS_PATH}/protobuf-build/libprotoc.a")
  endif()
endif()

MESSAGE(STATUS "Looking for following dependencies ...")
MESSAGE(STATUS "Platform : ${PLATFORM}")
MESSAGE(STATUS "ORT home : ${ORT_HOME}")
MESSAGE(STATUS "ORT lib  : ${ORT_LIB}")
MESSAGE(STATUS "Deps path: ${DEPS_PATH}")
MESSAGE(STATUS "Deps libs: ${DEPS_LIBS}")
MESSAGE(STATUS "TRT libs : ${TRT_LIBS}")

target_include_directories(TensorRTEp PUBLIC "${ORT_HOME}/include/onnxruntime/core/session/"
                                             "./utils"
                                             "/usr/local/cuda/include"
                                             ${TENSORRT_HOME}/include
                                             "${DEPS_PATH}/flatbuffers-src/include"
                                             "${DEPS_PATH}/gsl-src/include"
                                             "${DEPS_PATH}/onnx-src"
                                             "${DEPS_PATH}/onnx-build"
                                             "${DEPS_PATH}/protobuf-src/src"
)

target_link_libraries(TensorRTEp PUBLIC ${ORT_LIB}
                                        ${TRT_LIBS}
                                        CUDA::cudart
                                        ${DEPS_LIBS})
