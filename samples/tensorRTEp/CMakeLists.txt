# usage:
# cd build/
# cmake -S ../ -B ./ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DTENSORRT_HOME=/home/lochi/tensorrt/TensorRT-10.3.0.26 (see the result of "nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits")
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
file(GLOB tensorrt_src "./*.cc" "../utils/*.cc" "./cuda/unary_elementwise_ops_impl.cu")
add_library(TensorRTEp SHARED ${tensorrt_src})

if (NOT TENSORRT_HOME)
  message(FATAL_ERROR "Please specify tensorrt home as -DTENSORRT_HOME=/path/to/trt/")
endif()

# Use release mode if not specified
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if (WIN32)
  set(PLATFORM "Windows")
  set(ORT_REPO "C:/Users/lochi/repos/ort_ep_plugin")
  set(ORT_LIB "${ORT_REPO}/build/${PLATFORM}/${CMAKE_BUILD_TYPE}/${CMAKE_BUILD_TYPE}/onnxruntime.lib")
  set(DEPS_PATH "${ORT_REPO}/build/${PLATFORM}/${CMAKE_BUILD_TYPE}/_deps")
  set(TRT_LIBS "${TENSORRT_HOME}/lib/nvinfer_10.lib" 
               "${TENSORRT_HOME}/lib/nvinfer_plugin_10.lib"
               "${TENSORRT_HOME}/lib/nvonnxparser_10.lib")
  set(DEPS_LIB "${DEPS_PATH}/flatbuffers-build/libflatbuffers.lib"
               "${DEPS_PATH}/onnx-build/libonnx.lib"
               "${DEPS_PATH}/onnx-build/libonnx_proto.lib")
  
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DEPS_LIB ${DEPS_LIB}
                 "${DEPS_PATH}/protobuf-build/libprotobufd.lib"
                 "${DEPS_PATH}/protobuf-build/libprotocd.lib")
  else()
    set(DEPS_LIB ${DEPS_LIB} 
                 "${DEPS_PATH}/protobuf-build/libprotobuf.lib"
                 "${DEPS_PATH}/protobuf-build/libprotoc.lib")
  endif()
  
else()
  set(PLATFORM "Linux")
  set(ORT_REPO "/home/lochi/repos/ort_for_docker_ep_plugin_2")
  set(ORT_LIB "${ORT_REPO}/build/${PLATFORM}/${CMAKE_BUILD_TYPE}/libonnxruntime.so")
  set(DEPS_PATH "${ORT_REPO}/build/${PLATFORM}/${CMAKE_BUILD_TYPE}/_deps")
  set(TRT_LIBS "${TENSORRT_HOME}/lib/libnvinfer.so"
               "${TENSORRT_HOME}/lib/libnvinfer_plugin.so"
               "${TENSORRT_HOME}/lib/libnvonnxparser.so")
  set(DEPS_LIB "${DEPS_PATH}/flatbuffers-build/libflatbuffers.a"
               "${DEPS_PATH}/onnx-build/libonnx.a"
               "${DEPS_PATH}/onnx-build/libonnx_proto.a")
  
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DEPS_LIB ${DEPS_LIB}
                 "${DEPS_PATH}/protobuf-build/libprotobufd.a"
                 "${DEPS_PATH}/protobuf-build/libprotocd.a")
  else()
    set(DEPS_LIB ${DEPS_LIB}
                 "${DEPS_PATH}/protobuf-build/libprotobuf.a"
                 "${DEPS_PATH}/protobuf-build/libprotoc.a")
  endif()
endif()

MESSAGE(STATUS "Looking for following libs or dependencies ...")
MESSAGE(STATUS "Platform : ${PLATFORM}")
MESSAGE(STATUS "ORT repo : ${ORT_REPO}")
MESSAGE(STATUS "ORT lib  : ${ORT_LIB}")
MESSAGE(STATUS "Deps path: ${DEPS_PATH}")
MESSAGE(STATUS "TRT libs : ${TRT_LIBS}")
MESSAGE(STATUS "Deps libs: ${DEPS_LIB}")

target_include_directories(TensorRTEp PUBLIC "../../include/onnxruntime"
                                             "../utils"
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
                                        ${DEPS_LIB})
