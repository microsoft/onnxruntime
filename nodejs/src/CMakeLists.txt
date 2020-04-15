cmake_minimum_required(VERSION 3.11)

project (onnxruntime-node)

set(CMAKE_CXX_STANDARD 14)

include_directories(${CMAKE_JS_INC})
include_directories(${CMAKE_SOURCE_DIR}/../../include/onnxruntime)
include_directories(${CMAKE_SOURCE_DIR}/../../include/onnxruntime/core/session)
include_directories(${CMAKE_SOURCE_DIR}/node_modules/node-addon-api)

file(GLOB SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.cc)

add_compile_definitions(NAPI_EXPERIMENTAL)

# CPU
add_library(onnxruntime SHARED ${SOURCE_FILES})
set_target_properties(onnxruntime PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(onnxruntime ${CMAKE_JS_LIB})
if (WIN32)
  target_link_libraries(onnxruntime ${CMAKE_SOURCE_DIR}/../../build/Windows/${CMAKE_BUILD_TYPE}/${CMAKE_BUILD_TYPE}/onnxruntime.lib)
endif()

# # GPU
# add_library(onnxruntime_gpu SHARED ${SOURCE_FILES})
# set_target_properties(onnxruntime_gpu PROPERTIES PREFIX "" SUFFIX ".node")
# target_link_libraries(onnxruntime_gpu ${CMAKE_JS_LIB})
# if (WIN32)
#   target_link_libraries(onnxruntime_gpu ${CMAKE_SOURCE_DIR}/onnxruntime/bin/win_gpu-x64/onnxruntime.lib)
# endif()
