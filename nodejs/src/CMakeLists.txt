cmake_minimum_required(VERSION 3.11)

project (onnxruntime-node)

set(CMAKE_CXX_STANDARD 14)

add_compile_definitions(NAPI_VERSION=${napi_build_version})
add_compile_definitions(NAPI_EXPERIMENTAL)

include_directories(${CMAKE_JS_INC})
include_directories(${CMAKE_SOURCE_DIR}/../../include/onnxruntime)
# include_directories(${CMAKE_SOURCE_DIR}/../../include/onnxruntime/core/session)
include_directories(${CMAKE_SOURCE_DIR}/node_modules/node-addon-api)

file(GLOB SOURCE_FILES ${CMAKE_SOURCE_DIR}/src/*.cc)

# CPU
add_library(onnxruntime_binding SHARED ${SOURCE_FILES})
set_target_properties(onnxruntime_binding PROPERTIES PREFIX "" SUFFIX ".node")
target_link_libraries(onnxruntime_binding ${CMAKE_JS_LIB})
if (WIN32)
  target_link_libraries(onnxruntime_binding ${CMAKE_SOURCE_DIR}/../../build/Windows/${CMAKE_BUILD_TYPE}/${CMAKE_BUILD_TYPE}/onnxruntime.lib)
endif()

# # GPU
# add_library(onnxruntime_gpu SHARED ${SOURCE_FILES})
# set_target_properties(onnxruntime_gpu PROPERTIES PREFIX "" SUFFIX ".node")
# target_link_libraries(onnxruntime_gpu ${CMAKE_JS_LIB})
# if (WIN32)
#   target_link_libraries(onnxruntime_gpu ${CMAKE_SOURCE_DIR}/onnxruntime/bin/win_gpu-x64/onnxruntime.lib)
# endif()

add_custom_command(
    TARGET onnxruntime_binding POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:onnxruntime_binding> ${CMAKE_SOURCE_DIR}/bin/
)

if (WIN32)
  add_custom_command(
    TARGET onnxruntime_binding POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_SOURCE_DIR}/../../build/Windows/${CMAKE_BUILD_TYPE}/${CMAKE_BUILD_TYPE}/onnxruntime.dll
      ${CMAKE_SOURCE_DIR}/bin/
  )
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_custom_command(
      TARGET onnxruntime_binding POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/../../build/Windows/${CMAKE_BUILD_TYPE}/${CMAKE_BUILD_TYPE}/onnxruntime.pdb
        ${CMAKE_SOURCE_DIR}/bin/
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE_DIR:onnxruntime_binding>/onnxruntime_binding.pdb ${CMAKE_SOURCE_DIR}/bin/
    )
  endif()
endif()
