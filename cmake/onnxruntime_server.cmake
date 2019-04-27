# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(SERVER_APP_NAME "onnxruntime_server")

# Generate .h and .cc files from protobuf file
add_library(server_proto ${ONNXRUNTIME_ROOT}/server/protobuf/predict.proto)
if(WIN32)
  target_compile_options(server_proto PRIVATE "/wd4125" "/wd4456")
endif()
target_include_directories(server_proto PUBLIC $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_INCLUDE_DIRECTORIES> "${CMAKE_CURRENT_BINARY_DIR}/.." ${CMAKE_CURRENT_BINARY_DIR}/onnx)
target_compile_definitions(server_proto PUBLIC $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_COMPILE_DEFINITIONS>)
onnxruntime_protobuf_generate(APPEND_PATH IMPORT_DIRS ${REPO_ROOT}/cmake/external/protobuf/src ${ONNXRUNTIME_ROOT}/server/protobuf ${ONNXRUNTIME_ROOT}/core/protobuf TARGET server_proto)
add_dependencies(server_proto onnx_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})
if(NOT WIN32)
  if(HAS_UNUSED_PARAMETER)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/model_metadata.pb.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/model_status.pb.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/predict.pb.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
  endif()
endif()

# Setup dependencies
include(get_boost.cmake)
set(re2_src ${REPO_ROOT}/cmake/external/re2)

# Setup source code
set(onnxruntime_server_lib_srcs
  "${ONNXRUNTIME_ROOT}/server/http/json_handling.cc"
  "${ONNXRUNTIME_ROOT}/server/http/predict_request_handler.cc"
  "${ONNXRUNTIME_ROOT}/server/http/util.cc"
  "${ONNXRUNTIME_ROOT}/server/environment.cc"
  "${ONNXRUNTIME_ROOT}/server/executor.cc"
  "${ONNXRUNTIME_ROOT}/server/converter.cc"
  "${ONNXRUNTIME_ROOT}/server/util.cc"
  )
if(NOT WIN32)
  if(HAS_UNUSED_PARAMETER)
    set_source_files_properties(${ONNXRUNTIME_ROOT}/server/http/json_handling.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
    set_source_files_properties(${ONNXRUNTIME_ROOT}/server/http/predict_request_handler.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
    set_source_files_properties(${ONNXRUNTIME_ROOT}/server/executor.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
    set_source_files_properties(${ONNXRUNTIME_ROOT}/server/converter.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
    set_source_files_properties(${ONNXRUNTIME_ROOT}/server/util.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
  endif()
endif()

file(GLOB_RECURSE onnxruntime_server_http_core_lib_srcs
  "${ONNXRUNTIME_ROOT}/server/http/core/*.cc"
  )

file(GLOB_RECURSE onnxruntime_server_srcs
  "${ONNXRUNTIME_ROOT}/server/main.cc"
)

# HTTP core library
add_library(onnxruntime_server_http_core_lib STATIC
  ${onnxruntime_server_http_core_lib_srcs})
target_include_directories(onnxruntime_server_http_core_lib
  PUBLIC
  ${ONNXRUNTIME_ROOT}/server/http/core
  ${Boost_INCLUDE_DIR}
  ${re2_src}
)
add_dependencies(onnxruntime_server_http_core_lib Boost)
target_link_libraries(onnxruntime_server_http_core_lib PRIVATE
  ${Boost_LIBRARIES}
)

# Server library
add_library(onnxruntime_server_lib ${onnxruntime_server_lib_srcs})
onnxruntime_add_include_to_target(onnxruntime_server_lib gsl onnx_proto server_proto)
target_include_directories(onnxruntime_server_lib PRIVATE
  ${ONNXRUNTIME_ROOT}
  ${CMAKE_CURRENT_BINARY_DIR}/onnx
  ${ONNXRUNTIME_ROOT}/server
  ${ONNXRUNTIME_ROOT}/server/http
  PUBLIC
  ${Boost_INCLUDE_DIR}
  ${re2_src}
)

target_link_libraries(onnxruntime_server_lib PRIVATE
  server_proto
  ${Boost_LIBRARIES}
  onnxruntime_server_http_core_lib
  onnxruntime_session
  onnxruntime_optimizer
  onnxruntime_providers
  onnxruntime_util
  onnxruntime_framework
  onnxruntime_util
  onnxruntime_graph
  onnxruntime_common
  onnxruntime_mlas
  ${onnxruntime_EXTERNAL_LIBRARIES}
)

# For IDE only
source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_server_srcs} ${onnxruntime_server_lib_srcs} ${onnxruntime_server_lib})

# Server Application
add_executable(${SERVER_APP_NAME} ${onnxruntime_server_srcs})
add_dependencies(${SERVER_APP_NAME} onnx server_proto onnx_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})

if(NOT WIN32)
  if(HAS_UNUSED_PARAMETER)
    set_source_files_properties("${ONNXRUNTIME_ROOT}/server/main.cc" PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
  endif()
endif()

onnxruntime_add_include_to_target(${SERVER_APP_NAME} onnxruntime_session onnxruntime_server_lib gsl onnx onnx_proto server_proto)

target_include_directories(${SERVER_APP_NAME} PRIVATE
    ${ONNXRUNTIME_ROOT}
    ${ONNXRUNTIME_ROOT}/server/http
)

target_link_libraries(${SERVER_APP_NAME} PRIVATE
    onnxruntime_server_http_core_lib
    onnxruntime_server_lib
)

