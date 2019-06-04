# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(SERVER_APP_NAME "onnxruntime_server")

set(gRPC_BUILD_TESTS OFF)
set(gRPC_GFLAGS_PROVIDER "")
set(gRPC_BENCHMARK_PROVIDER "")
set(gRPC_ZLIB_PROVIDER "package")
set(gRPC_PROTOBUF_PROVIDER "")

if(HAS_UNUSED_PARAMETER)
  string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-parameter")
  string(APPEND CMAKE_C_FLAGS " -Wno-unused-parameter")
endif()
# protobuf targets have already been included as submodules - adapted from https://github.com/grpc/grpc/blob/master/cmake/protobuf.cmake
set(_gRPC_PROTOBUF_LIBRARY_NAME "libprotobuf")
set(_gRPC_PROTOBUF_LIBRARIES protobuf::${_gRPC_PROTOBUF_LIBRARY_NAME})

set(_gRPC_PROTOBUF_PROTOC_LIBRARIES protobuf::libprotoc)
# extract the include dir from target's properties

set(_gRPC_PROTOBUF_WELLKNOWN_INCLUDE_DIR ${REPO_ROOT}/cmake/external/protobuf/src)
set(_gRPC_PROTOBUF_PROTOC protobuf::protoc)
set(_gRPC_PROTOBUF_PROTOC_EXECUTABLE $<TARGET_FILE:protobuf::protoc>)

set(_gRPC_PROTOBUF_INCLUDE_DIR ${PROTOBUF_INCLUDE_DIRS})

MESSAGE(STATUS ${_gRPC_PROTOBUF_WELLKNOWN_INCLUDE_DIR})

add_subdirectory(${PROJECT_SOURCE_DIR}/external/grpc EXCLUDE_FROM_ALL)

if(HAS_UNUSED_PARAMETER)
  string(APPEND CMAKE_CXX_FLAGS " -Wunused-parameter")
  string(APPEND CMAKE_C_FLAGS " -Wunused-parameter")
endif()

set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
set(_GRPC_PY_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_python_plugin>)


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

# Genrate GPRC stuff
get_filename_component(grpc_proto "${ONNXRUNTIME_ROOT}/server/protobuf/prediction_service.proto" ABSOLUTE)
get_filename_component(grpc_proto_path "${grpc_proto}" PATH)

# Generated sources

set(grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/prediction_service.grpc.pb.cc")
set(grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/prediction_service.grpc.pb.h")
add_custom_command(
      OUTPUT "${grpc_srcs}" "${grpc_hdrs}"
      COMMAND $<TARGET_FILE:protobuf::protoc>
      ARGS 
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        -I ${grpc_proto_path}
        "${grpc_proto}"
      DEPENDS "${grpc_proto}" ${_GRPC_CPP_PLUGIN_EXECUTABLE}
      COMMENT "Running ${_GRPC_CPP_PLUGIN_EXECUTABLE} on ${grpc_proto}"
    )

add_library(server_grpc ${grpc_srcs})
target_include_directories(server_grpc PUBLIC $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_INCLUDE_DIRECTORIES> "${CMAKE_CURRENT_BINARY_DIR}" ${CMAKE_CURRENT_BINARY_DIR}/onnx)
set(grpc_reflection -Wl,--whole-archive grpc++_reflection -Wl,--no-whole-archive)
set(grpc_static_libs grpc++ grpcpp_channelz) #TODO: stuff for Windows.
target_link_libraries(server_grpc ${grpc_static_libs})
add_dependencies(server_grpc server_proto)
# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

if(NOT WIN32)
  if(HAS_UNUSED_PARAMETER)
  set_source_files_properties(${grpc_srcs} PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
  set_source_files_properties(${onnxruntime_server_grpc_srcs} PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
  endif()
endif()


# Setup source code
set(onnxruntime_server_lib_srcs
  "${ONNXRUNTIME_ROOT}/server/http/json_handling.cc"
  "${ONNXRUNTIME_ROOT}/server/http/predict_request_handler.cc"
  "${ONNXRUNTIME_ROOT}/server/http/util.cc"
  "${ONNXRUNTIME_ROOT}/server/environment.cc"
  "${ONNXRUNTIME_ROOT}/server/executor.cc"
  "${ONNXRUNTIME_ROOT}/server/converter.cc"
  "${ONNXRUNTIME_ROOT}/server/util.cc"
  "${ONNXRUNTIME_ROOT}/server/grpc/prediction_service_impl.cc"
  "${ONNXRUNTIME_ROOT}/server/grpc/grpc_app.cc"
  )
if(NOT WIN32)
  if(HAS_UNUSED_PARAMETER)
    set_source_files_properties(${onnxruntime_server_lib_srcs} PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
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
  server_grpc
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

set(onnxruntime_SERVER_VERSION "local-build" CACHE STRING "Sever version")
target_compile_definitions(${SERVER_APP_NAME} PUBLIC SRV_VERSION="${onnxruntime_SERVER_VERSION}")
message(STATUS "ONNX Runtime Server version set to: ${onnxruntime_SERVER_VERSION}")

set(onnxruntime_LATEST_COMMIT_ID "default" CACHE STRING "The latest commit id")
target_compile_definitions(${SERVER_APP_NAME} PUBLIC LATEST_COMMIT_ID="${onnxruntime_LATEST_COMMIT_ID}")
message(STATUS "ONNX Runtime Server latest commit id is: ${onnxruntime_LATEST_COMMIT_ID}")

onnxruntime_add_include_to_target(${SERVER_APP_NAME} onnxruntime_session onnxruntime_server_lib gsl onnx onnx_proto server_proto)

target_include_directories(${SERVER_APP_NAME} PRIVATE
    ${ONNXRUNTIME_ROOT}
    ${ONNXRUNTIME_ROOT}/server/http
)


target_link_libraries(${SERVER_APP_NAME} PRIVATE
    onnxruntime_server_http_core_lib
    onnxruntime_server_lib
    ${grpc_reflection} #Note that this will break the tests if we try to link it to the normal lib so just link to server.
)

