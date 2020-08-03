# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.5)
# Project
project(onnxruntime C CXX)

option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_USE_OPENVINO "Build with OpenVINO support" OFF)
option(onnxruntime_USE_NNAPI_BUILTIN "Build with builtin NNAPI lib for Android NNAPI support" OFF)
option(onnxruntime_USE_DNNL "Build with DNNL support" OFF)
option(onnxruntime_USE_NGRAPH "Build with nGraph support" OFF)
option(onnxruntime_USE_NUPHAR "Build with Nuphar" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(onnxruntime_USE_DML "Build with DirectML support" OFF)
option(onnxruntime_USE_ACL "Build with ACL support" OFF)

#The macros are used in ServerEnvironment::RegisterExecutionProviders function
if(onnxruntime_USE_CUDA)
  add_definitions(-DUSE_CUDA=1)
endif()
if(onnxruntime_USE_OPENVINO)
  add_definitions(-DUSE_OPENVINO=1)
endif()
if(onnxruntime_USE_NNAPI_BUILTIN)
  add_definitions(-DUSE_NNAPI=1)
endif()
if(onnxruntime_USE_DNNL)
  add_definitions(-DUSE_DNNL=1)
endif()
if(onnxruntime_USE_NGRAPH)
  add_definitions(-DUSE_NGRAPH=1)
endif()
if(onnxruntime_USE_TENSORRT)
  add_definitions(-DUSE_TENSORRT=1)
endif()
if(onnxruntime_USE_DML)
  add_definitions(-DUSE_DML=1)
endif()
if(onnxruntime_USE_ACL)
  add_definitions(-DUSE_ACL=1)
endif()

set(CMAKE_CXX_STANDARD 14)

find_package(Threads)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(protobuf_function.cmake)
find_package(gRPC REQUIRED)

set(SERVER_APP_NAME "onnxruntime_server")
set(onnxruntime_USE_FULL_PROTOBUF ON)

set(ONNXRUNTIME_SERVER_ROOT ${PROJECT_SOURCE_DIR})

# Generate .h and .cc files from protobuf file
add_library(server_proto ${ONNXRUNTIME_SERVER_ROOT}/protobuf/predict.proto ${ONNXRUNTIME_SERVER_ROOT}/protobuf/onnx-ml.proto)
if(WIN32)
  target_compile_options(server_proto PRIVATE "/wd4125" "/wd4456")
endif()
target_include_directories(server_proto PUBLIC $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_INCLUDE_DIRECTORIES> "${CMAKE_CURRENT_BINARY_DIR}/.." ${CMAKE_CURRENT_BINARY_DIR}/onnx)
target_compile_definitions(server_proto PUBLIC $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_COMPILE_DEFINITIONS>)
onnxruntime_protobuf_generate(APPEND_PATH ${ONNXRUNTIME_SERVER_ROOT}/protobuf ${ONNXRUNTIME_ROOT}/core/protobuf TARGET server_proto)
if(NOT WIN32)
  if(HAS_UNUSED_PARAMETER)
     set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/predict.pb.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
     set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/onnx-ml.pb.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
  endif()
endif()

# Setup dependencies
include(get_boost.cmake)
set(SPDLOG_BUILD_EXAMPLES OFF)
add_subdirectory(external/spdlog)

# Generate GRPC service source and headers.
get_filename_component(grpc_proto "${ONNXRUNTIME_SERVER_ROOT}/protobuf/prediction_service.proto" ABSOLUTE)
get_filename_component(grpc_proto_path "${grpc_proto}" PATH)

set(grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/prediction_service.grpc.pb.cc")
set(grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/prediction_service.grpc.pb.h")

if(NOT _gRPC_PYTHON_PLUGIN_EXECUTABLE)
  find_program(_gRPC_PYTHON_PLUGIN_EXECUTABLE grpc_python_plugin DOC "The gRPC Python plugin for protoc")
endif()

add_custom_command(
      OUTPUT "${grpc_srcs}" "${grpc_hdrs}"
      COMMAND protoc
      ARGS
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --plugin=protoc-gen-grpc="${_gRPC_CPP_PLUGIN_EXECUTABLE}"
        -I ${grpc_proto_path}
        "${grpc_proto}"
      DEPENDS "${grpc_proto}"
      COMMENT "Running ${_gRPC_CPP_PLUGIN_EXECUTABLE} on ${grpc_proto}"
    )

add_library(server_grpc_proto ${grpc_srcs})
target_include_directories(server_grpc_proto PUBLIC $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_INCLUDE_DIRECTORIES> "${CMAKE_CURRENT_BINARY_DIR}" ${CMAKE_CURRENT_BINARY_DIR}/onnx PRIVATE)
set(grpc_reflection -Wl,--whole-archive grpc++_reflection -Wl,--no-whole-archive)
set(grpc_static_libs gRPC::grpc++ grpcpp_channelz)

add_dependencies(server_grpc_proto server_proto Boost)
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
  "${ONNXRUNTIME_SERVER_ROOT}/http/json_handling.cc"
  "${ONNXRUNTIME_SERVER_ROOT}/http/predict_request_handler.cc"
  "${ONNXRUNTIME_SERVER_ROOT}/http/util.cc"
  "${ONNXRUNTIME_SERVER_ROOT}/environment.cc"
  "${ONNXRUNTIME_SERVER_ROOT}/executor.cc"
  "${ONNXRUNTIME_SERVER_ROOT}/converter.cc"
  "${ONNXRUNTIME_SERVER_ROOT}/util.cc"
  "${ONNXRUNTIME_SERVER_ROOT}/core/request_id.cc"
  "${ONNXRUNTIME_SERVER_ROOT}/grpc/prediction_service_impl.cc"
  "${ONNXRUNTIME_SERVER_ROOT}/grpc/grpc_app.cc"
  "${ONNXRUNTIME_SERVER_ROOT}/serializing/tensorprotoutils.cc"
  )
if(NOT WIN32)
  if(HAS_UNUSED_PARAMETER)
    set_source_files_properties(${onnxruntime_server_lib_srcs} PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
  endif()
endif()

file(GLOB_RECURSE onnxruntime_server_http_core_lib_srcs
  "${ONNXRUNTIME_SERVER_ROOT}/http/core/*.cc"
  )

file(GLOB_RECURSE onnxruntime_server_srcs
  "${ONNXRUNTIME_SERVER_ROOT}/main.cc"
)




# HTTP core library
add_library(onnxruntime_server_http_core_lib STATIC
  ${onnxruntime_server_http_core_lib_srcs})
target_include_directories(onnxruntime_server_http_core_lib
  PUBLIC
  ${ONNXRUNTIME_ROOT}
  ${ONNXRUNTIME_SERVER_ROOT}/http/core
  ${ONNXRUNTIME_SERVER_ROOT}/core
  ${Boost_INCLUDE_DIR}

)
add_dependencies(onnxruntime_server_http_core_lib Boost)

# Server library
add_library(onnxruntime_server_lib ${onnxruntime_server_lib_srcs})
target_include_directories(onnxruntime_server_lib PRIVATE
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${ONNXRUNTIME_ROOT}
  ${ONNXRUNTIME_SERVER_ROOT}
  ${ONNXRUNTIME_SERVER_ROOT}/http
  ${ONNXRUNTIME_SERVER_ROOT}/logging
  ${ONNXRUNTIME_SERVER_ROOT}/core
  PUBLIC
  ${ONNXRUNTIME_SERVER_ROOT}/external/spdlog/include
  ${ONNXRUNTIME_SERVER_ROOT}/http/core
  ${ONNXRUNTIME_SERVER_ROOT}
  ${Boost_INCLUDE_DIR}
)


if (onnxruntime_USE_SYSLOG)
  target_compile_definitions(onnxruntime_server_lib PUBLIC USE_SYSLOG="1")
endif()
add_dependencies(onnxruntime_server_lib server_proto Boost)

# Server Application
add_executable(${SERVER_APP_NAME} ${onnxruntime_server_srcs})
add_dependencies(${SERVER_APP_NAME} server_proto Boost)

if(NOT WIN32)
  if(HAS_UNUSED_PARAMETER)
    set_source_files_properties("${ONNXRUNTIME_SERVER_ROOT}/main.cc" PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
  endif()
endif()

set(onnxruntime_SERVER_VERSION "local-build" CACHE STRING "Sever version")
target_compile_definitions(${SERVER_APP_NAME} PUBLIC SRV_VERSION="${onnxruntime_SERVER_VERSION}")
message(STATUS "ONNX Runtime Server version set to: ${onnxruntime_SERVER_VERSION}")

set(onnxruntime_LATEST_COMMIT_ID "default" CACHE STRING "The latest commit id")
target_compile_definitions(${SERVER_APP_NAME} PUBLIC LATEST_COMMIT_ID="${onnxruntime_LATEST_COMMIT_ID}")
message(STATUS "ONNX Runtime Server latest commit id is: ${onnxruntime_LATEST_COMMIT_ID}")

target_include_directories(${SERVER_APP_NAME} PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${ONNXRUNTIME_SERVER_ROOT}/http
)


target_link_libraries(${SERVER_APP_NAME} PRIVATE
    onnxruntime_server_http_core_lib
    onnxruntime_server_lib
    ${grpc_reflection} #Note that this will break the tests if we try to link it to the lib so just link to the executable.
   server_grpc_proto server_proto
  ${Boost_LIBRARIES}
  onnxruntime_server_http_core_lib
  ${grpc_static_libs}  protobuf::libprotobuf
  spdlog
  onnxruntime gtest re2  ${CMAKE_DL_LIBS} Threads::Threads
)




file(GLOB onnxruntime_test_server_src
    "test/unit_tests/*.cc"
    "test/unit_tests/*.h"
  )

  file(GLOB onnxruntime_integration_test_server_src
    "test/integration_tests/*.py"
  )
  if(NOT WIN32)
    if(HAS_UNUSED_PARAMETER)
      set_source_files_properties("test/unit_tests/json_handling_tests.cc" PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
      set_source_files_properties("test/unit_tests/converter_tests.cc" PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
      set_source_files_properties("test/unit_tests/util_tests.cc" PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
      set_source_files_properties("test/unit_tests/prediction_service_impl_test.cc" PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
      set_source_files_properties("test/unit_tests/executor_test.cc" PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
    endif()
  endif()

  add_executable(onnxruntime_server_tests ${onnxruntime_test_server_src})
  add_dependencies(onnxruntime_server_tests server_proto Boost)
  target_link_libraries(onnxruntime_server_tests PRIVATE onnxruntime_server_http_core_lib
    onnxruntime_server_lib
    ${grpc_reflection} #Note that this will break the tests if we try to link it to the lib so just link to the executable.
   server_grpc_proto server_proto
  ${Boost_LIBRARIES}
  onnxruntime_server_http_core_lib
  ${grpc_static_libs} protobuf::libprotobuf
  spdlog
  onnxruntime gtest re2 gtest ${CMAKE_DL_LIBS} Threads::Threads)
  target_include_directories(onnxruntime_server_tests PRIVATE ${ONNXRUNTIME_SERVER_ROOT}/external/spdlog/include
  ${ONNXRUNTIME_SERVER_ROOT}/http/core
  ${ONNXRUNTIME_SERVER_ROOT} ${ONNXRUNTIME_SERVER_ROOT}/core)
  onnxruntime_protobuf_generate(
          APPEND_PATH IMPORT_DIRS ${ONNXRUNTIME_SERVER_ROOT}/protobuf
          PROTOS ${ONNXRUNTIME_SERVER_ROOT}/protobuf/predict.proto protobuf/onnx-ml.proto
          LANGUAGE python
          TARGET onnxruntime_server_tests
          OUT_VAR server_test_py)

  set(grpc_py "${CMAKE_CURRENT_BINARY_DIR}/prediction_service_pb2_grpc.py")

  add_custom_command(
    TARGET onnxruntime_server_tests
    COMMAND protoc
    ARGS
      --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
      --plugin=protoc-gen-grpc=${_gRPC_PYTHON_PLUGIN_EXECUTABLE}
      -I ${grpc_proto_path}
      "${grpc_proto}"
    DEPENDS "${grpc_proto}"
    COMMENT "Running ${_gRPC_PYTHON_PLUGIN_EXECUTABLE} on ${grpc_proto}"
    )

  add_custom_command(
    TARGET onnxruntime_server_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/server_test
    COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_integration_test_server_src}
      ${CMAKE_CURRENT_BINARY_DIR}/server_test/
      COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_CURRENT_BINARY_DIR}/onnx_ml_pb2.py
      ${CMAKE_CURRENT_BINARY_DIR}/server_test/
    COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_CURRENT_BINARY_DIR}/predict_pb2.py
      ${CMAKE_CURRENT_BINARY_DIR}/server_test/
    COMMAND ${CMAKE_COMMAND} -E copy
      ${grpc_py}
      ${CMAKE_CURRENT_BINARY_DIR}/server_test/)

    add_test(NAME onnxruntime_server_tests
    COMMAND onnxruntime_server_tests
    WORKING_DIRECTORY ${ONNXRUNTIME_SERVER_ROOT}/test/testdata>
    )
