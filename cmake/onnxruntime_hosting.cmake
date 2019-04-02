# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

project(onnxruntime_hosting)

# Generate .h and .cc files from protobuf file
add_library(hosting_proto
  ${ONNXRUNTIME_ROOT}/hosting/protobuf/predict.proto
  ${ONNXRUNTIME_ROOT}/hosting/protobuf/model_metadata.proto
  ${ONNXRUNTIME_ROOT}/hosting/protobuf/model_status.proto
  ${ONNXRUNTIME_ROOT}/hosting/protobuf/error_code.proto)
target_include_directories(hosting_proto PUBLIC $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_INCLUDE_DIRECTORIES> "${CMAKE_CURRENT_BINARY_DIR}/.." ${CMAKE_CURRENT_BINARY_DIR}/onnx)
target_compile_definitions(hosting_proto PUBLIC $<TARGET_PROPERTY:protobuf::libprotobuf,INTERFACE_COMPILE_DEFINITIONS>)
onnxruntime_protobuf_generate(APPEND_PATH IMPORT_DIRS ${REPO_ROOT}/cmake/external/protobuf/src ${ONNXRUNTIME_ROOT}/hosting/protobuf ${ONNXRUNTIME_ROOT}/core/protobuf TARGET hosting_proto)
add_dependencies(hosting_proto onnx_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})
if(NOT WIN32)
  if(HAS_UNUSED_PARAMETER)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/error_code.pb.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/model_metadata.pb.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/model_status.pb.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/predict.pb.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
  endif()
endif()

# Setup dependencies
find_package(Boost 1.69 COMPONENTS system context thread program_options REQUIRED)
set(re2_src ${REPO_ROOT}/cmake/external/re2)

# Setup source code
file(GLOB_RECURSE onnxruntime_hosting_lib_srcs
  "${ONNXRUNTIME_ROOT}/hosting/http/*.cc"
  "${ONNXRUNTIME_ROOT}/hosting/environment.cc"
  "${ONNXRUNTIME_ROOT}/hosting/executor.cc"
  "${ONNXRUNTIME_ROOT}/hosting/converter.cc"
  "${ONNXRUNTIME_ROOT}/hosting/util.cc"
  )
if(NOT WIN32)
  if(HAS_UNUSED_PARAMETER)
    set_source_files_properties(${ONNXRUNTIME_ROOT}/hosting/http/json_handling.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
    set_source_files_properties(${ONNXRUNTIME_ROOT}/hosting/http/predict_request_handler.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
    set_source_files_properties(${ONNXRUNTIME_ROOT}/hosting/executor.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
    set_source_files_properties(${ONNXRUNTIME_ROOT}/hosting/converter.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
    set_source_files_properties(${ONNXRUNTIME_ROOT}/hosting/util.cc PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
  endif()
endif()

file(GLOB_RECURSE onnxruntime_hosting_srcs
  "${ONNXRUNTIME_ROOT}/hosting/main.cc"
)

# Hosting library
add_library(onnxruntime_hosting_lib ${onnxruntime_hosting_lib_srcs})
onnxruntime_add_include_to_target(onnxruntime_hosting_lib gsl onnx_proto hosting_proto)
target_include_directories(onnxruntime_hosting_lib PRIVATE
  ${ONNXRUNTIME_ROOT}
  ${CMAKE_CURRENT_BINARY_DIR}/onnx
  ${ONNXRUNTIME_ROOT}/hosting
  ${ONNXRUNTIME_ROOT}/hosting/http
  PUBLIC
  ${Boost_INCLUDE_DIR}
  ${re2_src}
)

target_link_libraries(onnxruntime_hosting_lib PRIVATE
  hosting_proto
  ${Boost_LIBRARIES}
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
source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_hosting_srcs} ${onnxruntime_hosting_lib_srcs} ${onnxruntime_hosting_lib})

# Hosting Application
add_executable(${PROJECT_NAME} ${onnxruntime_hosting_srcs})
add_dependencies(${PROJECT_NAME} onnx hosting_proto onnx_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})

if(NOT WIN32)
  if(HAS_UNUSED_PARAMETER)
    set_source_files_properties("${ONNXRUNTIME_ROOT}/hosting/main.cc" PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
  endif()
endif()

onnxruntime_add_include_to_target(${PROJECT_NAME} onnxruntime_session onnxruntime_hosting_lib gsl onnx onnx_proto hosting_proto)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${ONNXRUNTIME_ROOT}
    ${ONNXRUNTIME_ROOT}/hosting/http
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    onnxruntime_hosting_lib
)

