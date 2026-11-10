# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(onnxruntime_ENABLE_STATEFUL)
  set(onnxruntime_stateful_srcs
      diffusion/ddim.cc
      diffusion/ddim.h
      framework/stateful_execution_provider.h
      framework/stateful_execution_utils.h
      generate/sampler_provider.cc
      generate/token_generation.cc
      generate/token_generation.h
      model_package/model_package_chain.cc
      model_package/model_package_chain.h
      model_package/session_chain_api.cc
      session/abi_stateful_impl.h
      session/kv_cache_manager.cc
      session/kv_cache_manager.h
      session/session_chain.cc
      session/session_chain.h
      session/session_loop.cc
      session/session_loop.h
      session/session_loop_ddim.cc
      session/stateful_api.cc
      session/stateful_provider.cc)

  onnxruntime_add_static_library(onnxruntime_stateful ${onnxruntime_stateful_srcs})
  onnxruntime_add_include_to_target(onnxruntime_stateful
    onnxruntime_common onnxruntime_framework onnxruntime_lora onnx onnx_proto ${PROTOBUF_LIB}
    flatbuffers::flatbuffers Boost::mp11 safeint_interface nlohmann_json::nlohmann_json Eigen3::Eigen)
  target_include_directories(onnxruntime_stateful PRIVATE
    "${PROJECT_BINARY_DIR}" "${ONNXRUNTIME_ROOT}" "${MODEL_PACKAGE_LIB_DIR}/include")
  target_link_libraries(onnxruntime_stateful PRIVATE onnxruntime_session)
  add_dependencies(onnxruntime_stateful ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_stateful PROPERTIES FOLDER "ONNXRuntime")

  if(NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_stateful EXPORT ${PROJECT_NAME}Targets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()
endif()

# The API tables live in the core; their optional implementation is selected at composition.
onnxruntime_add_object_library(onnxruntime_stateful_bridge bridge/stateful_bridge.cc)
onnxruntime_add_include_to_target(onnxruntime_stateful_bridge
  onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB}
  flatbuffers::flatbuffers Boost::mp11 safeint_interface nlohmann_json::nlohmann_json Eigen3::Eigen)
target_include_directories(onnxruntime_stateful_bridge PRIVATE "${PROJECT_BINARY_DIR}" "${ONNXRUNTIME_ROOT}")
if(onnxruntime_ENABLE_STATEFUL)
  target_compile_definitions(onnxruntime_stateful_bridge PRIVATE ORT_ENABLE_STATEFUL)
  target_link_libraries(onnxruntime_stateful_bridge PRIVATE onnxruntime_stateful)
endif()
add_dependencies(onnxruntime_stateful_bridge ${onnxruntime_EXTERNAL_DEPENDENCIES})
set_target_properties(onnxruntime_stateful_bridge PROPERTIES FOLDER "ONNXRuntime")

if(NOT onnxruntime_BUILD_SHARED_LIB)
  install(TARGETS onnxruntime_stateful_bridge EXPORT ${PROJECT_NAME}Targets
          OBJECTS DESTINATION ${CMAKE_INSTALL_LIBDIR}/onnxruntime/objects)
endif()

set(onnxruntime_STATEFUL_LINK_ITEMS onnxruntime_stateful_bridge)
if(onnxruntime_ENABLE_STATEFUL)
  list(APPEND onnxruntime_STATEFUL_LINK_ITEMS onnxruntime_stateful)
endif()
set(onnxruntime_STATEFUL_LINK_ITEMS "${onnxruntime_STATEFUL_LINK_ITEMS}" PARENT_SCOPE)

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND onnxruntime_BUILD_WEBASSEMBLY_STATIC_LIB)
  onnxruntime_add_static_library(onnxruntime_stateful_bridge_archive
    $<TARGET_OBJECTS:onnxruntime_stateful_bridge>)
  set(onnxruntime_STATEFUL_BUNDLE_TARGETS onnxruntime_stateful_bridge_archive)
  if(onnxruntime_ENABLE_STATEFUL)
    list(APPEND onnxruntime_STATEFUL_BUNDLE_TARGETS onnxruntime_stateful)
  endif()
  set(onnxruntime_STATEFUL_BUNDLE_TARGETS "${onnxruntime_STATEFUL_BUNDLE_TARGETS}" PARENT_SCOPE)
endif()
