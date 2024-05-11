
  file(GLOB_RECURSE
    onnxruntime_providers_npu_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/npu/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/npu/*.cc"
  )
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_npu_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_npu ${onnxruntime_providers_npu_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_npu onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers Boost::mp11 safeint_interface)
  add_dependencies(onnxruntime_providers_npu onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_npu PROPERTIES FOLDER "ONNXRuntime")
  #target_include_directories(onnxruntime_providers_npu PRIVATE
  #  ${ONNXRUNTIME_ROOT} ${npu_INCLUDE_DIRS} ${RKNPU_DDK_INCLUDE_DIR}
  #)
  #link_directories(onnxruntime_providers_npu ${RKNPU_DDK_LIB_DIR})
  set_target_properties(onnxruntime_providers_npu PROPERTIES LINKER_LANGUAGE CXX)

  if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_providers_npu
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()