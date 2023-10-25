# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  add_compile_definitions(USE_INTREE=1)

  file(GLOB_RECURSE onnxruntime_providers_intree_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_INCLUDE_DIR}/core/providers/intree/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/intree/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/intree/*.cc"
  )

  source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_providers_intree_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_intree ${onnxruntime_providers_intree_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_intree
    onnxruntime_common onnxruntime_framework onnx pthreadpool Boost::mp11 safeint_interface
  )

  add_dependencies(onnxruntime_providers_intree onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_intree PROPERTIES FOLDER "ONNXRuntime")

  set_target_properties(onnxruntime_providers_intree PROPERTIES LINKER_LANGUAGE CXX)
  #target_include_directories(onnxruntime_providers_intree PUBLIC "/bert_ort/leca/code/onnxruntime2/include/onnxruntime")

  if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_providers_intree
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
