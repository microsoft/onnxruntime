# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  add_compile_definitions(USE_XNNPACK=1)

  file(GLOB_RECURSE onnxruntime_providers_xnnpack_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_INCLUDE_DIR}/core/providers/xnnpack/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/xnnpack/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/xnnpack/*.cc"
  )

  source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_providers_xnnpack_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_xnnpack ${onnxruntime_providers_xnnpack_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_xnnpack
    onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} XNNPACK pthreadpool
    flatbuffers::flatbuffers Boost::mp11 safeint_interface
  )

  # TODO fix stringop-overflow warnings
  # Add compile option to suppress stringop-overflow error in Flatbuffers.
  if (HAS_STRINGOP_OVERFLOW)
    target_compile_options(onnxruntime_providers_xnnpack PRIVATE -Wno-error=stringop-overflow)
  endif()

  add_dependencies(onnxruntime_providers_xnnpack onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_xnnpack PROPERTIES FOLDER "ONNXRuntime")

  set_target_properties(onnxruntime_providers_xnnpack PROPERTIES LINKER_LANGUAGE CXX)

  if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_providers_xnnpack
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()

  # TODO fix shorten-64-to-32 warnings
  # there are some in builds where sizeof(size_t) != sizeof(int64_t), e.g., in 'ONNX Runtime Web CI Pipeline'
  if (HAS_SHORTEN_64_TO_32 AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    target_compile_options(onnxruntime_providers_xnnpack PRIVATE -Wno-error=shorten-64-to-32)
  endif()
