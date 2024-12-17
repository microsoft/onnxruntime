# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  add_compile_definitions(USE_QNN=1)

  file(GLOB_RECURSE
    onnxruntime_providers_qnn_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/qnn/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/qnn/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/qnn/builder/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/qnn/builder/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/qnn/builder/qnn_node_group/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/qnn/builder/qnn_node_group/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/qnn/builder/opbuilder/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/qnn/builder/opbuilder/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_qnn_cc_srcs})
  onnxruntime_add_shared_library_module(onnxruntime_providers_qnn ${onnxruntime_providers_qnn_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_qnn ${ONNXRUNTIME_PROVIDERS_SHARED} ${GSL_TARGET} onnx onnxruntime_common Boost::mp11 safeint_interface)
  target_link_libraries(onnxruntime_providers_qnn PRIVATE ${ONNXRUNTIME_PROVIDERS_SHARED} ${ABSEIL_LIBS} ${CMAKE_DL_LIBS})
  add_dependencies(onnxruntime_providers_qnn onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_include_directories(onnxruntime_providers_qnn PRIVATE ${ONNXRUNTIME_ROOT}
	                                                       ${CMAKE_CURRENT_BINARY_DIR}
	                                                       ${onnxruntime_QNN_HOME}/include/QNN
							       ${onnxruntime_QNN_HOME}/include)

  # Set linker flags for function(s) exported by EP DLL
  if(UNIX)
    set_property(TARGET onnxruntime_providers_qnn APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/qnn/version_script.lds -Xlinker --gc-sections -Xlinker -rpath=\$ORIGIN")
  elseif(WIN32)
    set_property(TARGET onnxruntime_providers_qnn APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/qnn/symbols.def")
  else()
    message(FATAL_ERROR "onnxruntime_providers_qnn unknown platform, need to specify shared library exports for it")
  endif()

  # Set compile options
  if(MSVC)
    target_compile_options(onnxruntime_providers_qnn PUBLIC /wd4099 /wd4005)
  else()
    # ignore the warning unknown-pragmas on "pragma region"
    target_compile_options(onnxruntime_providers_qnn PRIVATE "-Wno-unknown-pragmas")
  endif()

  set_target_properties(onnxruntime_providers_qnn PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_qnn PROPERTIES CXX_STANDARD_REQUIRED ON)
  set_target_properties(onnxruntime_providers_qnn PROPERTIES FOLDER "ONNXRuntime")

  install(TARGETS onnxruntime_providers_qnn
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
