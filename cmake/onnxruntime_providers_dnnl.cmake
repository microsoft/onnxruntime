# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  file(GLOB_RECURSE onnxruntime_providers_dnnl_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/dnnl/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/dnnl/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_dnnl_cc_srcs})
  onnxruntime_add_shared_library_module(onnxruntime_providers_dnnl ${onnxruntime_providers_dnnl_cc_srcs})
  target_link_directories(onnxruntime_providers_dnnl PRIVATE ${DNNL_LIB_DIR})
  if (MSVC AND onnxruntime_ENABLE_STATIC_ANALYSIS)
    # dnnl_convgrad.cc(47,0): Warning C6262: Function uses '38816' bytes of stack:  exceeds /analyze:stacksize '16384'.  Consider moving some data to heap.
    target_compile_options(onnxruntime_providers_dnnl PRIVATE  "/analyze:stacksize 131072")
  endif()

  add_dependencies(onnxruntime_providers_dnnl onnxruntime_providers_shared project_dnnl ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_include_directories(onnxruntime_providers_dnnl PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${DNNL_INCLUDE_DIR} ${DNNL_OCL_INCLUDE_DIR})
  # ${CMAKE_CURRENT_BINARY_DIR} is so that #include "onnxruntime_config.h" inside tensor_shape.h is found
  target_link_libraries(onnxruntime_providers_dnnl PRIVATE dnnl ${ONNXRUNTIME_PROVIDERS_SHARED} Boost::mp11 ${ABSEIL_LIBS} ${GSL_TARGET} safeint_interface)
  install(FILES ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/dnnl/dnnl_provider_options.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/)
  set_target_properties(onnxruntime_providers_dnnl PROPERTIES FOLDER "ONNXRuntime")
  set_target_properties(onnxruntime_providers_dnnl PROPERTIES LINKER_LANGUAGE CXX)

  # Needed for the provider interface, as it includes training headers when training is enabled
  if (onnxruntime_ENABLE_TRAINING_OPS)
    target_include_directories(onnxruntime_providers_dnnl PRIVATE ${ORTTRAINING_ROOT})
  endif()

  # Needed for threadpool handling
  if(onnxruntime_BUILD_JAVA)
    add_compile_definitions(DNNL_JAVA)
  endif()

  if(APPLE)
    set_property(TARGET onnxruntime_providers_dnnl APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/core/providers/dnnl/exported_symbols.lst")
    set_target_properties(onnxruntime_providers_dnnl PROPERTIES
      INSTALL_RPATH "@loader_path"
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH_USE_LINK_PATH FALSE)
    target_link_libraries(onnxruntime_providers_dnnl PRIVATE nsync::nsync_cpp)
  elseif(UNIX)
    set_property(TARGET onnxruntime_providers_dnnl APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/dnnl/version_script.lds -Xlinker --gc-sections -Xlinker -rpath=\$ORIGIN")
    target_link_libraries(onnxruntime_providers_dnnl PRIVATE nsync::nsync_cpp)
  elseif(WIN32)
    set_property(TARGET onnxruntime_providers_dnnl APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/dnnl/symbols.def")
  else()
    message(FATAL_ERROR "onnxruntime_providers_dnnl unknown platform, need to specify shared library exports for it")
  endif()

  install(TARGETS onnxruntime_providers_dnnl
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})