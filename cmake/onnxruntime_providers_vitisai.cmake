# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  if ("${GIT_COMMIT_ID}" STREQUAL "")
  execute_process(
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND git rev-parse HEAD
    OUTPUT_VARIABLE GIT_COMMIT_ID
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  configure_file(${ONNXRUNTIME_ROOT}/core/providers/vitisai/imp/version_info.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/VitisAI/version_info.h)
  file(GLOB onnxruntime_providers_vitisai_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/vitisai/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/vitisai/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/vitisai/imp/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/vitisai/imp/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_vitisai_cc_srcs})
  onnxruntime_add_shared_library(onnxruntime_providers_vitisai ${onnxruntime_providers_vitisai_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_vitisai ${ONNXRUNTIME_PROVIDERS_SHARED} nlohmann_json::nlohmann_json safeint_interface flatbuffers::flatbuffers)
  target_link_libraries(onnxruntime_providers_vitisai PRIVATE ${ONNXRUNTIME_PROVIDERS_SHARED})
  if(MSVC)
    onnxruntime_add_include_to_target(onnxruntime_providers_vitisai dbghelp)
    set_property(TARGET onnxruntime_providers_vitisai APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/vitisai/symbols.def")
  else(MSVC)
    set_property(TARGET onnxruntime_providers_vitisai APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/vitisai/version_script.lds -Xlinker --gc-sections")
  endif(MSVC)

  target_include_directories(onnxruntime_providers_vitisai PRIVATE "${ONNXRUNTIME_ROOT}/core/providers/vitisai/include" ${XRT_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR}/VitisAI)
  if(MSVC)
    target_compile_options(onnxruntime_providers_vitisai PRIVATE "/Zc:__cplusplus")
    # for dll interface warning.
    target_compile_options(onnxruntime_providers_vitisai PRIVATE "/wd4251")
    # for unused formal parameter
    target_compile_options(onnxruntime_providers_vitisai PRIVATE "/wd4100")
    # for type name first seen using 'class' now seen using 'struct'
    target_compile_options(onnxruntime_providers_vitisai PRIVATE "/wd4099")
  else(MSVC)
    target_compile_options(onnxruntime_providers_vitisai PUBLIC $<$<CONFIG:DEBUG>:-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0>)
    target_compile_options(onnxruntime_providers_vitisai PRIVATE -Wno-unused-parameter)
  endif(MSVC)

  set_target_properties(onnxruntime_providers_vitisai PROPERTIES FOLDER "ONNXRuntime")
  set_target_properties(onnxruntime_providers_vitisai PROPERTIES LINKER_LANGUAGE CXX)

  install(TARGETS onnxruntime_providers_vitisai
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
