# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  find_path(VAIP_CMAKE_LIST_TEXT_IN_LOCAL_WORKING_DIR
    NAME "CMakeLists.txt"
    PATHS "${ONNXRUNTIME_ROOT}/../../vaip"
    NO_DEFAULT_PATH)
  if(VAIP_CMAKE_LIST_TEXT_IN_LOCAL_WORKING_DIR)
    message(STATUS "Found local vaip CMakeLists.txt: ${VAIP_CMAKE_LIST_TEXT_IN_LOCAL_WORKING_DIR}")
    FetchContent_Declare(
      vaip
      SOURCE_DIR "${VAIP_CMAKE_LIST_TEXT_IN_LOCAL_WORKING_DIR}"
      OVERRIDE_FIND_PACKAGE
      )
  else()
    message(STATUS "Did not find local vaip CMakeLists.txt, using FetchContent")
    FetchContent_Declare(
      vaip
      GIT_REPOSITORY ${DEP_URL_vaip}
      GIT_TAG ${DEP_SHA1_vaip}
      GIT_SUBMODULES_RECURSE FALSE
      GIT_SHALLOW TRUE
      EXCLUDE_FROM_ALL
    OVERRIDE_FIND_PACKAGE
  )
  endif()

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
    "${ONNXRUNTIME_ROOT}/core/providers/vitisai/include/vaip/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/vitisai/imp/*.cc"
    "${ONNXRUNTIME_ROOT}/core/providers/vitisai/imp/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_vitisai_cc_srcs})
  onnxruntime_add_shared_library(onnxruntime_providers_vitisai ${onnxruntime_providers_vitisai_cc_srcs})
  # VAIP must be made available to after the onnxruntime_providers_vitisai target is created so that VAIP can detect if it is build as a subproject or not.
  find_package(vaip)
  onnxruntime_add_include_to_target(onnxruntime_providers_vitisai ${ONNXRUNTIME_PROVIDERS_SHARED} ${GSL_TARGET} safeint_interface flatbuffers::flatbuffers  Boost::mp11)
  target_link_libraries(onnxruntime_providers_vitisai PRIVATE ${ONNXRUNTIME_PROVIDERS_SHARED} morphizen::morphizen-core-static)
  if(MSVC)
    onnxruntime_add_include_to_target(onnxruntime_providers_vitisai dbghelp)
    target_sources(onnxruntime_providers_vitisai PRIVATE ${ONNXRUNTIME_ROOT}/core/providers/vitisai/imp/onnxruntime_providers_vitisai.def)
  else(MSVC)
    set_property(TARGET onnxruntime_providers_vitisai APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/vitisai/version_script.lds -Xlinker --gc-sections")
  endif(MSVC)

  target_include_directories(onnxruntime_providers_vitisai PRIVATE "${ONNXRUNTIME_ROOT}/core/providers/vitisai/include" ${CMAKE_CURRENT_BINARY_DIR}/VitisAI)
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

  if(MSVC)
    target_link_options(onnxruntime_providers_vitisai PRIVATE
      "$<$<CONFIG:Debug>:/NODEFAULTLIB:libucrtd.lib /DEFAULTLIB:ucrtd.lib>"
      "$<$<CONFIG:Release>:/NODEFAULTLIB:libucrt.lib /DEFAULTLIB:ucrt.lib>"
      )
  endif(MSVC)

  set_target_properties(onnxruntime_providers_vitisai PROPERTIES FOLDER "ONNXRuntime")
  set_target_properties(onnxruntime_providers_vitisai PROPERTIES LINKER_LANGUAGE CXX)

  install(TARGETS onnxruntime_providers_vitisai
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
