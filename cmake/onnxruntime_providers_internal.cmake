# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# internal providers
# this file is meant to be included in the parent onnxruntime_providers.cmake file

if (onnxruntime_USE_BRAINSLICE)
  add_definitions(-DUSE_BRAINSLICE=1)
  if (WIN32)
    include_directories(${onnxruntime_BRAINSLICE_LIB_PATH}/build/native/include)
    set(fpga_core_lib ${onnxruntime_BRAINSLICE_LIB_PATH}/build/native/lib/x64/dynamic/FPGACoreLib.lib)
    #TODO: this is the pre-build v2 bs_client lib, we might need to build it offline for code-gen case
    if (CMAKE_BUILD_TYPE MATCHES Debug)
      set(bs_client_lib ${onnxruntime_BS_CLIENT_PACKAGE}/x64/Debug/client.lib)
    else(CMAKE_BUILD_TYPE MATCHES Release OR CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
      set(bs_client_lib ${onnxruntime_BS_CLIENT_PACKAGE}/x64/Release/client.lib)
    endif()
    if (MSVC)
      set(DISABLED_WARNINGS_FOR_FPGA "/wd4996" "/wd4200")
    endif()
  else()
    include_directories(${onnxruntime_BRAINSLICE_LIB_PATH}/build/native/include)
    set(fpga_core_lib ${onnxruntime_BRAINSLICE_LIB_PATH}/build/native/x64/libFPGACoreLib.so)
    #TODO: this is the pre-built v3 bs_client lib, we might need to build it offline for code-gen case
    if (CMAKE_BUILD_TYPE MATCHES Debug)
      set(bs_client_lib ${onnxruntime_BS_CLIENT_PACKAGE}/x64/Debug/libclient.so)
    else(CMAKE_BUILD_TYPE MATCHES Release OR CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
      set(bs_client_lib ${onnxruntime_BS_CLIENT_PACKAGE}/x64/Release/libclient.so)
    endif()
  endif()

  file(GLOB_RECURSE onnxruntime_providers_brainslice_cc_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/brainslice/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/brainslice/*.cc"
  )

  if (CMAKE_COMPILER_IS_GNUCXX)
    set_source_files_properties(${onnxruntime_providers_brainslice_cc_srcs} PROPERTIES COMPILE_FLAGS "-mf16c")
  endif()

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_brainslice_cc_srcs})
  add_library(onnxruntime_providers_brainslice ${onnxruntime_providers_brainslice_cc_srcs})
  target_link_libraries(onnxruntime_providers_brainslice ${fpga_core_lib} ${bs_client_lib})
  onnxruntime_add_include_to_target(onnxruntime_providers_brainslice onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf)
  target_include_directories(onnxruntime_providers_brainslice PRIVATE ${ONNXRUNTIME_ROOT})
  target_include_directories(onnxruntime_providers_brainslice PRIVATE
                             ${onnxruntime_BS_CLIENT_PACKAGE}/inc )
  add_definitions(-DBOOST_LOCALE_NO_LIB)
  add_dependencies(onnxruntime_providers_brainslice ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_brainslice PROPERTIES FOLDER "ONNXRuntime")
  set_target_properties(onnxruntime_providers_brainslice PROPERTIES LINKER_LANGUAGE CXX)
  target_compile_options(onnxruntime_providers_brainslice PRIVATE ${DISABLED_WARNINGS_FOR_FPGA})
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/brainslice  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
  list(APPEND onnxruntime_libs onnxruntime_providers_brainslice ${fpga_core_lib} ${bs_client_lib})
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES brainslice)

  if (WIN32)
    add_custom_command(
    TARGET onnxruntime_providers_brainslice POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${onnxruntime_BRAINSLICE_dynamic_lib_PATH}/build/native/bin/x64/dynamic/FPGACoreLib.dll
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/FPGACoreLib.dll)

    add_custom_command(
    TARGET onnxruntime_providers_brainslice POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${REPO_ROOT}/firmware/NicholasPeakInis
    ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE})

    if (CMAKE_BUILD_TYPE MATCHES Debug)
      add_custom_command(
      TARGET onnxruntime_providers_brainslice POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_BS_CLIENT_PACKAGE}/x64/Debug/client.dll
      ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/client.dll)
    else(CMAKE_BUILD_TYPE MATCHES Release OR CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
      add_custom_command(
      TARGET onnxruntime_providers_brainslice POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_BS_CLIENT_PACKAGE}/x64/Release/client.dll
      ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/client.dll)
    endif()
  else()
    add_custom_command(
    TARGET onnxruntime_providers_brainslice POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${fpga_core_lib}
    ${CMAKE_CURRENT_BINARY_DIR}/libFPGACoreLib.so)

    add_custom_command(
    TARGET onnxruntime_providers_brainslice POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${bs_client_lib}
    ${CMAKE_CURRENT_BINARY_DIR}/libclient.so)
  endif()
endif()
