# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  add_compile_definitions(USE_QNN=1)


remove_definitions(-DBUILD_QNN_EP_STATIC_LIB)
add_compile_definitions(BUILD_QNN_EP_STATIC_LIB=0)
# add_compile_definitions(QNN_ABI_SHARED_LIBRARY_ONLY=1)

  file(GLOB_RECURSE
       onnxruntime_providers_qnn_abi_ep_srcs CONFIGURE_DEPENDS
       "${ONNXRUNTIME_ROOT}/core/providers/qnn-abi/*.h"
       "${ONNXRUNTIME_ROOT}/core/providers/qnn-abi/*.cc"
  )
  # Exclude the actual EP factory files from the build
  list(REMOVE_ITEM onnxruntime_providers_qnn_abi_ep_srcs
       "${ONNXRUNTIME_ROOT}/core/providers/qnn-abi/qnn_ep_factory.h"
       "${ONNXRUNTIME_ROOT}/core/providers/qnn-abi/qnn_ep_factory.cc")


  set(example_plugin_ep_utils_srcs "${ONNXRUNTIME_ROOT}/test/autoep/library/example_plugin_ep_utils.cc")

  function(extract_qnn_sdk_version_from_yaml QNN_SDK_YAML_FILE QNN_VERSION_OUTPUT)
    file(READ "${QNN_SDK_YAML_FILE}" QNN_SDK_YAML_CONTENT)
    # Match a line of text like "version: 1.33.2"
    string(REGEX MATCH "(^|\n|\r)version: ([0-9]+\\.[0-9]+\\.[0-9]+)" QNN_VERSION_MATCH "${QNN_SDK_YAML_CONTENT}")
    if(QNN_VERSION_MATCH)
      set(${QNN_VERSION_OUTPUT} "${CMAKE_MATCH_2}" PARENT_SCOPE)
      message(STATUS "Extracted QNN SDK version ${CMAKE_MATCH_2} from ${QNN_SDK_YAML_FILE}")
    else()
      message(WARNING "Failed to extract QNN SDK version from ${QNN_SDK_YAML_FILE}")
    endif()
  endfunction()

  if(NOT QNN_SDK_VERSION)
    if(EXISTS "${onnxruntime_QNN_HOME}/sdk.yaml")
      extract_qnn_sdk_version_from_yaml("${onnxruntime_QNN_HOME}/sdk.yaml" QNN_SDK_VERSION)
    else()
      message(WARNING "Cannot open sdk.yaml to extract QNN SDK version")
    endif()
  endif()
  message(STATUS "QNN SDK version ${QNN_SDK_VERSION}")

  file(GLOB_RECURSE
     onnxruntime_providers_qnn_shared_lib_srcs CONFIGURE_DEPENDS
     "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
     "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )

  set(onnxruntime_providers_qnn_abi_srcs ${onnxruntime_providers_qnn_abi_ep_srcs}
                                  #  ${onnxruntime_providers_qnn_shared_lib_srcs}
                                          ${example_plugin_ep_utils_srcs})

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_qnn_abi_ep_srcs}) # ${onnxruntime_providers_qnn_shared_lib_srcs})
  source_group(TREE ${ONNXRUNTIME_ROOT}/test FILES ${example_plugin_ep_utils_srcs})

  set(onnxruntime_providers_qnn_abi_all_srcs ${onnxruntime_providers_qnn_abi_srcs})
  if(WIN32)
    # Sets the DLL version info on Windows: https://learn.microsoft.com/en-us/windows/win32/menurc/versioninfo-resource
    list(APPEND onnxruntime_providers_qnn_abi_all_srcs "${ONNXRUNTIME_ROOT}/core/providers/qnn-abi/onnxruntime_providers_qnn_abi_simulation.rc")
  endif()

  onnxruntime_add_shared_library_module(onnxruntime_providers_qnn_abi_simulation ${onnxruntime_providers_qnn_abi_all_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_qnn_abi_simulation ${ONNXRUNTIME_PROVIDERS_SHARED} ${GSL_TARGET} onnx
                                                                             onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers onnxruntime_common Boost::mp11 safeint_interface
                                                                             nlohmann_json::nlohmann_json)
  target_link_libraries(onnxruntime_providers_qnn_abi_simulation PRIVATE ${ONNXRUNTIME_PROVIDERS_SHARED} ${ABSEIL_LIBS} ${CMAKE_DL_LIBS} onnxruntime_common ${PROTOBUF_LIB} onnx_proto)

  # Link cpuinfo if supported - needed for CPU feature detection
  if (CPUINFO_SUPPORTED)
    onnxruntime_add_include_to_target(onnxruntime_providers_qnn_abi_simulation cpuinfo::cpuinfo)
    target_link_libraries(onnxruntime_providers_qnn_abi_simulation PRIVATE cpuinfo::cpuinfo ${ONNXRUNTIME_CLOG_TARGET_NAME})
  endif()

  add_dependencies(onnxruntime_providers_qnn_abi_simulation onnxruntime_providers_shared onnx onnx_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})
  target_include_directories(onnxruntime_providers_qnn_abi_simulation PRIVATE ${ONNXRUNTIME_ROOT}
                                                                              ${CMAKE_CURRENT_BINARY_DIR}
                                                                              ${onnxruntime_QNN_HOME}/include/QNN
                                                                              ${onnxruntime_QNN_HOME}/include)


  target_compile_definitions(onnxruntime_providers_qnn PRIVATE
    BUILD_QNN_EP_STATIC_LIB=0
    SHARED_PROVIDER=1
  )
  # Set preprocessor definitions used in onnxruntime_providers_qnn_abi_simulation.rc
  if(WIN32)
    if(NOT QNN_SDK_VERSION)
      set(QNN_DLL_FILE_DESCRIPTION "ONNX Runtime QNN Provider")
    else()
      set(QNN_DLL_FILE_DESCRIPTION "ONNX Runtime QNN Provider (QAIRT ${QNN_SDK_VERSION})")
    endif()

    target_compile_definitions(onnxruntime_providers_qnn_abi_simulation PRIVATE FILE_DESC=\"${QNN_DLL_FILE_DESCRIPTION}\")
    target_compile_definitions(onnxruntime_providers_qnn_abi_simulation PRIVATE FILE_NAME=\"onnxruntime_providers_qnn_abi_simulation.dll\")
  endif()

  # Set linker flags for function(s) exported by EP DLL
  if(UNIX)
    target_link_options(onnxruntime_providers_qnn_abi_simulation PRIVATE
                        "LINKER:--version-script=${ONNXRUNTIME_ROOT}/core/providers/qnn-abi/version_script.lds"
                        "LINKER:--gc-sections"
                        "LINKER:-rpath=\$ORIGIN"
    )
  elseif(WIN32)
    set_property(TARGET onnxruntime_providers_qnn_abi_simulation APPEND_STRING PROPERTY LINK_FLAGS
                  "-DEF:${ONNXRUNTIME_ROOT}/core/providers/qnn-abi/symbols.def")
  else()
    message(FATAL_ERROR "onnxruntime_providers_qnn_abi_simulation unknown platform, need to specify shared library exports for it")
  endif()

  # Set compile options
  if(MSVC)
    target_compile_options(onnxruntime_providers_qnn_abi_simulation PUBLIC /wd4099 /wd4005)
  else()
    # ignore the warning unknown-pragmas on "pragma region"
    target_compile_options(onnxruntime_providers_qnn_abi_simulation PRIVATE "-Wno-unknown-pragmas")
  endif()

  set_target_properties(onnxruntime_providers_qnn_abi_simulation PROPERTIES LINKER_LANGUAGE CXX)
  set_target_properties(onnxruntime_providers_qnn_abi_simulation PROPERTIES CXX_STANDARD_REQUIRED ON)
  set_target_properties(onnxruntime_providers_qnn_abi_simulation PROPERTIES FOLDER "ONNXRuntime")

  install(TARGETS onnxruntime_providers_qnn_abi_simulation
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

  set(onnxruntime_providers_qnn_abi_simulation_target onnxruntime_providers_qnn_abi_simulation)

  if (MSVC OR ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    # Create destination directory first to ensure it exists
    add_custom_command(
      TARGET ${onnxruntime_providers_qnn_abi_simulation_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_abi_simulation_target}>
      COMMENT "Creating QNN library destination directory"
    )
    
    # Copy QNN library files with better error handling
    if(QNN_LIB_FILES)
      foreach(QNN_LIB_FILE ${QNN_LIB_FILES})
        add_custom_command(
          TARGET ${onnxruntime_providers_qnn_abi_simulation_target} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different "${QNN_LIB_FILE}" $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_abi_simulation_target}>
          COMMENT "Copying QNN library: ${QNN_LIB_FILE}"
        )
      endforeach()
    endif()
  endif()
  if (EXISTS "${onnxruntime_QNN_HOME}/Qualcomm AI Hub Proprietary License.pdf")
    add_custom_command(
      TARGET ${onnxruntime_providers_qnn_abi_simulation_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy "${onnxruntime_QNN_HOME}/Qualcomm AI Hub Proprietary License.pdf" $<TARGET_FILE_DIR:${onnxruntime_providers_qnn_abi_simulation_target}>
      )
  endif()

if (NOT onnxruntime_BUILD_SHARED_LIB)
  install(TARGETS onnxruntime_providers_qnn_abi_simulation EXPORT ${PROJECT_NAME}Targets
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
