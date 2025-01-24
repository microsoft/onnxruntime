# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  add_compile_definitions(USE_QNN=1)

  if(onnxruntime_BUILD_QNN_EP_STATIC_LIB)
    add_compile_definitions(BUILD_QNN_EP_STATIC_LIB=1)
  endif()

  file(GLOB_RECURSE
       onnxruntime_providers_qnn_ep_srcs CONFIGURE_DEPENDS
       "${ONNXRUNTIME_ROOT}/core/providers/qnn/*.h"
       "${ONNXRUNTIME_ROOT}/core/providers/qnn/*.cc"
  )

  if(onnxruntime_BUILD_QNN_EP_STATIC_LIB)
    #
    # Build QNN EP as a static library
    #
    set(onnxruntime_providers_qnn_srcs ${onnxruntime_providers_qnn_ep_srcs})
    source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_qnn_srcs})
    onnxruntime_add_static_library(onnxruntime_providers_qnn ${onnxruntime_providers_qnn_srcs})
    onnxruntime_add_include_to_target(onnxruntime_providers_qnn onnxruntime_common onnxruntime_framework onnx
                                                                onnx_proto protobuf::libprotobuf-lite
                                                                flatbuffers::flatbuffers Boost::mp11
								nlohmann_json::nlohmann_json)
    add_dependencies(onnxruntime_providers_qnn onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
    set_target_properties(onnxruntime_providers_qnn PROPERTIES CXX_STANDARD_REQUIRED ON)
    set_target_properties(onnxruntime_providers_qnn PROPERTIES FOLDER "ONNXRuntime")
    target_include_directories(onnxruntime_providers_qnn PRIVATE ${ONNXRUNTIME_ROOT}
                                                                 ${onnxruntime_QNN_HOME}/include/QNN
                                                                 ${onnxruntime_QNN_HOME}/include)
    set_target_properties(onnxruntime_providers_qnn PROPERTIES LINKER_LANGUAGE CXX)

    # ignore the warning unknown-pragmas on "pragma region"
    if(NOT MSVC)
      target_compile_options(onnxruntime_providers_qnn PRIVATE "-Wno-unknown-pragmas")
    endif()
  else()
    #
    # Build QNN EP as a shared library
    #
    file(GLOB_RECURSE
         onnxruntime_providers_qnn_shared_lib_srcs CONFIGURE_DEPENDS
         "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
         "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
    )
    set(onnxruntime_providers_qnn_srcs ${onnxruntime_providers_qnn_ep_srcs}
	                               ${onnxruntime_providers_qnn_shared_lib_srcs})

    source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_qnn_srcs})
    onnxruntime_add_shared_library_module(onnxruntime_providers_qnn ${onnxruntime_providers_qnn_srcs})
    onnxruntime_add_include_to_target(onnxruntime_providers_qnn ${ONNXRUNTIME_PROVIDERS_SHARED} ${GSL_TARGET} onnx
	                                                        onnxruntime_common Boost::mp11 safeint_interface
								nlohmann_json::nlohmann_json)
    target_link_libraries(onnxruntime_providers_qnn PRIVATE ${ONNXRUNTIME_PROVIDERS_SHARED} ${ABSEIL_LIBS} ${CMAKE_DL_LIBS})
    add_dependencies(onnxruntime_providers_qnn onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})
    target_include_directories(onnxruntime_providers_qnn PRIVATE ${ONNXRUNTIME_ROOT}
                                                                 ${CMAKE_CURRENT_BINARY_DIR}
                                                                 ${onnxruntime_QNN_HOME}/include/QNN
                                                                 ${onnxruntime_QNN_HOME}/include)

    # Set linker flags for function(s) exported by EP DLL
    if(UNIX)
      target_link_options(onnxruntime_providers_qnn PRIVATE
                          "LINKER:--version-script=${ONNXRUNTIME_ROOT}/core/providers/qnn/version_script.lds"
                          "LINKER:--gc-sections"
                          "LINKER:-rpath=\$ORIGIN"
      )
    elseif(WIN32)
      set_property(TARGET onnxruntime_providers_qnn APPEND_STRING PROPERTY LINK_FLAGS
                   "-DEF:${ONNXRUNTIME_ROOT}/core/providers/qnn/symbols.def")
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
  endif()
