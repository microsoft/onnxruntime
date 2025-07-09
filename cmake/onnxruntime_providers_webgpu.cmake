# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  if (onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_EXTENDED_MINIMAL_BUILD)
    message(FATAL_ERROR "WebGPU EP can not be used in a basic minimal build. Please build with '--minimal_build extended'")
  endif()

  add_compile_definitions(USE_WEBGPU=1)
  if(onnxruntime_BUILD_WEBGPU_EP_STATIC_LIB)
    add_compile_definitions(BUILD_WEBGPU_EP_STATIC_LIB=1)
  endif()

  if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
    add_definitions(-DENABLE_WEBASSEMBLY_THREADS=1)
  endif()
  file(GLOB_RECURSE onnxruntime_providers_webgpu_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/webgpu/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/webgpu/*.cc"
  )
  if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
    source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_webgpu_contrib_ops_cc_srcs})
    list(APPEND onnxruntime_providers_webgpu_cc_srcs ${onnxruntime_webgpu_contrib_ops_cc_srcs})
  endif()

  if(onnxruntime_BUILD_WEBGPU_EP_STATIC_LIB)
    #
    # Build WebGPU EP as a static library
    #
    set(onnxruntime_providers_webgpu_srcs ${onnxruntime_providers_webgpu_cc_srcs})
    source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_webgpu_srcs})
    onnxruntime_add_static_library(onnxruntime_providers_webgpu ${onnxruntime_providers_webgpu_srcs})
    onnxruntime_add_include_to_target(onnxruntime_providers_webgpu
      onnxruntime_common onnx onnx_proto flatbuffers::flatbuffers Boost::mp11 safeint_interface)
    add_dependencies(onnxruntime_providers_webgpu onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})
    set_target_properties(onnxruntime_providers_webgpu PROPERTIES CXX_STANDARD_REQUIRED ON)
    set_target_properties(onnxruntime_providers_webgpu PROPERTIES FOLDER "ONNXRuntime")

    set(onnxruntime_providers_webgpu_target onnxruntime_providers_webgpu)
  else()
    #
    # Build WebGPU EP as a shared library
    #
    file(GLOB_RECURSE
         onnxruntime_providers_webgpu_shared_lib_srcs CONFIGURE_DEPENDS
         "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
         "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
    )
    set(onnxruntime_providers_webgpu_srcs ${onnxruntime_providers_webgpu_cc_srcs}
                                         ${onnxruntime_providers_webgpu_shared_lib_srcs})

    source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_webgpu_srcs})

    set(onnxruntime_providers_webgpu_all_srcs ${onnxruntime_providers_webgpu_srcs})
    if(WIN32)
      # Sets the DLL version info on Windows: https://learn.microsoft.com/en-us/windows/win32/menurc/versioninfo-resource
      list(APPEND onnxruntime_providers_webgpu_all_srcs "${ONNXRUNTIME_ROOT}/core/providers/webgpu/onnxruntime_providers_webgpu.rc")
    endif()

    onnxruntime_add_shared_library_module(onnxruntime_providers_webgpu ${onnxruntime_providers_webgpu_all_srcs})
    onnxruntime_add_include_to_target(onnxruntime_providers_webgpu ${ONNXRUNTIME_PROVIDERS_SHARED} ${GSL_TARGET} onnx
                                                                  onnxruntime_common Boost::mp11 safeint_interface)
    target_link_libraries(onnxruntime_providers_webgpu PRIVATE ${ONNXRUNTIME_PROVIDERS_SHARED} ${ABSEIL_LIBS})
    add_dependencies(onnxruntime_providers_webgpu onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})
    target_include_directories(onnxruntime_providers_webgpu PRIVATE ${ONNXRUNTIME_ROOT}
                                                                   ${CMAKE_CURRENT_BINARY_DIR})

    # Set preprocessor definitions used in onnxruntime_providers_webgpu.rc
    if(WIN32)
      set(WEBGPU_DLL_FILE_DESCRIPTION "ONNX Runtime WebGPU Provider")

      target_compile_definitions(onnxruntime_providers_webgpu PRIVATE FILE_DESC=\"${WEBGPU_DLL_FILE_DESCRIPTION}\")
      target_compile_definitions(onnxruntime_providers_webgpu PRIVATE FILE_NAME=\"onnxruntime_providers_webgpu.dll\")
    endif()

    # Set linker flags for function(s) exported by EP DLL
    if(UNIX)
      target_link_options(onnxruntime_providers_webgpu PRIVATE
                          "LINKER:--version-script=${ONNXRUNTIME_ROOT}/core/providers/webgpu/version_script.lds"
                          "LINKER:--gc-sections"
                          "LINKER:-rpath=\$ORIGIN"
      )
    elseif(WIN32)
      set_property(TARGET onnxruntime_providers_webgpu APPEND_STRING PROPERTY LINK_FLAGS
                   "-DEF:${ONNXRUNTIME_ROOT}/core/providers/webgpu/symbols.def")
    else()
      message(FATAL_ERROR "onnxruntime_providers_webgpu unknown platform, need to specify shared library exports for it")
    endif()

    set_target_properties(onnxruntime_providers_webgpu PROPERTIES LINKER_LANGUAGE CXX)
    set_target_properties(onnxruntime_providers_webgpu PROPERTIES CXX_STANDARD_REQUIRED ON)
    set_target_properties(onnxruntime_providers_webgpu PROPERTIES FOLDER "ONNXRuntime")

    install(TARGETS onnxruntime_providers_webgpu
            ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

    set(onnxruntime_providers_webgpu_target onnxruntime_providers_webgpu)
  endif()

  if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    # target "emdawnwebgpu_c" is created by Dawn, including "-fno-exceptions" in its compile options by default.
    #
    # in ONNX Runtime build, "-s DISABLE_EXCEPTION_CATCHING=0" is appended to CMAKE_CXX_FLAGS by default unless build flag
    # "--disable_wasm_exception_catching" is specified. It is not compatible with "-fno-exceptions".
    #
    # if "-s DISABLE_EXCEPTION_CATCHING=0" is set, we need to remove "-fno-exceptions" from emdawnwebgpu_c
    if (CMAKE_CXX_FLAGS MATCHES "DISABLE_EXCEPTION_CATCHING=0")
      get_property(EM_DAWN_WEBGPU_C_COMPILE_OPTIONS TARGET emdawnwebgpu_c PROPERTY COMPILE_OPTIONS)
      list(REMOVE_ITEM EM_DAWN_WEBGPU_C_COMPILE_OPTIONS "-fno-exceptions")
      set_property(TARGET emdawnwebgpu_c PROPERTY COMPILE_OPTIONS ${EM_DAWN_WEBGPU_C_COMPILE_OPTIONS})
    endif()

    # target "emdawnwebgpu_cpp" is created by Dawn. When it is linked to onnxruntime_providers_webgpu as "PUBLIC"
    # dependency, a few build/link flags will be set automatically to make sure emscripten can generate correct
    # WebAssembly/JavaScript code for WebGPU support.
    target_link_libraries(${onnxruntime_providers_webgpu_target} PUBLIC emdawnwebgpu_cpp)

    # ASYNCIFY is required for WGPUFuture support (ie. async functions in WebGPU API)
    target_link_options(${onnxruntime_providers_webgpu_target} PUBLIC
      "SHELL:-s ASYNCIFY=1"
      "SHELL:-s ASYNCIFY_STACK_SIZE=65536"
    )
  else()
    onnxruntime_add_include_to_target(${onnxruntime_providers_webgpu_target} dawn::dawncpp_headers dawn::dawn_headers)

    set(onnxruntime_providers_webgpu_dll_deps)

    if (onnxruntime_BUILD_DAWN_MONOLITHIC_LIBRARY)
      target_link_libraries(${onnxruntime_providers_webgpu_target} dawn::webgpu_dawn)

      if (WIN32)
        if (onnxruntime_ENABLE_DELAY_LOADING_WIN_DLLS)
          list(APPEND onnxruntime_DELAYLOAD_FLAGS "/DELAYLOAD:webgpu_dawn.dll")
        endif()

        # TODO: the following code is used to disable building Dawn using vcpkg temporarily
        # until we figure out how to resolve the packaging pipeline failures
        #
        # if (onnxruntime_USE_VCPKG)
        if (FALSE)
          # Fix Dawn vcpkg build issue (missing IMPORTED_IMPLIB and IMPORTED_LOCATION for target dawn::webgpu_dawn)
          get_target_property(webgpu_dawn_target_IMPORTED_IMPLIB dawn::webgpu_dawn IMPORTED_IMPLIB)
          if (NOT webgpu_dawn_target_IMPORTED_IMPLIB)
            set_target_properties(dawn::webgpu_dawn PROPERTIES IMPORTED_IMPLIB "webgpu_dawn.lib")
          endif()
          get_target_property(webgpu_dawn_target_IMPORTED_LOCATION dawn::webgpu_dawn IMPORTED_LOCATION)
          if (NOT webgpu_dawn_target_IMPORTED_LOCATION)
            set_target_properties(dawn::webgpu_dawn PROPERTIES IMPORTED_LOCATION "webgpu_dawn.dll")
          endif()
        endif()
      endif()

      list(APPEND onnxruntime_providers_webgpu_dll_deps "$<TARGET_FILE:dawn::webgpu_dawn>")
    else()
      if (NOT onnxruntime_USE_EXTERNAL_DAWN)
        target_link_libraries(${onnxruntime_providers_webgpu_target} dawn::dawn_native)
      endif()
      target_link_libraries(${onnxruntime_providers_webgpu_target} dawn::dawn_proc)
    endif()

    if (WIN32 AND onnxruntime_ENABLE_DAWN_BACKEND_D3D12)
      # Ensure dxil.dll and dxcompiler.dll exist in the output directory $<TARGET_FILE_DIR:dxcompiler>
      # TODO: the following code is used to disable building Dawn using vcpkg temporarily
      # until we figure out how to resolve the packaging pipeline failures
      #
      # if (onnxruntime_USE_VCPKG)
      if (FALSE)
        find_package(directx-dxc CONFIG REQUIRED)
        target_link_libraries(${onnxruntime_providers_webgpu_target} Microsoft::DirectXShaderCompiler)
        target_link_libraries(${onnxruntime_providers_webgpu_target} Microsoft::DXIL)
        list(APPEND onnxruntime_providers_webgpu_dll_deps "$<TARGET_FILE:Microsoft::DXIL>")
        list(APPEND onnxruntime_providers_webgpu_dll_deps "$<TARGET_FILE:Microsoft::DirectXShaderCompiler>")
      else()
        add_dependencies(${onnxruntime_providers_webgpu_target} copy_dxil_dll)
        add_dependencies(${onnxruntime_providers_webgpu_target} dxcompiler)

        list(APPEND onnxruntime_providers_webgpu_dll_deps "$<TARGET_FILE_DIR:dxcompiler>/dxil.dll")
        list(APPEND onnxruntime_providers_webgpu_dll_deps "$<TARGET_FILE_DIR:dxcompiler>/dxcompiler.dll")
      endif()
    endif()

    if (onnxruntime_providers_webgpu_dll_deps)
      # Copy dependency DLLs to the output directory
      add_custom_command(
        TARGET ${onnxruntime_providers_webgpu_target}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${onnxruntime_providers_webgpu_dll_deps}" "$<TARGET_FILE_DIR:${onnxruntime_providers_webgpu_target}>"
        COMMAND_EXPAND_LISTS
        VERBATIM )
    endif()
  endif()

if (NOT onnxruntime_BUILD_SHARED_LIB)
  install(TARGETS onnxruntime_providers_webgpu EXPORT ${PROJECT_NAME}Targets
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
