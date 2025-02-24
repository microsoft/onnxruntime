# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  if (onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_EXTENDED_MINIMAL_BUILD)
    message(FATAL_ERROR "WebGPU EP can not be used in a basic minimal build. Please build with '--minimal_build extended'")
  endif()

  add_compile_definitions(USE_WEBGPU=1)
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

  source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_providers_webgpu_cc_srcs})
  onnxruntime_add_static_library(onnxruntime_providers_webgpu ${onnxruntime_providers_webgpu_cc_srcs})
  onnxruntime_add_include_to_target(onnxruntime_providers_webgpu
    onnxruntime_common onnx onnx_proto flatbuffers::flatbuffers Boost::mp11 safeint_interface)

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
    target_link_libraries(onnxruntime_providers_webgpu PUBLIC emdawnwebgpu_cpp)

    # ASYNCIFY is required for WGPUFuture support (ie. async functions in WebGPU API)
    target_link_options(onnxruntime_providers_webgpu PUBLIC
      "SHELL:-s ASYNCIFY=1"
      "SHELL:-s ASYNCIFY_STACK_SIZE=65536"
    )
  else()
    onnxruntime_add_include_to_target(onnxruntime_providers_webgpu dawn::dawncpp_headers dawn::dawn_headers)

    set(onnxruntime_providers_webgpu_dll_deps)

    if (onnxruntime_BUILD_DAWN_MONOLITHIC_LIBRARY)
      target_link_libraries(onnxruntime_providers_webgpu dawn::webgpu_dawn)

      if (WIN32)
        if (onnxruntime_ENABLE_DELAY_LOADING_WIN_DLLS)
          list(APPEND onnxruntime_DELAYLOAD_FLAGS "/DELAYLOAD:webgpu_dawn.dll")
        endif()

        list(APPEND onnxruntime_providers_webgpu_dll_deps "$<TARGET_FILE:dawn::webgpu_dawn>")
      endif()
    else()
      if (NOT onnxruntime_USE_EXTERNAL_DAWN)
        target_link_libraries(onnxruntime_providers_webgpu dawn::dawn_native)
      endif()
      target_link_libraries(onnxruntime_providers_webgpu dawn::dawn_proc)
    endif()

    if (WIN32 AND onnxruntime_ENABLE_DAWN_BACKEND_D3D12)
      # Ensure dxil.dll and dxcompiler.dll exist in the output directory $<TARGET_FILE_DIR:dxcompiler>
      add_dependencies(onnxruntime_providers_webgpu copy_dxil_dll)
      add_dependencies(onnxruntime_providers_webgpu dxcompiler)

      list(APPEND onnxruntime_providers_webgpu_dll_deps "$<TARGET_FILE_DIR:dxcompiler>/dxil.dll")
      list(APPEND onnxruntime_providers_webgpu_dll_deps "$<TARGET_FILE_DIR:dxcompiler>/dxcompiler.dll")
    endif()

    if (onnxruntime_providers_webgpu_dll_deps)
      # Copy dependency DLLs to the output directory
      add_custom_command(
        TARGET onnxruntime_providers_webgpu
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${onnxruntime_providers_webgpu_dll_deps}" "$<TARGET_FILE_DIR:onnxruntime_providers_webgpu>"
        COMMAND_EXPAND_LISTS
        VERBATIM )
    endif()
  endif()

  add_dependencies(onnxruntime_providers_webgpu ${onnxruntime_EXTERNAL_DEPENDENCIES})
  set_target_properties(onnxruntime_providers_webgpu PROPERTIES FOLDER "ONNXRuntime")
