# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

  if (onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_EXTENDED_MINIMAL_BUILD)
    message(FATAL_ERROR "WebGPU EP can not be used in a basic minimal build. Please build with '--minimal_build extended'")
  endif()

  add_compile_definitions(USE_WEBGPU=1)
  if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
    add_definitions(-DENABLE_WEBASSEMBLY_THREADS=1)
  endif()
  if (onnxruntime_WGSL_TEMPLATE)
    if (onnxruntime_WGSL_TEMPLATE STREQUAL "static")
      add_definitions(-DORT_WGSL_TEMPLATE=1)
    elseif(NOT onnxruntime_WGSL_TEMPLATE STREQUAL "dynamic")
      add_definitions(-DORT_WGSL_TEMPLATE=2)
    else()
      message(FATAL_ERROR "Unsupported value for onnxruntime_WGSL_TEMPLATE: ${onnxruntime_WGSL_TEMPLATE}. Supported values are 'static' or 'dynamic'.")
    endif()
  else()
    add_definitions(-DORT_WGSL_TEMPLATE=0)
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
  target_compile_features(onnxruntime_providers_webgpu PRIVATE cxx_std_20)
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
        target_link_libraries(onnxruntime_providers_webgpu dawn::dawn_native)
      endif()
      target_link_libraries(onnxruntime_providers_webgpu dawn::dawn_proc)
    endif()

    if (WIN32 AND onnxruntime_ENABLE_DAWN_BACKEND_D3D12)
      # Ensure dxil.dll and dxcompiler.dll exist in the output directory $<TARGET_FILE_DIR:dxcompiler>
      # TODO: the following code is used to disable building Dawn using vcpkg temporarily
      # until we figure out how to resolve the packaging pipeline failures
      #
      # if (onnxruntime_USE_VCPKG)
      if (FALSE)
        find_package(directx-dxc CONFIG REQUIRED)
        target_link_libraries(onnxruntime_providers_webgpu Microsoft::DirectXShaderCompiler)
        target_link_libraries(onnxruntime_providers_webgpu Microsoft::DXIL)
        list(APPEND onnxruntime_providers_webgpu_dll_deps "$<TARGET_FILE:Microsoft::DXIL>")
        list(APPEND onnxruntime_providers_webgpu_dll_deps "$<TARGET_FILE:Microsoft::DirectXShaderCompiler>")
      else()
        add_dependencies(onnxruntime_providers_webgpu copy_dxil_dll)
        add_dependencies(onnxruntime_providers_webgpu dxcompiler)

        list(APPEND onnxruntime_providers_webgpu_dll_deps "$<TARGET_FILE_DIR:dxcompiler>/dxil.dll")
        list(APPEND onnxruntime_providers_webgpu_dll_deps "$<TARGET_FILE_DIR:dxcompiler>/dxcompiler.dll")
      endif()
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

  if (onnxruntime_WGSL_TEMPLATE)
    # Define the WGSL templates directory and output directory
    set(WGSL_TEMPLATES_DIR "${ONNXRUNTIME_ROOT}/core/providers/webgpu/wgsl_templates")
    set(WGSL_GENERATED_ROOT "${CMAKE_CURRENT_BINARY_DIR}/wgsl_generated")
    set(WGSL_GENERATED_DIR "${WGSL_GENERATED_ROOT}/wgsl_template_gen")

    # Ensure the output directory exists
    file(MAKE_DIRECTORY ${WGSL_GENERATED_DIR})

    # Find npm and node executables
    find_program(NPM_EXECUTABLE "npm.cmd" "npm" REQUIRED)
    if(NOT NPM_EXECUTABLE)
      message(FATAL_ERROR "npm is required for WGSL template generation but was not found")
    endif()
    find_program(NODE_EXECUTABLE "node" REQUIRED)
    if (NOT NODE_EXECUTABLE)
      message(FATAL_ERROR "Node is required for WGSL template generation but was not found")
    endif()

    # Install npm dependencies
    add_custom_command(
      OUTPUT "${WGSL_TEMPLATES_DIR}/node_modules/.install_complete"
      COMMAND ${NPM_EXECUTABLE} ci
      COMMAND ${CMAKE_COMMAND} -E touch "${WGSL_TEMPLATES_DIR}/node_modules/.install_complete"
      DEPENDS "${WGSL_TEMPLATES_DIR}/package.json" "${WGSL_TEMPLATES_DIR}/package-lock.json"
      WORKING_DIRECTORY ${WGSL_TEMPLATES_DIR}
      COMMENT "Installing npm dependencies for WGSL template generation"
      VERBATIM
    )

    # Find all WGSL template input files
    file(GLOB_RECURSE WGSL_TEMPLATE_FILES "${ONNXRUNTIME_ROOT}/core/providers/webgpu/*.wgsl.template")

    # Define the output files that will be generated
    set(WGSL_GENERATED_INDEX_H "${WGSL_GENERATED_DIR}/index.h")
    set(WGSL_GENERATED_INDEX_IMPL_H "${WGSL_GENERATED_DIR}/index_impl.h")

    # Generate WGSL templates
    add_custom_command(
      OUTPUT ${WGSL_GENERATED_INDEX_H} ${WGSL_GENERATED_INDEX_IMPL_H}
      COMMAND ${NPM_EXECUTABLE} run gen -- -i ../ -o ${WGSL_GENERATED_DIR} -I wgsl_template_gen/ --generator static-cpp --debug
      DEPENDS "${WGSL_TEMPLATES_DIR}/node_modules/.install_complete" ${WGSL_TEMPLATE_FILES}
      WORKING_DIRECTORY ${WGSL_TEMPLATES_DIR}
      COMMENT "Generating WGSL templates from *.wgsl.template files"
      VERBATIM
    )

    # Create a target to represent the generation step
    add_custom_target(onnxruntime_webgpu_wgsl_generation
      DEPENDS ${WGSL_GENERATED_INDEX_H} ${WGSL_GENERATED_INDEX_IMPL_H}
      SOURCES ${WGSL_TEMPLATE_FILES}
    )

    # Add the generated directory to include paths
    target_include_directories(onnxruntime_providers_webgpu PRIVATE ${WGSL_GENERATED_ROOT})

    # Make sure generation happens before building the provider
    add_dependencies(onnxruntime_providers_webgpu onnxruntime_webgpu_wgsl_generation)
  endif()

  set_target_properties(onnxruntime_providers_webgpu PROPERTIES FOLDER "ONNXRuntime")
