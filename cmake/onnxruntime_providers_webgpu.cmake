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
  if (onnxruntime_WGSL_TEMPLATE STREQUAL "dynamic")
    if (onnxruntime_DISABLE_EXCEPTIONS)
      message(FATAL_ERROR "Dynamic WGSL template generation requires exception handling to be enabled.")
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      message(FATAL_ERROR "Dynamic WGSL template generation is not supported when targeting WebAssembly.")
    endif()
    add_definitions(-DORT_WGSL_TEMPLATE_DYNAMIC=1)
  elseif (NOT onnxruntime_WGSL_TEMPLATE STREQUAL "static")
    message(FATAL_ERROR "Unsupported value for onnxruntime_WGSL_TEMPLATE: ${onnxruntime_WGSL_TEMPLATE}. Supported values are 'static' or 'dynamic'.")
  endif()

  file(GLOB_RECURSE onnxruntime_providers_webgpu_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/webgpu/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/webgpu/*.cc"
  )
  if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
    list(APPEND onnxruntime_providers_webgpu_cc_srcs ${onnxruntime_webgpu_contrib_ops_cc_srcs})
  endif()

  if(onnxruntime_BUILD_WEBGPU_EP_STATIC_LIB)
    #
    # Build WebGPU EP as a static library
    #

    # For static library build, exclude the 'ep' folder
    file(GLOB_RECURSE ep_files_to_exclude
      "${ONNXRUNTIME_ROOT}/core/providers/webgpu/ep/*.h"
      "${ONNXRUNTIME_ROOT}/core/providers/webgpu/ep/*.cc"
    )
    list(REMOVE_ITEM onnxruntime_providers_webgpu_cc_srcs ${ep_files_to_exclude})

    source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_providers_webgpu_cc_srcs})
    onnxruntime_add_static_library(onnxruntime_providers_webgpu ${onnxruntime_providers_webgpu_cc_srcs})
    onnxruntime_add_include_to_target(onnxruntime_providers_webgpu
      onnxruntime_common onnx onnx_proto flatbuffers::flatbuffers Boost::mp11 safeint_interface)
  else()
    #
    # Build WebGPU EP as a shared library
    #
    if(WIN32)
      # Sets the DLL version info on Windows: https://learn.microsoft.com/en-us/windows/win32/menurc/versioninfo-resource
      list(APPEND onnxruntime_providers_webgpu_cc_srcs "${ONNXRUNTIME_ROOT}/core/providers/webgpu/ep/versioninfo.rc")
    endif()
    source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_providers_webgpu_cc_srcs})

    onnxruntime_add_shared_library(onnxruntime_providers_webgpu ${onnxruntime_providers_webgpu_cc_srcs})
    onnxruntime_add_include_to_target(onnxruntime_providers_webgpu
        ${REPO_ROOT}/include/onnxruntime/core/session
        onnxruntime_common
        onnx
        onnx_proto
        flatbuffers::flatbuffers
        Boost::mp11
        safeint_interface)

    target_link_libraries(onnxruntime_providers_webgpu PRIVATE
        onnxruntime_optimizer
        onnxruntime_providers
        onnxruntime_lora
        onnxruntime_framework
        onnxruntime_graph
        onnxruntime_util
        ${ONNXRUNTIME_MLAS_LIBS}
        onnxruntime_common
        onnxruntime_flatbuffers
        ${onnxruntime_EXTERNAL_LIBRARIES}
    )

    # Add ONNX compiler definitions
    add_definitions("-DONNX_ML=1")
    add_definitions("-DONNX_NAMESPACE=onnx")
    add_definitions("-DONNX_USE_LITE_PROTO=1")

    # Set preprocessor definitions used in onnxruntime_providers_webgpu.rc
    if(WIN32)
      set(WEBGPU_DLL_FILE_DESCRIPTION "ONNX Runtime WebGPU Provider")

      target_compile_definitions(onnxruntime_providers_webgpu PRIVATE FILE_DESC=\"${WEBGPU_DLL_FILE_DESCRIPTION}\")
      target_compile_definitions(onnxruntime_providers_webgpu PRIVATE FILE_NAME=\"onnxruntime_providers_webgpu.dll\")
    endif()

    # Set linker flags for function(s) exported by EP DLL
    if(UNIX)
      if (APPLE)
        set_property(TARGET onnxruntime_providers_webgpu APPEND_STRING PROPERTY LINK_FLAGS
                     "-Xlinker -dead_strip")
      elseif (NOT CMAKE_SYSTEM_NAME MATCHES "AIX")
        target_link_options(onnxruntime_providers_webgpu PRIVATE
                            "LINKER:--version-script=${ONNXRUNTIME_ROOT}/core/providers/webgpu/ep/version_script.lds"
                            "LINKER:--gc-sections"
                            "LINKER:-rpath=\$ORIGIN")
        # TODO: -z noexecstack
      endif()
    elseif(WIN32)
      set_property(TARGET onnxruntime_providers_webgpu APPEND_STRING PROPERTY LINK_FLAGS
                   "-DEF:${ONNXRUNTIME_ROOT}/core/providers/webgpu/ep/symbols.def")
    else()
      message(FATAL_ERROR "onnxruntime_providers_webgpu unknown platform, need to specify shared library exports for it")
    endif()

    set_target_properties(onnxruntime_providers_webgpu PROPERTIES LINKER_LANGUAGE CXX)

    if (onnxruntime_BUILD_CACHE)
      message(FATAL_ERROR "WebGPU EP shared library build does not support build cache. Please disable build cache or use static library build.")
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      message(FATAL_ERROR "WebGPU EP shared library build is not supported on Emscripten. Please use static library build.")
    endif()
  endif()

  set_target_properties(onnxruntime_providers_webgpu PROPERTIES CXX_STANDARD_REQUIRED ON)
  set_target_properties(onnxruntime_providers_webgpu PROPERTIES FOLDER "ONNXRuntime")

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
    if (CMAKE_CXX_FLAGS MATCHES "-fwasm-exceptions")
      get_property(EM_DAWN_WEBGPU_C_COMPILE_OPTIONS TARGET emdawnwebgpu_c PROPERTY COMPILE_OPTIONS)
      list(REMOVE_ITEM EM_DAWN_WEBGPU_C_COMPILE_OPTIONS "-fno-exceptions")
      set_property(TARGET emdawnwebgpu_c PROPERTY COMPILE_OPTIONS ${EM_DAWN_WEBGPU_C_COMPILE_OPTIONS})
    endif()

    # target "emdawnwebgpu_cpp" is created by Dawn. When it is linked to onnxruntime_providers_webgpu as "PUBLIC"
    # dependency, a few build/link flags will be set automatically to make sure emscripten can generate correct
    # WebAssembly/JavaScript code for WebGPU support.
    target_link_libraries(onnxruntime_providers_webgpu PUBLIC emdawnwebgpu_cpp)

    # Dawn's emdawnwebgpu_cpp target has a bug: it lists ${DAWN_INCLUDE_DIR}/webgpu/webgpu_enum_class_bitmasks.h
    # in INTERFACE_SOURCES but doesn't add ${DAWN_INCLUDE_DIR} to INTERFACE_INCLUDE_DIRECTORIES.
    # In emsdk 4.0.11, this was masked because Emscripten bundled its own copy of the WebGPU headers.
    # In emsdk 4.0.21+, Emscripten removed the bundled WebGPU headers, exposing this bug.
    # We need to manually add the Dawn include directory to find webgpu_enum_class_bitmasks.h.
    #
    # IMPORTANT: We must also add the generated emdawnwebgpu include directory BEFORE the Dawn source
    # include directory, because ${dawn_SOURCE_DIR}/include/webgpu/webgpu_cpp.h is a stub that redirects
    # to dawn/webgpu_cpp.h (native Dawn), but we need the generated Emscripten-specific webgpu_cpp.h.
    target_include_directories(onnxruntime_providers_webgpu PRIVATE
        "${dawn_BINARY_DIR}/gen/src/emdawnwebgpu/include"
        "${dawn_SOURCE_DIR}/include"
    )

    if (onnxruntime_ENABLE_WEBASSEMBLY_JSPI)
      target_link_options(onnxruntime_providers_webgpu PUBLIC
        "SHELL:-s JSPI=1"
      )
    else()
      # ASYNCIFY is required for WGPUFuture support (ie. async functions in WebGPU API)
      target_link_options(onnxruntime_providers_webgpu PUBLIC
        "SHELL:-s ASYNCIFY=1"
        "SHELL:-s ASYNCIFY_STACK_SIZE=65536"
      )
    endif()
  else()
    onnxruntime_add_include_to_target(onnxruntime_providers_webgpu dawn::dawncpp_headers dawn::dawn_headers)

    set(onnxruntime_providers_webgpu_dll_deps)

    if (onnxruntime_BUILD_DAWN_SHARED_LIBRARY)
      target_link_libraries(onnxruntime_providers_webgpu PUBLIC dawn::webgpu_dawn)

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
        target_link_libraries(onnxruntime_providers_webgpu PRIVATE dawn::dawn_native)
      endif()
      target_link_libraries(onnxruntime_providers_webgpu PRIVATE dawn::dawn_proc)
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

  target_compile_features(onnxruntime_providers_webgpu PRIVATE cxx_std_20)
  add_dependencies(onnxruntime_providers_webgpu onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})

  if (onnxruntime_WGSL_TEMPLATE)
    # Define the WGSL templates directory and output directory
    set(WGSL_TEMPLATES_DIR "${ONNXRUNTIME_ROOT}/core/providers/webgpu/wgsl_templates")
    set(WGSL_GENERATED_ROOT "${CMAKE_CURRENT_BINARY_DIR}/wgsl_generated")

    # Include the Node.js helper for finding and validating Node.js and NPM
    include(node_helper.cmake)

    # Install npm dependencies
    add_custom_command(
      OUTPUT "${WGSL_TEMPLATES_DIR}/node_modules/.install_complete"
      COMMAND ${NPM_CLI} ci
      COMMAND ${CMAKE_COMMAND} -E touch "${WGSL_TEMPLATES_DIR}/node_modules/.install_complete"
      DEPENDS "${WGSL_TEMPLATES_DIR}/package.json" "${WGSL_TEMPLATES_DIR}/package-lock.json"
      WORKING_DIRECTORY ${WGSL_TEMPLATES_DIR}
      COMMENT "Installing npm dependencies for WGSL template generation"
      VERBATIM
    )

    if (onnxruntime_WGSL_TEMPLATE STREQUAL "static")
      set(WGSL_GENERATED_DIR "${WGSL_GENERATED_ROOT}/wgsl_template_gen")
      # set(WGSL_GEN_OUTPUTS "${WGSL_GENERATED_DIR}/index.h" "${WGSL_GENERATED_DIR}/index_impl.h")
      # Define the output files that will be generated
      set(WGSL_GENERATED_INDEX_H "${WGSL_GENERATED_DIR}/index.h")
      set(WGSL_GENERATED_INDEX_IMPL_H "${WGSL_GENERATED_DIR}/index_impl.h")
    elseif(onnxruntime_WGSL_TEMPLATE STREQUAL "dynamic")
      set(WGSL_GENERATED_DIR "${WGSL_GENERATED_ROOT}/dynamic")
      # set(WGSL_GEN_OUTPUTS "${WGSL_GENERATED_DIR}/templates.js")
      set(WGSL_GENERATED_TEMPLATES_JS "${WGSL_GENERATED_DIR}/templates.js")
    endif()

    # Ensure the output directory exists
    file(MAKE_DIRECTORY ${WGSL_GENERATED_DIR})

    # Find all WGSL template input files
    set(WGSL_SEARCH_PATHS "${ONNXRUNTIME_ROOT}/core/providers/webgpu/*.wgsl.template")
    if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
        list(APPEND WGSL_SEARCH_PATHS "${ONNXRUNTIME_ROOT}/contrib_ops/webgpu/*.wgsl.template")
    endif()
    file(GLOB_RECURSE WGSL_TEMPLATE_FILES ${WGSL_SEARCH_PATHS})

    # Set wgsl-gen command line options as a list
    set(WGSL_GEN_OPTIONS
        "--output" "${WGSL_GENERATED_DIR}"
        "-I" "wgsl_template_gen/"
        "--preserve-code-ref"
        "--verbose"
        "-i" "${ONNXRUNTIME_ROOT}/core/providers/webgpu"
    )
    if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
        list(APPEND WGSL_GEN_OPTIONS "-i" "${ONNXRUNTIME_ROOT}/contrib_ops/webgpu")
    endif()

    if (onnxruntime_WGSL_TEMPLATE STREQUAL "static")
      if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        list(APPEND WGSL_GEN_OPTIONS "--generator" "static-cpp-literal")
      else()
        list(APPEND WGSL_GEN_OPTIONS "--generator" "static-cpp")
      endif()
    elseif(onnxruntime_WGSL_TEMPLATE STREQUAL "dynamic")
      list(APPEND WGSL_GEN_OPTIONS "--generator" "dynamic")
    endif()

    # Generate WGSL templates
    add_custom_command(
      OUTPUT ${WGSL_GENERATED_INDEX_H} ${WGSL_GENERATED_INDEX_IMPL_H} ${WGSL_GENERATED_TEMPLATES_JS}
      COMMAND ${NPM_CLI} run gen -- ${WGSL_GEN_OPTIONS}
      DEPENDS "${WGSL_TEMPLATES_DIR}/node_modules/.install_complete" ${WGSL_TEMPLATE_FILES}
      WORKING_DIRECTORY ${WGSL_TEMPLATES_DIR}
      COMMENT "Generating WGSL templates from *.wgsl.template files"
      COMMAND_EXPAND_LISTS
      VERBATIM
    )

    # Create a target to represent the generation step
    add_custom_target(onnxruntime_webgpu_wgsl_generation
      DEPENDS ${WGSL_GENERATED_INDEX_H} ${WGSL_GENERATED_INDEX_IMPL_H} ${WGSL_GENERATED_TEMPLATES_JS}
      SOURCES ${WGSL_TEMPLATE_FILES}
    )

    if (onnxruntime_WGSL_TEMPLATE STREQUAL "static")
      # Add the generated directory to include paths
      target_include_directories(onnxruntime_providers_webgpu PRIVATE ${WGSL_GENERATED_ROOT})
    elseif(onnxruntime_WGSL_TEMPLATE STREQUAL "dynamic")
      target_link_libraries(onnxruntime_providers_webgpu PRIVATE duktape_static)
      onnxruntime_add_include_to_target(onnxruntime_providers_webgpu duktape_static)

      # Define the path to the generated templates.js file
      target_compile_definitions(onnxruntime_providers_webgpu PRIVATE
        "ORT_WGSL_TEMPLATES_JS_PATH=\"${WGSL_GENERATED_TEMPLATES_JS}\"")
    endif()

    # Make sure generation happens before building the provider
    add_dependencies(onnxruntime_providers_webgpu onnxruntime_webgpu_wgsl_generation)
  endif()

  if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_providers_webgpu EXPORT ${PROJECT_NAME}Targets
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
