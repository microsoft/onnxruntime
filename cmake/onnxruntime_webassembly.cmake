# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

function(bundle_static_library bundled_target_name)
  function(recursively_collect_dependencies input_target)
    set(input_link_libraries LINK_LIBRARIES)
    get_target_property(input_type ${input_target} TYPE)
    if (${input_type} STREQUAL "INTERFACE_LIBRARY")
      set(input_link_libraries INTERFACE_LINK_LIBRARIES)
    endif()
    get_target_property(public_dependencies ${input_target} ${input_link_libraries})
    foreach(dependency IN LISTS public_dependencies)
      if(TARGET ${dependency})
        get_target_property(alias ${dependency} ALIASED_TARGET)
        if (TARGET ${alias})
          set(dependency ${alias})
        endif()
        get_target_property(type ${dependency} TYPE)
        if (${type} STREQUAL "STATIC_LIBRARY")
          list(APPEND static_libs ${dependency})
        endif()

        get_property(library_already_added GLOBAL PROPERTY ${target_name}_static_bundle_${dependency})
        if (NOT library_already_added)
          set_property(GLOBAL PROPERTY ${target_name}_static_bundle_${dependency} ON)
          recursively_collect_dependencies(${dependency})
        endif()
      endif()
    endforeach()
    set(static_libs ${static_libs} PARENT_SCOPE)
  endfunction()

  foreach(target_name IN ITEMS ${ARGN})
    list(APPEND static_libs ${target_name})
    recursively_collect_dependencies(${target_name})
  endforeach()

  list(REMOVE_DUPLICATES static_libs)

  set(bundled_target_full_name
    ${CMAKE_BINARY_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${bundled_target_name}${CMAKE_STATIC_LIBRARY_SUFFIX})

  file(WRITE ${CMAKE_BINARY_DIR}/${bundled_target_name}.ar.in
    "CREATE ${bundled_target_full_name}\n" )

  foreach(target IN LISTS static_libs)
    file(APPEND ${CMAKE_BINARY_DIR}/${bundled_target_name}.ar.in
      "ADDLIB $<TARGET_FILE:${target}>\n")
  endforeach()

  file(APPEND ${CMAKE_BINARY_DIR}/${bundled_target_name}.ar.in "SAVE\n")
  file(APPEND ${CMAKE_BINARY_DIR}/${bundled_target_name}.ar.in "END\n")

  file(GENERATE
    OUTPUT ${CMAKE_BINARY_DIR}/${bundled_target_name}.ar
    INPUT ${CMAKE_BINARY_DIR}/${bundled_target_name}.ar.in)

  set(ar_tool ${CMAKE_AR})
  if (CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    set(ar_tool ${CMAKE_CXX_COMPILER_AR})
  endif()

  add_custom_command(
    COMMAND ${ar_tool} -M < ${CMAKE_BINARY_DIR}/${bundled_target_name}.ar
    OUTPUT ${bundled_target_full_name}
    COMMENT "Bundling ${bundled_target_name}"
    VERBATIM)

  add_custom_target(bundling_target ALL DEPENDS ${bundled_target_full_name})
  foreach(target_name IN ITEMS ${ARGN})
    add_dependencies(bundling_target ${target_name})
  endforeach()

  add_library(${bundled_target_name} STATIC IMPORTED GLOBAL)
  set_target_properties(${bundled_target_name}
    PROPERTIES
      IMPORTED_LOCATION ${bundled_target_full_name})
  foreach(target_name IN ITEMS ${ARGN})
    set_property(TARGET ${bundled_target_name} APPEND
      PROPERTY INTERFACE_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:${target_name},INTERFACE_INCLUDE_DIRECTORIES>)
    set_property(TARGET ${bundled_target_name} APPEND
      PROPERTY INTERFACE_COMPILE_DEFINITIONS $<TARGET_PROPERTY:${target_name},INTERFACE_COMPILE_DEFINITIONS>)
  endforeach()
  add_dependencies(${bundled_target_name} bundling_target)
endfunction()

if (NOT onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
  add_compile_definitions(
    BUILD_MLAS_NO_ONNXRUNTIME
  )

  # Override re2 compiler options to remove -pthread
  set_property(TARGET re2 PROPERTY COMPILE_OPTIONS )
endif()

target_compile_options(onnx PRIVATE -Wno-unused-parameter -Wno-unused-variable)

if (onnxruntime_BUILD_WEBASSEMBLY_STATIC_LIB)
    bundle_static_library(onnxruntime_webassembly
      nsync::nsync_cpp
      ${PROTOBUF_LIB}
      onnx
      onnx_proto
      onnxruntime_common
      onnxruntime_flatbuffers
      onnxruntime_framework
      onnxruntime_graph
      onnxruntime_mlas
      onnxruntime_optimizer
      onnxruntime_providers
      ${PROVIDERS_JS}
      ${PROVIDERS_XNNPACK}
      ${PROVIDERS_WEBNN}
      onnxruntime_session
      onnxruntime_util
      re2::re2
    )

    if (onnxruntime_ENABLE_TRAINING)
      bundle_static_library(onnxruntime_webassembly tensorboard)
    endif()

    if (onnxruntime_BUILD_UNIT_TESTS)
      file(GLOB_RECURSE onnxruntime_webassembly_test_src CONFIGURE_DEPENDS
        "${ONNXRUNTIME_ROOT}/test/wasm/test_main.cc"
        "${ONNXRUNTIME_ROOT}/test/wasm/test_inference.cc"
      )

      source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_webassembly_test_src})

      add_executable(onnxruntime_webassembly_test
        ${onnxruntime_webassembly_test_src}
      )

      set_target_properties(onnxruntime_webassembly_test PROPERTIES LINK_FLAGS
        "-s ALLOW_MEMORY_GROWTH=1 -s \"EXPORTED_RUNTIME_METHODS=['FS']\" --preload-file ${CMAKE_CURRENT_BINARY_DIR}/testdata@/testdata -s EXIT_RUNTIME=1"
      )

      target_link_libraries(onnxruntime_webassembly_test PUBLIC
        onnxruntime_webassembly
        GTest::gtest
      )

      find_program(NODE_EXECUTABLE node required)
      if (NOT NODE_EXECUTABLE)
        message(FATAL_ERROR "Node is required for a test")
      endif()

      add_test(NAME onnxruntime_webassembly_test
        COMMAND ${NODE_EXECUTABLE} onnxruntime_webassembly_test.js
        WORKING_DIRECTORY $<TARGET_FILE_DIR:onnxruntime_webassembly_test>
      )
    endif()
else()
  file(GLOB_RECURSE onnxruntime_webassembly_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/wasm/api.cc"
  )

  source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_webassembly_src})

  add_executable(onnxruntime_webassembly
    ${onnxruntime_webassembly_src}
  )

  if (onnxruntime_ENABLE_WEBASSEMBLY_API_EXCEPTION_CATCHING)
    # we catch exceptions at the api level
    file(GLOB_RECURSE onnxruntime_webassembly_src_exc CONFIGURE_DEPENDS
      "${ONNXRUNTIME_ROOT}/wasm/api.cc"
      "${ONNXRUNTIME_ROOT}/core/session/onnxruntime_c_api.cc"
    )
    set (WASM_API_EXCEPTION_CATCHING "-s DISABLE_EXCEPTION_CATCHING=0")
    message(STATUS "onnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_CATCHING_ON_API set")
    set_source_files_properties(${onnxruntime_webassembly_src_exc} PROPERTIES COMPILE_FLAGS ${WASM_API_EXCEPTION_CATCHING})
  endif()

  target_link_libraries(onnxruntime_webassembly PRIVATE
    nsync::nsync_cpp
    ${PROTOBUF_LIB}
    onnx
    onnx_proto
    onnxruntime_common
    onnxruntime_flatbuffers
    onnxruntime_framework
    onnxruntime_graph
    onnxruntime_mlas
    onnxruntime_optimizer
    onnxruntime_providers
    ${PROVIDERS_JS}
    ${PROVIDERS_XNNPACK}
    ${PROVIDERS_WEBNN}
    onnxruntime_session
    onnxruntime_util
    re2::re2
  )

  set(EXPORTED_RUNTIME_METHODS "'stackAlloc','stackRestore','stackSave','UTF8ToString','stringToUTF8','lengthBytesUTF8'")

  if (onnxruntime_USE_XNNPACK)
    target_link_libraries(onnxruntime_webassembly PRIVATE XNNPACK)
    string(APPEND EXPORTED_RUNTIME_METHODS ",'addFunction'")
    target_link_options(onnxruntime_webassembly PRIVATE "SHELL:-s ALLOW_TABLE_GROWTH=1")
  endif()

  if(onnxruntime_USE_WEBNN)
    target_link_libraries(onnxruntime_webassembly PRIVATE onnxruntime_providers_webnn)
  endif()

  if (onnxruntime_ENABLE_TRAINING)
    target_link_libraries(onnxruntime_webassembly PRIVATE tensorboard)
  endif()

  if (onnxruntime_USE_JSEP)
    set(EXPORTED_FUNCTIONS "_malloc,_free,_JsepOutput,_JsepGetNodeName")
  else()
    set(EXPORTED_FUNCTIONS "_malloc,_free")
  endif()

  target_link_options(onnxruntime_webassembly PRIVATE
    "SHELL:-s EXPORTED_RUNTIME_METHODS=[${EXPORTED_RUNTIME_METHODS}]"
    "SHELL:-s EXPORTED_FUNCTIONS=${EXPORTED_FUNCTIONS}"
    "SHELL:-s MAXIMUM_MEMORY=4294967296"
    "SHELL:-s EXIT_RUNTIME=0"
    "SHELL:-s ALLOW_MEMORY_GROWTH=1"
    "SHELL:-s MODULARIZE=1"
    "SHELL:-s EXPORT_ALL=0"
    "SHELL:-s VERBOSE=0"
    "SHELL:-s FILESYSTEM=0"
    "SHELL:-s INCOMING_MODULE_JS_API=[locateFile]"
    ${WASM_API_EXCEPTION_CATCHING}
    --no-entry
  )

  if (onnxruntime_USE_JSEP)
    # NOTE: "-s ASYNCIFY=1" is required for JSEP to work with WebGPU
    #       This flag allows async functions to be called from sync functions, in the cost of binary size and
    #       build time. See https://emscripten.org/docs/porting/asyncify.html for more details.

    target_compile_definitions(onnxruntime_webassembly PRIVATE USE_JSEP=1)
    target_link_options(onnxruntime_webassembly PRIVATE
      --pre-js "${ONNXRUNTIME_ROOT}/wasm/js_internal_api.js"
      "SHELL:-s ASYNCIFY=1"
      "SHELL:-s ASYNCIFY_STACK_SIZE=65536"
    )
    set_target_properties(onnxruntime_webassembly PROPERTIES LINK_DEPENDS ${ONNXRUNTIME_ROOT}/wasm/js_internal_api.js)
  endif()

  if (onnxruntime_EMSCRIPTEN_SETTINGS)
    foreach(setting IN LISTS onnxruntime_EMSCRIPTEN_SETTINGS)
      target_link_options(onnxruntime_webassembly PRIVATE "SHELL:-s ${setting}")
    endforeach()
  endif()

  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_options(onnxruntime_webassembly PRIVATE
      "SHELL:-s ASSERTIONS=2"
      "SHELL:-s SAFE_HEAP=1"
      "SHELL:-s STACK_OVERFLOW_CHECK=2"
      "SHELL:-s DEMANGLE_SUPPORT=1"
    )
  else()
    target_link_options(onnxruntime_webassembly PRIVATE
      "SHELL:-s ASSERTIONS=0"
      "SHELL:-s SAFE_HEAP=0"
      "SHELL:-s STACK_OVERFLOW_CHECK=0"
      "SHELL:-s DEMANGLE_SUPPORT=0"
      --closure 1
    )
  endif()

  if (onnxruntime_USE_WEBNN)
   set_property(TARGET onnxruntime_webassembly APPEND_STRING PROPERTY LINK_FLAGS " --bind -sWASM_BIGINT")
  endif()

  # Set link flag to enable exceptions support, this will override default disabling exception throwing behavior when disable exceptions.
  target_link_options(onnxruntime_webassembly PRIVATE "SHELL:-s DISABLE_EXCEPTION_THROWING=0")

  if (onnxruntime_ENABLE_WEBASSEMBLY_PROFILING)
    target_link_options(onnxruntime_webassembly PRIVATE --profiling --profiling-funcs)
  endif()

  if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
    target_link_options(onnxruntime_webassembly PRIVATE
      "SHELL:-s EXPORT_NAME=ortWasmThreaded"
      "SHELL:-s DEFAULT_PTHREAD_STACK_SIZE=131072"
    )
  else()
    target_link_options(onnxruntime_webassembly PRIVATE
      "SHELL:-s EXPORT_NAME=ortWasm"
    )
  endif()

  set(target_name_list ort)

  if (onnxruntime_ENABLE_TRAINING_APIS)
    list(APPEND target_name_list  "training")
  endif()

  list(APPEND target_name_list  "wasm")

  if (onnxruntime_ENABLE_WEBASSEMBLY_SIMD)
    list(APPEND target_name_list  "simd")
  endif()

  if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
    list(APPEND target_name_list  "threaded")
  endif()

  list(JOIN target_name_list  "-" target_name)

  set_target_properties(onnxruntime_webassembly PROPERTIES OUTPUT_NAME ${target_name})
endif()
