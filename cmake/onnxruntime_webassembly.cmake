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

  add_library(${bundled_target_name} STATIC IMPORTED)
  foreach(target_name IN ITEMS ${ARGN})
    set_target_properties(${bundled_target_name}
      PROPERTIES
        IMPORTED_LOCATION ${bundled_target_full_name}
        INTERFACE_INCLUDE_DIRECTORIES $<TARGET_PROPERTY:${target_name},INTERFACE_INCLUDE_DIRECTORIES>)
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
      nsync_cpp
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
      onnxruntime_session
      onnxruntime_util
      re2::re2
    )

    if (onnxruntime_ENABLE_TRAINING OR onnxruntime_ENABLE_TRAINING_OPS)
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

  target_link_libraries(onnxruntime_webassembly PRIVATE
    nsync_cpp
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
    onnxruntime_session
    onnxruntime_util
    re2::re2
  )

  if (onnxruntime_ENABLE_TRAINING OR onnxruntime_ENABLE_TRAINING_OPS)
    target_link_libraries(onnxruntime_webassembly PRIVATE tensorboard)
  endif()

  set(EXPORTED_RUNTIME_METHODS "['stackAlloc','stackRestore','stackSave','UTF8ToString','stringToUTF8','lengthBytesUTF8']")

  set_target_properties(onnxruntime_webassembly PROPERTIES LINK_FLAGS "             \
                        -s \"EXPORTED_RUNTIME_METHODS=${EXPORTED_RUNTIME_METHODS}\" \
                        -s WASM=1                                                   \
                        -s NO_EXIT_RUNTIME=0                                        \
                        -s ALLOW_MEMORY_GROWTH=1                                    \
                        -s MODULARIZE=1                                             \
                        -s EXPORT_ALL=0                                             \
                        -s LLD_REPORT_UNDEFINED                                     \
                        -s VERBOSE=0                                                \
                        -s NO_FILESYSTEM=1                                          \
                        --closure 1                                                 \
                        --no-entry")

  if (onnxruntime_EMSCRIPTEN_SETTINGS)
    foreach(setting IN LISTS onnxruntime_EMSCRIPTEN_SETTINGS)
    set_property(TARGET onnxruntime_webassembly APPEND_STRING PROPERTY LINK_FLAGS
      " -s ${setting}")
    endforeach()
  endif()

  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_property(TARGET onnxruntime_webassembly APPEND_STRING PROPERTY LINK_FLAGS " -s ASSERTIONS=2 -s SAFE_HEAP=1 -s STACK_OVERFLOW_CHECK=1 -s DEMANGLE_SUPPORT=1")
  else()
    set_property(TARGET onnxruntime_webassembly APPEND_STRING PROPERTY LINK_FLAGS " -s ASSERTIONS=0 -s SAFE_HEAP=0 -s STACK_OVERFLOW_CHECK=0 -s DEMANGLE_SUPPORT=0")
  endif()

  # Set link flag to enable exceptions support, this will override default disabling exception throwing behavior when disable exceptions.
  if (onnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_THROWING)
    set_property(TARGET onnxruntime_webassembly APPEND_STRING PROPERTY LINK_FLAGS " -s DISABLE_EXCEPTION_THROWING=0")
  endif()

  if (onnxruntime_ENABLE_WEBASSEMBLY_PROFILING)
    set_property(TARGET onnxruntime_webassembly APPEND_STRING PROPERTY LINK_FLAGS " --profiling --profiling-funcs")
  endif()

  if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
    if (onnxruntime_ENABLE_WEBASSEMBLY_SIMD)
      set_property(TARGET onnxruntime_webassembly APPEND_STRING PROPERTY LINK_FLAGS " -s EXPORT_NAME=ortWasmSimdThreaded -s USE_PTHREADS=1")
      set_target_properties(onnxruntime_webassembly PROPERTIES OUTPUT_NAME "ort-wasm-simd-threaded")
    else()
      set_property(TARGET onnxruntime_webassembly APPEND_STRING PROPERTY LINK_FLAGS " -s EXPORT_NAME=ortWasmThreaded -s USE_PTHREADS=1")
      set_target_properties(onnxruntime_webassembly PROPERTIES OUTPUT_NAME "ort-wasm-threaded")
    endif()
  else()
    if (onnxruntime_ENABLE_WEBASSEMBLY_SIMD)
      set_property(TARGET onnxruntime_webassembly APPEND_STRING PROPERTY LINK_FLAGS " -s EXPORT_NAME=ortWasmSimd")
      set_target_properties(onnxruntime_webassembly PROPERTIES OUTPUT_NAME "ort-wasm-simd")
    else()
      set_property(TARGET onnxruntime_webassembly APPEND_STRING PROPERTY LINK_FLAGS " -s EXPORT_NAME=ortWasm")
      set_target_properties(onnxruntime_webassembly PROPERTIES OUTPUT_NAME "ort-wasm")
    endif()
  endif()
endif()
