# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB_RECURSE onnxruntime_wasm_src CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/wasm/api.cc"
)

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_wasm_src})

add_executable(onnxruntime_wasm
  ${onnxruntime_wasm_src}
)

if (NOT onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
  add_compile_definitions(
    MLAS_NO_ONNXRUNTIME_THREADPOOL
  )

  # Override re2 compiler options to remove -pthread
  set_property(TARGET re2 PROPERTY COMPILE_OPTIONS )
endif()

target_compile_options(onnx PRIVATE -Wno-unused-parameter -Wno-unused-variable)

target_link_libraries(onnxruntime_wasm PRIVATE
  nsync_cpp
  protobuf::libprotobuf-lite
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

set(EXTRA_EXPORTED_RUNTIME_METHODS "['stackAlloc','stackRestore','stackSave','UTF8ToString','stringToUTF8','lengthBytesUTF8']")

set_target_properties(onnxruntime_wasm PROPERTIES LINK_FLAGS "                                \
                      -s \"EXTRA_EXPORTED_RUNTIME_METHODS=${EXTRA_EXPORTED_RUNTIME_METHODS}\" \
                      -s WASM=1                                                               \
                      -s NO_EXIT_RUNTIME=0                                                    \
                      -s ALLOW_MEMORY_GROWTH=1                                                \
                      -s MODULARIZE=1                                                         \
                      -s EXPORT_ALL=0                                                         \
                      -s LLD_REPORT_UNDEFINED                                                 \
                      -s VERBOSE=0                                                            \
                      -s NO_FILESYSTEM=1                                                      \
                      --no-entry")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set_property(TARGET onnxruntime_wasm APPEND_STRING PROPERTY LINK_FLAGS " -s ASSERTIONS=2 -s SAFE_HEAP=1 -s STACK_OVERFLOW_CHECK=1 -s DEMANGLE_SUPPORT=1")
else()
  set_property(TARGET onnxruntime_wasm APPEND_STRING PROPERTY LINK_FLAGS " -s ASSERTIONS=0 -s SAFE_HEAP=0 -s STACK_OVERFLOW_CHECK=0 -s DEMANGLE_SUPPORT=0")
endif()

if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
  set_property(TARGET onnxruntime_wasm APPEND_STRING PROPERTY LINK_FLAGS " -s EXPORT_NAME=onnxWasmThreadsBindingJs -s USE_PTHREADS=1")
  set_target_properties(onnxruntime_wasm PROPERTIES OUTPUT_NAME "onnxruntime_wasm_threads")
else()
  set_property(TARGET onnxruntime_wasm APPEND_STRING PROPERTY LINK_FLAGS " -s EXPORT_NAME=onnxWasmBindingJs")
endif()
