# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)
project(onnxruntime_wasm)

if (BUILD_NATIVE)
  message(STATUS "Building native")
else()
  message(STATUS "Building webassembly")
endif()

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dist/${CMAKE_BUILD_TYPE})
if (BUILD_NATIVE)
  string(APPEND CMAKE_CXX_FLAGS " /MP")
endif()

set(REPO_ROOT ${PROJECT_SOURCE_DIR}/../..)
set(ONNXRUNTIME_ROOT ${PROJECT_SOURCE_DIR}/..)
set(ONNXRUNTIME_INCLUDE_DIR ${REPO_ROOT}/include/onnxruntime)
set(ONNXRUNTIME_BINARY_DIR ${REPO_ROOT}/build/${CMAKE_HOST_SYSTEM_NAME}/${CMAKE_BUILD_TYPE})

# Protobuf src file reference
set(wasm_protobuf_src
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/arena.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/extension_set.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/generated_enum_util.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/generated_message_util.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/implicit_weak_message.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/message_lite.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/parse_context.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/repeated_field.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/wire_format_lite.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/io/coded_stream.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/io/io_win32.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/io/zero_copy_stream.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/io/zero_copy_stream_impl.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/io/zero_copy_stream_impl_lite.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/stubs/common.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/stubs/int128.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/stubs/status.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/stubs/stringpiece.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/stubs/stringprintf.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/stubs/structurally_valid.cc"
  "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/stubs/strutil.cc"
)

# Onnx protobuf src file reference
set(wasm_onnx_src
  "${ONNXRUNTIME_BINARY_DIR}/external/onnx/onnx/onnx-data.pb.cc"
  "${ONNXRUNTIME_BINARY_DIR}/external/onnx/onnx/onnx-ml.pb.cc"
  "${ONNXRUNTIME_BINARY_DIR}/external/onnx/onnx/onnx-operators-ml.pb.cc"
  "${REPO_ROOT}/cmake/external/onnx/onnx/checker.cc"
  "${REPO_ROOT}/cmake/external/onnx/onnx/common/assertions.cc"
  "${REPO_ROOT}/cmake/external/onnx/onnx/common/path.cc"
  "${REPO_ROOT}/cmake/external/onnx/onnx/shape_inference/implementation.cc"
)
file(GLOB_RECURSE wasm_onnx_defs_src "${REPO_ROOT}/cmake/external/onnx/onnx/defs/*.cc")

# re2 src file reference
file(GLOB wasm_re2_src "${REPO_ROOT}/cmake/external/re2/re2/*.cc")
list(APPEND wasm_re2_src "${REPO_ROOT}/cmake/external/re2/util/rune.cc"
                    "${REPO_ROOT}/cmake/external/re2/util/strutil.cc"
)

if (NOT BUILD_NATIVE)
  # mlas src file reference
  file(GLOB wasm_mlas_src CONFIGURE_DEPENDS "${ONNXRUNTIME_ROOT}/core/mlas/lib/*.cpp")
  list(APPEND wasm_mlas_src "${ONNXRUNTIME_ROOT}/core/mlas/lib/wasm/sgemmc.cpp")
endif()

file(GLOB_RECURSE wasm_contrib_ops_src CONFIGURE_DEPENDS "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/*.cc")
file(GLOB_RECURSE wasm_core_common_src CONFIGURE_DEPENDS "${ONNXRUNTIME_ROOT}/core/common/*.cc")
file(GLOB_RECURSE wasm_core_flatbuffers_src CONFIGURE_DEPENDS "${ONNXRUNTIME_ROOT}/core/flatbuffers/*.cc")
file(GLOB_RECURSE wasm_core_framework_src CONFIGURE_DEPENDS "${ONNXRUNTIME_ROOT}/core/framework/*.cc")
file(GLOB wasm_core_graph_contrib_ops_src CONFIGURE_DEPENDS "${ONNXRUNTIME_ROOT}/core/graph/contrib_ops/*.cc")
file(GLOB wasm_core_graph_src CONFIGURE_DEPENDS "${ONNXRUNTIME_ROOT}/core/graph/*.cc")
file(GLOB_RECURSE wasm_core_optimizer_src CONFIGURE_DEPENDS "${ONNXRUNTIME_ROOT}/core/optimizer/*.cc")
file(GLOB wasm_core_platform_src CONFIGURE_DEPENDS "${ONNXRUNTIME_ROOT}/core/platform/*.cc")
if (BUILD_NATIVE)
  file(GLOB wasm_core_platform_type_src CONFIGURE_DEPENDS "${ONNXRUNTIME_ROOT}/core/platform/windows/*.cc")
else()
  file(GLOB wasm_core_platform_type_src CONFIGURE_DEPENDS "${ONNXRUNTIME_ROOT}/core/platform/posix/*.cc")
endif()
file(GLOB_RECURSE wasm_core_providers_src CONFIGURE_DEPENDS "${ONNXRUNTIME_ROOT}/core/providers/cpu/*.cc")
list(APPEND wasm_core_providers_src
  "${ONNXRUNTIME_ROOT}/core/providers/get_execution_providers.cc"
)
file(GLOB_RECURSE wasm_core_session_src CONFIGURE_DEPENDS "${ONNXRUNTIME_ROOT}/core/session/*.cc")
file(GLOB_RECURSE wasm_core_util_src CONFIGURE_DEPENDS "${ONNXRUNTIME_ROOT}/core/util/*.cc")

if (BUILD_NATIVE)
  list(APPEND wasm_core_platform_type_src
    "${ONNXRUNTIME_ROOT}/core/platform/windows/telemetry.cc"
  )
  set(wasm_core_wasm_src
    "${ONNXRUNTIME_ROOT}/wasm/example_native.cc" 
  )
else()
  list(REMOVE_ITEM wasm_core_platform_type_src
    "${ONNXRUNTIME_ROOT}/core/platform/posix/ort_mutex.cc"
  )
  set(wasm_core_wasm_src
    "${ONNXRUNTIME_ROOT}/wasm/example_wasm.cc" 
  )
endif()

# compile the executables
add_executable(onnxruntime_wasm
  ${wasm_contrib_ops_src}
  ${wasm_core_common_src}
  ${wasm_core_flatbuffers_src}
  ${wasm_core_framework_src}
  ${wasm_core_graph_contrib_ops_src}
  ${wasm_core_graph_src}
  ${wasm_core_optimizer_src}
  ${wasm_core_platform_src}
  ${wasm_core_platform_type_src}
  ${wasm_core_providers_src}
  ${wasm_core_session_src}
  ${wasm_core_util_src}
  ${wasm_core_wasm_src}
  ${wasm_mlas_src}
  ${wasm_onnx_defs_src}
  ${wasm_onnx_src}
  ${wasm_protobuf_src}
  ${wasm_re2_src}
)

target_compile_definitions(onnxruntime_wasm PUBLIC
  -DENABLE_ORT_WASM
  -DONNX_API=
  -DONNX_ML=1
  -DONNX_NAMESPACE=onnx
  -DEIGEN_MPL2_ONLY
  -DMLAS_NO_ONNXRUNTIME_THREADPOOL
  -DONNX_USE_LITE_PROTO
  -DHAVE_PTHREAD
  -D_USE_MATH_DEFINES
  -D__ONNX_DISABLE_STATIC_REGISTRATION
)
if (BUILD_NATIVE)
  target_compile_definitions(onnxruntime_wasm PUBLIC
    -DPLATFORM_WINDOWS
    -DNOGDI
    -DNOMINMAX
  )
else()
  target_compile_definitions(onnxruntime_wasm PUBLIC
    -DMLAS_TARGET_WASM
  )

  target_compile_options(onnxruntime_wasm PRIVATE
    -Wno-macro-redefined
  )

  set(CMAKE_CXX_FLAGS_DEBUG "-g4 --source-map-base http://localhost:8000/")
endif()

# add all these include directory to the WebAssembly
target_include_directories(onnxruntime_wasm PRIVATE
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${ONNXRUNTIME_INCLUDE_DIR}/core/session
  ${ONNXRUNTIME_ROOT}
  ${ONNXRUNTIME_ROOT}/core/mlas/inc   # mlas.h
  ${ONNXRUNTIME_ROOT}/core/mlas/lib   # mlasi.h
  ${REPO_ROOT}/cmake/external/date/include
  ${REPO_ROOT}/cmake/external/eigen
  ${REPO_ROOT}/cmake/external/flatbuffers/include
  ${REPO_ROOT}/cmake/external/json
  ${REPO_ROOT}/cmake/external/mp11/include
  ${REPO_ROOT}/cmake/external/optional-lite/include
  ${REPO_ROOT}/cmake/external/onnx
  ${REPO_ROOT}/cmake/external/protobuf/src
  ${REPO_ROOT}/cmake/external/re2
  ${REPO_ROOT}/cmake/external/SafeInt
  ${REPO_ROOT}/cmake/external/wil/include
  PUBLIC ${ONNXRUNTIME_BINARY_DIR}
  PUBLIC ${ONNXRUNTIME_BINARY_DIR}/external/onnx
)

if (BUILD_NATIVE)
  # add the appropriate include directory and compilation flags
  target_compile_options(onnxruntime_wasm PRIVATE
    "/wd4005"
    "/wd4018"
    "/wd4100"
    "/wd4244"
    "/wd4267"
    "/wd4389"
    "/wd4456"
    "/wd4505"
    "/wd4702"
    "/wd4996"
  )

  target_link_libraries(onnxruntime_wasm PRIVATE
    onnxruntime_mlas
    advapi32
    dbghelp
  )

  enable_language(ASM_MASM)

  file(GLOB mlas_common_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/mlas/lib/*.cpp"
  )
  list(REMOVE_ITEM mlas_common_srcs "${ONNXRUNTIME_ROOT}/core/lib/dgemm.cpp")

  file(GLOB mlas_platform_srcs_avx CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/mlas/lib/intrinsics/avx/*.cpp"
  )
  set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "/arch:AVX")

  file(GLOB mlas_platform_srcs_avx2 CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/mlas/lib/intrinsics/avx2/*.cpp"
  )
  set_source_files_properties(${mlas_platform_srcs_avx2} PROPERTIES COMPILE_FLAGS "/arch:AVX2")

  file(GLOB mlas_platform_amd64_asms CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/*.asm"
  )

  set(mlas_platform_srcs
    ${ONNXRUNTIME_ROOT}/core/mlas/lib/dgemm.cpp
    ${mlas_platform_srcs_avx}
    ${mlas_platform_srcs_avx2}
    ${mlas_platform_amd64_asms}
    ${ONNXRUNTIME_ROOT}/core/mlas/lib/intrinsics/avx512/quantize_avx512f.cpp
  )

  add_library(onnxruntime_mlas STATIC ${mlas_common_srcs} ${mlas_platform_srcs})
  target_include_directories(onnxruntime_mlas PRIVATE ${ONNXRUNTIME_ROOT}/core/mlas/inc ${ONNXRUNTIME_ROOT}/core/mlas/lib)
  target_compile_options(onnxruntime_mlas PRIVATE "/wd6385")
  target_compile_definitions(onnxruntime_mlas PUBLIC -DMLAS_NO_ONNXRUNTIME_THREADPOOL)
else()
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(onnxruntime_wasm PROPERTIES LINK_FLAGS "-s DEMANGLE_SUPPORT=1 -s ASYNCIFY=1 -s EXPORT_NAME=onnxjs -s WASM=1 -s NO_EXIT_RUNTIME=0 -s ALLOW_MEMORY_GROWTH=1 -s SAFE_HEAP=0 -s MODULARIZE=1 -s SAFE_HEAP_LOG=0 -s STACK_OVERFLOW_CHECK=0 -s EXPORT_ALL=0 -s LLD_REPORT_UNDEFINED -s VERBOSE=0 -s ASSERTIONS=1 -s NO_FILESYSTEM=1 -s DISABLE_EXCEPTION_CATCHING=0 --no-entry --bind")
  else()
    set_target_properties(onnxruntime_wasm PROPERTIES LINK_FLAGS "-s DEMANGLE_SUPPORT=1 -s ASYNCIFY=1 -s EXPORT_NAME=onnxjs -s WASM=1 -s NO_EXIT_RUNTIME=0 -s ALLOW_MEMORY_GROWTH=1 -s SAFE_HEAP=0 -s MODULARIZE=1 -s SAFE_HEAP_LOG=0 -s STACK_OVERFLOW_CHECK=0 -s EXPORT_ALL=0 -s LLD_REPORT_UNDEFINED -s NO_FILESYSTEM=1 -s DISABLE_EXCEPTION_CATCHING=1 --no-entry --bind")
  endif()
endif()

