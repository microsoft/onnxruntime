# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.13)
project(onnxruntime_wasm)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/dist/${CMAKE_BUILD_TYPE})

set(REPO_ROOT ${PROJECT_SOURCE_DIR}/../..)
set(ONNXRUNTIME_ROOT ${PROJECT_SOURCE_DIR}/..)
set(ONNXRUNTIME_INCLUDE_DIR ${REPO_ROOT}/include/onnxruntime)
set(ONNXRUNTIME_BINARY_DIR ${REPO_ROOT}/build/${CMAKE_HOST_SYSTEM_NAME}/${CMAKE_BUILD_TYPE})

# add WebAssbmly Build Configuration 
message(STATUS "Building webassembly")

file(GLOB_RECURSE WASM_PROVIDERS_SRC CONFIGURE_DEPENDS "${ONNXRUNTIME_ROOT}/core/providers/cpu/*.cc")
list(REMOVE_ITEM WASM_PROVIDERS_SRC
  "${ONNXRUNTIME_ROOT}/core/providers/cpu/cpu_provider_factory.cc"
  "${ONNXRUNTIME_ROOT}/core/providers/cpu/wasm_execution_provider.cc"
)

file(GLOB MLAS_SRC CONFIGURE_DEPENDS "${ONNXRUNTIME_ROOT}/core/mlas/lib/*.cpp")
list(APPEND MLAS_SRC "${ONNXRUNTIME_ROOT}/core/mlas/lib/wasm/sgemmc.cpp")

file(GLOB_RECURSE CONTRIB_OPS_SRC CONFIGURE_DEPENDS "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/*.cc")

# Protobuf src file reference
set(PROTOBUF_SRC "${REPO_ROOT}/cmake/external/protobuf/src/google/protobuf/arena.cc"
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
set(ONNX_SRC "${ONNXRUNTIME_BINARY_DIR}/onnx/onnx-data.pb.cc"
             "${ONNXRUNTIME_BINARY_DIR}/onnx/onnx-ml.pb.cc"
             "${ONNXRUNTIME_BINARY_DIR}/onnx/onnx-operators-ml.pb.cc"
             "${REPO_ROOT}/cmake/external/onnx/onnx/checker.cc"
             "${REPO_ROOT}/cmake/external/onnx/onnx/common/assertions.cc"
             "${REPO_ROOT}/cmake/external/onnx/onnx/common/path.cc"
             "${REPO_ROOT}/cmake/external/onnx/onnx/shape_inference/implementation.cc"
)
file(GLOB_RECURSE ONNX_DEFS_SRC "${REPO_ROOT}/cmake/external/onnx/onnx/defs/*.cc")

# re2 src file reference
file(GLOB RE2_SRC "${REPO_ROOT}/cmake/external/re2/re2/*.cc")
list(APPEND RE2_SRC "${REPO_ROOT}/cmake/external/re2/util/rune.cc"
                    "${REPO_ROOT}/cmake/external/re2/util/strutil.cc"
)

# WebAssembly src file reference 
set(WASM_SRC "${ONNXRUNTIME_ROOT}/core/common/cpuid_info.cc"
             "${ONNXRUNTIME_ROOT}/core/common/denormal.cc"
             "${ONNXRUNTIME_ROOT}/core/common/logging/capture.cc"
             "${ONNXRUNTIME_ROOT}/core/common/logging/logging.cc"
             "${ONNXRUNTIME_ROOT}/core/common/path.cc"
             "${ONNXRUNTIME_ROOT}/core/common/profiler.cc"
             "${ONNXRUNTIME_ROOT}/core/common/status.cc"
             "${ONNXRUNTIME_ROOT}/core/common/str_helper.cc"
             "${ONNXRUNTIME_ROOT}/core/common/threadpool.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/allocator.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/allocation_planner.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/allocatormgr.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/bfc_arena.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/data_transfer.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/data_transfer_manager.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/data_types.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/endian_utils.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/error_code.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/ex_lib_loader.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/execution_frame.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/execution_provider.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/feeds_fetches_manager.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/kernel_def_builder.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/kernel_registry.cc"
             #"${ONNXRUNTIME_ROOT}/core/framework/kernel_registry_manager.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/mldata_type_utils.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/murmurhash3.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/node_index_info.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/ort_value_pattern_planner.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/op_kernel.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/op_kernel_info.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/op_node_proto_helper.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/ort_value_tensor_slicer.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/parallel_executor.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/random_seed.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/random_generator.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/sequential_executor.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/session_options.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/session_state.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/session_state_utils.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/simple_tensor_allocator.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/sparse_tensor.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/tensor.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/tensor_allocator.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/tensor_external_data_info.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/tensor_shape.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/tensorprotoutils.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/utils.cc"
             "${ONNXRUNTIME_ROOT}/core/graph/function.cc"
             "${ONNXRUNTIME_ROOT}/core/graph/graph.cc"
             "${ONNXRUNTIME_ROOT}/core/graph/graph_viewer.cc"
             "${ONNXRUNTIME_ROOT}/core/graph/model.cc"
             "${ONNXRUNTIME_ROOT}/core/graph/schema_registry.cc"
             "${ONNXRUNTIME_ROOT}/core/optimizer/optimizer_execution_frame.cc"
             "${ONNXRUNTIME_ROOT}/core/platform/posix/env.cc"
             "${ONNXRUNTIME_ROOT}/core/platform/posix/env_time.cc"
             "${ONNXRUNTIME_ROOT}/core/platform/posix/stacktrace.cc"
             "${ONNXRUNTIME_ROOT}/core/platform/env.cc"
             "${ONNXRUNTIME_ROOT}/core/platform/env_time.cc"
             "${ONNXRUNTIME_ROOT}/core/platform/telemetry.cc"
             "${ONNXRUNTIME_ROOT}/core/platform/path_lib.cc"
             "${ONNXRUNTIME_ROOT}/core/util/math_cpu.cc"
             "${ONNXRUNTIME_ROOT}/wasm/example.cc" 
)

# compile the executables
add_executable(onnxruntime_wasm
  ${CONTRIB_OPS_SRC}
  ${MLAS_SRC}
  ${ONNX_DEFS_SRC}
  ${ONNX_SRC}
  ${PROTOBUF_SRC}
  ${RE2_SRC}
  ${WASM_PROVIDERS_SRC}
  ${WASM_SRC}
)
        
target_compile_definitions(onnxruntime_wasm PUBLIC
  -DENABLE_ORT_WASM
  -DDISABLE_ML_OPS
  -DONNX_ML=1
  -DONNX_NAMESPACE=onnx
  -DEIGEN_MPL2_ONLY
  -DMLAS_NO_ONNXRUNTIME_THREADPOOL
  -DMLAS_TARGET_WASM
  -DONNX_USE_LITE_PROTO
  -DHAVE_PTHREAD
)

SET(CMAKE_CXX_FLAGS_DEBUG "-g4")

remove_definitions(ENABLE_ORT_FORMAT_LOAD)

# add all these include directory to the WebAssembly
target_include_directories(onnxruntime_wasm
  PRIVATE ${ONNXRUNTIME_INCLUDE_DIR}
  ${ONNXRUNTIME_ROOT}
  ${ONNXRUNTIME_ROOT}/core/mlas/inc   # mlas.h
  ${ONNXRUNTIME_ROOT}/core/mlas/lib   # mlasi.h
  ${REPO_ROOT}/cmake/external/eigen
  ${REPO_ROOT}/cmake/external/flatbuffers/include
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

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set_target_properties(onnxruntime_wasm PROPERTIES LINK_FLAGS "-s DEMANGLE_SUPPORT=1 -s ASYNCIFY=1 -s EXPORT_NAME=onnxjs -s WASM=1 -s NO_EXIT_RUNTIME=0 -s ALLOW_MEMORY_GROWTH=1 -s SAFE_HEAP=0 -s MODULARIZE=1 -s SAFE_HEAP_LOG=0 -s STACK_OVERFLOW_CHECK=0 -s EXPORT_ALL=0 -s LLD_REPORT_UNDEFINED -s VERBOSE=0 -s ASSERTIONS=1 -s NO_FILESYSTEM=1 -s DISABLE_EXCEPTION_CATCHING=1 --source-map-base http://localhost:8000/ --no-entry --bind")
else()
  set_target_properties(onnxruntime_wasm PROPERTIES LINK_FLAGS "-s DEMANGLE_SUPPORT=1 -s ASYNCIFY=1 -s EXPORT_NAME=onnxjs -s WASM=1 -s NO_EXIT_RUNTIME=0 -s ALLOW_MEMORY_GROWTH=1 -s SAFE_HEAP=0 -s MODULARIZE=1 -s SAFE_HEAP_LOG=0 -s STACK_OVERFLOW_CHECK=0 -s EXPORT_ALL=0 -s LLD_REPORT_UNDEFINED -s NO_FILESYSTEM=1 -s DISABLE_EXCEPTION_CATCHING=1 --no-entry --bind")
endif()

