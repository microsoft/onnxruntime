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

# set WebAssembly root directory

#file(GLOB_RECURSE WASM_PROVIDERS_SRC CONFIGURE_DEPENDS
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/activation/*.h"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/activation/*.cc"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/generator/*.h"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/generator/*.cc"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/math/*.h"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/math/*.cc"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/nn/*.h"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/nn/*.cc"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/object_detection/*.h"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/object_detection/*.cc"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/reduction/*.h"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/reduction/*.cc"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/sequence/*.h"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/sequence/*.cc"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/*.h"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/*.cc"
#)
# Mlas dependency
#list(REMOVE_ITEM WASM_PROVIDERS_SRC
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/math/matmul_integer.cc"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/math/quantize_linear_matmul.cc"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/nn/conv_integer.cc"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/nn/qlinearconv.cc"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/cast_op.cc"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/dynamicquantizelinear.cc"
#  "${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/quantize_linear.cc"
#)

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
             "${ONNXRUNTIME_ROOT}/core/framework/kernel_def_builder.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/kernel_registry.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/kernel_registry_manager.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/mldata_type_utils.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/murmurhash3.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/node_index_info.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/ort_value_pattern_planner.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/op_kernel.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/op_kernel_info.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/op_node_proto_helper.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/random_seed.cc"
             "${ONNXRUNTIME_ROOT}/core/framework/random_generator.cc"
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
             #"${ONNXRUNTIME_ROOT}/core/framework/utils.cc"
             #"${ONNXRUNTIME_ROOT}/core/framework/wasm_kernel_registry_manager.cc"
             "${ONNXRUNTIME_ROOT}/core/graph/function.cc"
             "${ONNXRUNTIME_ROOT}/core/graph/graph.cc"
             "${ONNXRUNTIME_ROOT}/core/graph/graph_viewer.cc"
             "${ONNXRUNTIME_ROOT}/core/graph/model.cc"
             "${ONNXRUNTIME_ROOT}/core/graph/schema_registry.cc"
             #"${ONNXRUNTIME_ROOT}/core/mlas/lib/activate.cpp"
             #"${ONNXRUNTIME_ROOT}/core/mlas/lib/compute.cpp"
             #"${ONNXRUNTIME_ROOT}/core/mlas/lib/convolve.cpp"
             #"${ONNXRUNTIME_ROOT}/core/mlas/lib/erf.cpp"
             #"${ONNXRUNTIME_ROOT}/core/mlas/lib/logistic.cpp"
             #"${ONNXRUNTIME_ROOT}/core/mlas/lib/platform.cpp"
             #"${ONNXRUNTIME_ROOT}/core/mlas/lib/sgemm.cpp"
             #"${ONNXRUNTIME_ROOT}/core/mlas/lib/wasm/sgemmc.cpp"
             #"${ONNXRUNTIME_ROOT}/core/mlas/lib/tanh.cpp"
             "${ONNXRUNTIME_ROOT}/core/platform/posix/env.cc"
             "${ONNXRUNTIME_ROOT}/core/platform/posix/env_time.cc"
             #"${ONNXRUNTIME_ROOT}/core/platform/posix/ort_mutex.cc"
             "${ONNXRUNTIME_ROOT}/core/platform/posix/stacktrace.cc"
             "${ONNXRUNTIME_ROOT}/core/platform/env.cc"
             "${ONNXRUNTIME_ROOT}/core/platform/env_time.cc"
             "${ONNXRUNTIME_ROOT}/core/platform/telemetry.cc"
             "${ONNXRUNTIME_ROOT}/core/platform/path_lib.cc"
             #"${ONNXRUNTIME_ROOT}/core/platform/windows/env.cc"
             #"${ONNXRUNTIME_ROOT}/core/platform/windows/env_time.cc"
             #"${ONNXRUNTIME_ROOT}/core/platform/windows/telemetry.cc"
             #"${ONNXRUNTIME_ROOT}/core/platform/windows/stacktrace.cc"
             #"${ONNXRUNTIME_ROOT}/core/platform/windows/telemetry.cc"
             "${ONNXRUNTIME_ROOT}/core/providers/cpu/wasm_execution_provider.cc"
             #"${ONNXRUNTIME_ROOT}/core/providers/cpu/activation/activations.cc"
             #"${ONNXRUNTIME_ROOT}/core/providers/cpu/math/gemm.cc"
             #"${ONNXRUNTIME_ROOT}/core/providers/cpu/math/element_wise_ops.cc"
             #"${ONNXRUNTIME_ROOT}/core/providers/cpu/math/matmul.cc"
             #"${ONNXRUNTIME_ROOT}/core/providers/cpu/nn/conv.cc"
             "${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/concat.cc"
             #"${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/gather.cc"
             #"${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/reshape.cc"
             #"${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/slice.cc"
             #"${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/unsqueeze.cc"
             #"${ONNXRUNTIME_ROOT}/core/providers/shared_library/provider_bridge_provider.cc"
             #"${ONNXRUNTIME_ROOT}/core/util/math_cpu.cc"
             "${ONNXRUNTIME_ROOT}/wasm/example.cc" 
             "${ONNXRUNTIME_ROOT}/wasm/utils.cc"
)

# compile the executables
add_executable(onnxruntime_wasm
  ${ONNX_DEFS_SRC}
  ${ONNX_SRC}
  ${PROTOBUF_SRC}
  ${WASM_PROVIDERS_SRC}
  ${WASM_SRC}
)
        
# compile option with __wasm__
target_compile_definitions(onnxruntime_wasm PUBLIC
  -DENABLE_ORT_WASM
  -DDISABLE_ML_OPS
  -DONNX_ML=1
  -DONNX_NAMESPACE=onnx
  -DEIGEN_MPL2_ONLY
  #-DORT_MINIMAL_BUILD
  #-DORT_EXTENDED_MINIMAL_BUILD
  -DMLAS_NO_ONNXRUNTIME_THREADPOOL
  -DMLAS_TARGET_WASM
  -DONNX_USE_LITE_PROTO
  -DHAVE_PTHREAD
)

add_definitions("-pthread")
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
  ${REPO_ROOT}/cmake/external/SafeInt
  ${REPO_ROOT}/cmake/external/wil/include
  PUBLIC ${ONNXRUNTIME_BINARY_DIR}
  PUBLIC ${ONNXRUNTIME_BINARY_DIR}/external/onnx
)

# add the appropriate include directory and compilation flags
#target_compile_options(onnxruntime_wasm PRIVATE
#  "/wd4018"
#  "/wd4100"
#  "/wd4244"
#  "/wd4267"
#  "/wd4389"
#  "/wd4456"
#  "/wd4505"
#  "/wd4702"
#  "/wd4996"
#)

#if (WIN32)
#  target_link_libraries(onnxruntime_wasm PRIVATE debug dbghelp advapi32)
#endif()

set_target_properties(onnxruntime_wasm PROPERTIES LINK_FLAGS "-s DEMANGLE_SUPPORT=1 -s LLD_REPORT_UNDEFINED -o out_wasm_main.js --no-entry --bind")


