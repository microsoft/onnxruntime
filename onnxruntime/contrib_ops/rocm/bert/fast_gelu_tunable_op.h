// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include "core/providers/rocm/tunable/rocm_tunable.h"
#include "core/providers/rocm/cu_inc/common.cuh"
#include "contrib_ops/rocm/bert/fast_gelu_impl_kernel.h"

using onnxruntime::rocm::CeilDiv;
using onnxruntime::rocm::GPU_WARP_SIZE;

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
struct FastGeluParams : OpParams {
  FastGeluParams(RocmTuningContext* tuning_ctx, hipStream_t stream, const T* input, const T* bias, T* output, int input_length, int bias_length) : OpParams(tuning_ctx, stream), input(input), bias(bias), output(output), input_length(input_length), bias_length(bias_length) {}

  std::string Signature() const override {
    std::string sig = std::to_string(input_length) + "_" + std::to_string(bias_length);
    return sig;
  }

  const T* input;
  const T* bias;
  T* output;
  int input_length;
  int bias_length;
};

template <typename T, int ThreadsPerBlock, int VecSize>
class FastGeluOp {
 public:
  Status operator()(const FastGeluParams<T>* params) {
    FastGeluKernelVec<T, ThreadsPerBlock, VecSize>
        <<<dim3(CeilDiv(params->input_length, ThreadsPerBlock * VecSize)),
           dim3(ThreadsPerBlock),
           0, params->stream>>>(
            params->input_length, params->bias_length, params->input, params->bias, params->output);
    return HIP_CALL(hipGetLastError());
  }

  Status IsSupported(const FastGeluParams<T>* params) {
    // TODO(anyone): Add tail handling for FastGelu
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
        !((params->bias_length > 0 && params->bias_length % VecSize == 0 && params->input_length % VecSize == 0) ||
          (params->bias_length == 0 && params->input_length % VecSize == 0)));
    // Avoid redundant configurations
    TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!(params->input_length > (ThreadsPerBlock - GPU_WARP_SIZE) * VecSize));

    return Status::OK();
  }
};

template <typename T>
Status FastGeluStaticSelection(const FastGeluParams<T>* params) {
  constexpr int block_size = 256;
  const int grid_size = (params->input_length + block_size - 1) / block_size;
  FastGeluKernel<T, block_size><<<dim3(grid_size), dim3(block_size), 0, params->stream>>>(
      params->input_length, params->bias_length, params->input, params->bias, params->output);
  return HIP_CALL(hipGetLastError());
}

template <>
Status FastGeluStaticSelection(const FastGeluParams<half>* params) {
  constexpr int block_size = 256;
  if (params->bias != nullptr) {
    if (0 == (params->bias_length % 8) && (params->input_length >= 3145728)) {  // 3145728=8*128*3072
      const int grid_size = (params->input_length / 8 + block_size - 1) / block_size;
      FastGeluKernelVec<half, block_size, 8><<<dim3(grid_size), dim3(block_size), 0, params->stream>>>(
          params->input_length, params->bias_length, params->input, params->bias, params->output);
    } else if (0 == (params->bias_length % 4)) {
      const int grid_size = (params->input_length / 4 + block_size - 1) / block_size;
      FastGeluKernelVec<half, block_size, 4><<<dim3(grid_size), dim3(block_size), 0, params->stream>>>(
          params->input_length, params->bias_length, params->input, params->bias, params->output);
    } else if (0 == (params->bias_length % 2)) {
      const int grid_size = (params->input_length / 2 + block_size - 1) / block_size;
      FastGeluKernelVec<half, block_size, 2><<<dim3(grid_size), dim3(block_size), 0, params->stream>>>(
          params->input_length, params->bias_length, params->input, params->bias, params->output);
    } else {
      const int grid_size = (params->input_length + block_size - 1) / block_size;
      FastGeluKernel<half, block_size><<<dim3(grid_size), dim3(block_size), 0, params->stream>>>(
          params->input_length, params->bias_length, params->input, params->bias, params->output);
    }
  } else {
    if (0 == (params->input_length % 8) && (params->input_length >= 3145728)) {  // 3145728=8*128*3072
      const int grid_size = (params->input_length / 8 + block_size - 1) / block_size;
      FastGeluKernelVec<half, block_size, 8><<<dim3(grid_size), dim3(block_size), 0, params->stream>>>(
          params->input_length, params->bias_length, params->input, params->bias, params->output);
    } else if (0 == (params->input_length % 4)) {
      const int grid_size = (params->input_length / 4 + block_size - 1) / block_size;
      FastGeluKernelVec<half, block_size, 4><<<dim3(grid_size), dim3(block_size), 0, params->stream>>>(
          params->input_length, params->bias_length, params->input, params->bias, params->output);
    } else if (0 == (params->input_length % 2)) {
      const int grid_size = (params->input_length / 2 + block_size - 1) / block_size;
      FastGeluKernelVec<half, block_size, 2><<<dim3(grid_size), dim3(block_size), 0, params->stream>>>(
          params->input_length, params->bias_length, params->input, params->bias, params->output);
    } else {
      const int grid_size = (params->input_length + block_size - 1) / block_size;
      FastGeluKernel<half, block_size><<<dim3(grid_size), dim3(block_size), 0, params->stream>>>(
          params->input_length, params->bias_length, params->input, params->bias, params->output);
    }
  }
  return HIP_CALL(hipGetLastError());
}

#define ADD_OP(threads_per_block)                          \
  this->RegisterOp(FastGeluOp<T, threads_per_block, 1>{}); \
  this->RegisterOp(FastGeluOp<T, threads_per_block, 2>{}); \
  this->RegisterOp(FastGeluOp<T, threads_per_block, 4>{}); \
  this->RegisterOp(FastGeluOp<T, threads_per_block, 8>{}); \
  this->RegisterOp(FastGeluOp<T, threads_per_block, 16>{});

template <typename T>
class FastGeluTunableOp : public TunableOp<FastGeluParams<T>> {
 public:
  FastGeluTunableOp() {
    this->RegisterOp(FastGeluStaticSelection<T>);
    ADD_OP(64);
    ADD_OP(128);
    ADD_OP(192);
    ADD_OP(256);
    ADD_OP(320);
    ADD_OP(384);
    ADD_OP(448);
    ADD_OP(512);

    // NOTE: the 1st kernel is FastGelu Original implementation.
    this->SetDefaultId(0);
  }
};

#undef ADD_OP

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
