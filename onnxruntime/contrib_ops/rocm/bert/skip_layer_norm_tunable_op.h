// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "contrib_ops/rocm/bert/skip_layer_norm_impl_kernel.h"
#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/tunable/tunable.h"

using onnxruntime::rocm::CeilDiv;

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
struct SkipLayerNormParams : onnxruntime::rocm::tunable::OpParams {
  SkipLayerNormParams(hipStream_t stream, T* output, const T* input,
                      const T* skip, const T* gamma, const T* beta,
                      const T* bias, float epsilon, int ld, int element_count)
      : OpParams(stream), output(output), input(input), skip(skip), gamma(gamma), beta(beta), bias(bias),
        epsilon(epsilon), ld(ld), element_count(element_count) {}

  std::string Signature() const override {
    std::string sig = std::to_string(ld) + "_" + std::to_string(element_count);
    return sig;
  }

  T* output;
  const T* input;
  const T* skip;
  const T* gamma;
  const T* beta;
  const T* bias;
  float epsilon;
  int ld;
  int element_count;
};

template <typename T, int ThreadsPerBlock, int VecSize>
Status SkipLayerNormSmallOp(const SkipLayerNormParams<T>* params) {
  TUNABLE_OP_RETURN_UNSUPPOTED_ARGUMENT_IF(
      !((params->ld <= 1024 && params->ld % VecSize == 0 &&
         params->ld <= ThreadsPerBlock * VecSize && params->ld > (ThreadsPerBlock - GPU_WARP_SIZE) * VecSize)));
  SkipLayerNormKernelSmall<T, ThreadsPerBlock, VecSize><<<dim3(CeilDiv(params->element_count, params->ld)),
                                                          dim3(ThreadsPerBlock),
                                                          0, params->stream>>>(
      params->ld, params->input, params->skip,
      params->beta, params->gamma, params->bias, maybe2half<T>(params->epsilon), params->output,
      (params->bias == nullptr) ? false : true);
  return HIP_CALL(hipGetLastError());
}

template <typename T, int ThreadsPerBlock, int VecSize>
Status SkipLayerNormRegularOp(const SkipLayerNormParams<T>* params) {
  TUNABLE_OP_RETURN_UNSUPPOTED_ARGUMENT_IF(
      !((params->ld > 0 && params->ld % VecSize == 0 &&
         (params->ld >= ThreadsPerBlock * VecSize ||
          (params->ld < GPU_WARP_SIZE && params->ld > (ThreadsPerBlock - GPU_WARP_SIZE) * VecSize)))));
  SkipLayerNormKernelVec<T, ThreadsPerBlock, VecSize><<<dim3(CeilDiv(params->element_count, params->ld)),
                                                        dim3(ThreadsPerBlock),
                                                        0, params->stream>>>(
      params->ld, params->input, params->skip,
      params->beta, params->gamma, params->bias, maybe2half<T>(params->epsilon), params->output,
      (params->bias == nullptr) ? false : true);
  return HIP_CALL(hipGetLastError());
}

template <typename T>
Status SkipLayerNormStaticSelection(const SkipLayerNormParams<T>* params) {
  bool hasBias = (params->bias == nullptr) ? false : true;
  if (0 == (params->ld % 4)) {
    const int grid_size = params->element_count / params->ld;
    if (params->ld <= 32) {
      constexpr int block_size = 32;
      SkipLayerNormKernelSmall<T, block_size, 1><<<grid_size, block_size, 0, params->stream>>>(
          params->ld, params->input, params->skip, params->beta, params->gamma, params->bias,
          maybe2half<T>(params->epsilon), params->output, hasBias);
    } else if (params->ld <= 64) {
      constexpr int block_size = 64 / 2;
      SkipLayerNormKernelSmall<T, block_size, 2><<<grid_size, block_size, 0, params->stream>>>(
          params->ld, params->input, params->skip, params->beta, params->gamma, params->bias,
          maybe2half<T>(params->epsilon), params->output, hasBias);
    } else if (params->ld <= 128) {
      constexpr int block_size = 128 / 4;
      SkipLayerNormKernelSmall<T, block_size, 4><<<grid_size, block_size, 0, params->stream>>>(
          params->ld, params->input, params->skip, params->beta, params->gamma, params->bias,
          maybe2half<T>(params->epsilon), params->output, hasBias);
    } else if (params->ld <= 384) {
      constexpr int block_size = 384 / 4;
      SkipLayerNormKernelSmall<T, block_size, 4><<<grid_size, block_size, 0, params->stream>>>(
          params->ld, params->input, params->skip, params->beta, params->gamma, params->bias,
          maybe2half<T>(params->epsilon), params->output, hasBias);
    } else if (params->ld <= 768) {
      constexpr int block_size = 768 / 4;
      SkipLayerNormKernelSmall<T, block_size, 4><<<grid_size, block_size, 0, params->stream>>>(
          params->ld, params->input, params->skip, params->beta, params->gamma, params->bias,
          maybe2half<T>(params->epsilon), params->output, hasBias);
    } else if (params->ld <= 1024) {
      constexpr int block_size = 1024 / 4;
      SkipLayerNormKernelSmall<T, block_size, 4><<<grid_size, block_size, 0, params->stream>>>(
          params->ld, params->input, params->skip, params->beta, params->gamma, params->bias,
          maybe2half<T>(params->epsilon), params->output, hasBias);
    } else {
      constexpr int block_size = 256;
      SkipLayerNormKernel<T, block_size><<<grid_size, block_size, 0, params->stream>>>(
          params->ld, params->input, params->skip, params->beta, params->gamma, params->bias,
          maybe2half<T>(params->epsilon), params->output);
    }
  } else {
    const int grid_size = params->element_count / params->ld;
    if (params->ld <= 32) {
      constexpr int block_size = 32;
      SkipLayerNormKernelSmall<T, block_size, 1><<<grid_size, block_size, 0, params->stream>>>(
          params->ld, params->input, params->skip, params->beta, params->gamma, params->bias,
          maybe2half<T>(params->epsilon), params->output, hasBias);
    } else if (params->ld <= 64) {
      constexpr int block_size = 64;
      SkipLayerNormKernelSmall<T, block_size, 1><<<grid_size, block_size, 0, params->stream>>>(
          params->ld, params->input, params->skip, params->beta, params->gamma, params->bias,
          maybe2half<T>(params->epsilon), params->output, hasBias);
    } else if (params->ld <= 128) {
      constexpr int block_size = 128;
      SkipLayerNormKernelSmall<T, block_size, 1><<<grid_size, block_size, 0, params->stream>>>(
          params->ld, params->input, params->skip, params->beta, params->gamma, params->bias,
          maybe2half<T>(params->epsilon), params->output, hasBias);
    } else if (params->ld == 384) {
      constexpr int block_size = 384;
      SkipLayerNormKernelSmall<T, block_size, 1><<<grid_size, block_size, 0, params->stream>>>(
          params->ld, params->input, params->skip, params->beta, params->gamma, params->bias,
          maybe2half<T>(params->epsilon), params->output, hasBias);
    } else {
      constexpr int block_size = 256;
      SkipLayerNormKernel<T, block_size><<<grid_size, block_size, 0, params->stream>>>(
          params->ld, params->input, params->skip, params->beta, params->gamma, params->bias,
          maybe2half<T>(params->epsilon), params->output);
    }
  }
  return HIP_CALL(hipPeekAtLastError());
}

#define ADD_OP_FOR_ALL_VEC_SIZE(name, threads_per_block)  \
  this->ops_.emplace_back(name<T, threads_per_block, 1>); \
  this->ops_.emplace_back(name<T, threads_per_block, 2>); \
  this->ops_.emplace_back(name<T, threads_per_block, 4>); \
  this->ops_.emplace_back(name<T, threads_per_block, 8>); \
  this->ops_.emplace_back(name<T, threads_per_block, 16>);

#define ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(name) \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 64)                         \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 128)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 192)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 256)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 320)                        \
  ADD_OP_FOR_ALL_VEC_SIZE(name, 384)

template <typename T>
class SkipLayerNormTunableOp : public onnxruntime::rocm::tunable::TunableOp<SkipLayerNormParams<T>> {
 public:
  SkipLayerNormTunableOp() {
    this->ops_.emplace_back(SkipLayerNormStaticSelection<T>);
    ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormSmallOp)
    ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormRegularOp)

    // NOTE: the 1st kernel is SkipLayerNorm Original implementation.
    this->SetDefaultId(0);
  }
};

#undef ADD_OP_FOR_ALL_VEC_SIZE
#undef ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
