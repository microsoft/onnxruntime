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
       (params->ld < 64 && params->ld > (ThreadsPerBlock - GPU_WARP_SIZE) * VecSize)))));
  SkipLayerNormKernelVec<T, ThreadsPerBlock, VecSize><<<dim3(CeilDiv(params->element_count, params->ld)),
                                                        dim3(ThreadsPerBlock),
                                                        0, params->stream>>>(
      params->ld, params->input, params->skip,
      params->beta, params->gamma, params->bias, maybe2half<T>(params->epsilon), params->output,
      (params->bias == nullptr) ? false : true);
  return HIP_CALL(hipGetLastError());
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
    ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormSmallOp)
    ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE(SkipLayerNormRegularOp)

    // NOTE: the 30-th kernel is SkipLayerNormRegularOp ThreadsPerBlock=64 VecSize=1
    this->SetDefaultId(30);
  }
};

#undef ADD_OP_FOR_ALL_VEC_SIZE
#undef ADD_OP_FOR_ALL_THREADS_PER_BLOCK_ALL_VEC_SIZE

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
