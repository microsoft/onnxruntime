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
  using onnxruntime::rocm::CeilDiv;
  TUNABLE_OP_RETURN_UNSUPPOTED_ARGUMENT_IF(
      !((params->ld <= 1024 && params->ld % VecSize == 0 && params->ld == ThreadsPerBlock * VecSize)));
  SkipLayerNormKernelSmall<T, ThreadsPerBlock, VecSize><<<dim3(CeilDiv(params->element_count, params->ld)),
                                                          dim3(ThreadsPerBlock),
                                                          0, params->stream>>>(
      params->ld, params->input, params->skip,
      params->beta, params->gamma, params->bias, maybe2half<T>(params->epsilon), params->output,
      (params->bias == nullptr) ? false : true);
  return HIP_CALL(hipGetLastError());
}

#define ADD_OP(threads_per_block)                                         \
  this->ops_.emplace_back(SkipLayerNormSmallOp<T, threads_per_block, 1>); \
  this->ops_.emplace_back(SkipLayerNormSmallOp<T, threads_per_block, 2>); \
  this->ops_.emplace_back(SkipLayerNormSmallOp<T, threads_per_block, 4>); \
  this->ops_.emplace_back(SkipLayerNormSmallOp<T, threads_per_block, 8>); \
  this->ops_.emplace_back(SkipLayerNormSmallOp<T, threads_per_block, 16>);

template <typename T>
class SkipLayerNormTunableOp : public onnxruntime::rocm::tunable::TunableOp<SkipLayerNormParams<T>> {
 public:
  SkipLayerNormTunableOp() {
    ADD_OP(64)
    ADD_OP(128)
    ADD_OP(192)
    ADD_OP(256)
    ADD_OP(320)
    ADD_OP(384)

    // NOTE: the 3-th kernel seems to be better in gerenal case, so set it as default one
    this->SetDefaultId(3);
  }
};

#undef ADD_OP

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
