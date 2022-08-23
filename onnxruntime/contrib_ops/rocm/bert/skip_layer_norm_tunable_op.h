// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include "contrib_ops/rocm/bert/skip_layer_norm_impl_kernel.h"
#include "contrib_ops/rocm/bert/tunable_op.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T>
struct SkipLayerNormParams : OpParams {
  SkipLayerNormParams(hipStream_t stream, T* output, const T* input,
                      const T* skip, const T* gamma, const T* beta,
                      const T* bias, float epsilon, const int ld,
                      const int element_count) : OpParams(stream), output(output), input(input), skip(skip), gamma(gamma), beta(beta), bias(bias), epsilon(epsilon), ld(ld), element_count(element_count) {}

  std::string Signature() const {
    std::string sig = std::to_string(ld) + "_" + std::to_string(element_count);
    return sig;
  }

  T* output;
  const T* input;
  const T* skip;
  const T* gamma;
  const T* beta;
  const T* bias;
  const float epsilon;
  const int ld;
  const int element_count;
};

template <typename T, int ThreadsPerBlock, int VecSize>
Status SkipLayerNormOp(const SkipLayerNormParams<T>* params) {
  if (params->ld <= 1024) {
    hipLaunchKernelGGL((SkipLayerNormKernelSmall<T, ThreadsPerBlock, VecSize>),
                       dim3(CeilingDivision(params->element_count, params->ld)),
                       dim3(ThreadsPerBlock),
                       0, params->stream, params->ld, params->input, params->skip,
                       params->beta, params->gamma, params->bias, maybe2half<T>(params->epsilon), params->output,
                       (params->bias == nullptr) ? false : true);
  } else {
    hipLaunchKernelGGL((SkipLayerNormKernel<T, ThreadsPerBlock>),
                       dim3(CeilingDivision(params->element_count, params->ld)),
                       dim3(ThreadsPerBlock),
                       0, params->stream, params->ld, params->input, params->skip,
                       params->beta, params->gamma, params->bias, maybe2half<T>(params->epsilon), params->output);
  }
  auto status = hipGetLastError();
  ORT_RETURN_IF(status != hipSuccess, ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, hipGetErrorName(status)));
  return Status::OK();
}

#define ADD_OP(threads_per_block)                                    \
  this->ops_.emplace_back(SkipLayerNormOp<T, threads_per_block, 1>); \
  this->ops_.emplace_back(SkipLayerNormOp<T, threads_per_block, 2>); \
  this->ops_.emplace_back(SkipLayerNormOp<T, threads_per_block, 4>); \
  this->ops_.emplace_back(SkipLayerNormOp<T, threads_per_block, 8>); \
  this->ops_.emplace_back(SkipLayerNormOp<T, threads_per_block, 16>);

template <typename T>
class SkipLayerNormTunableOp : public TunableOp<SkipLayerNormParams<T>> {
 public:
  SkipLayerNormTunableOp() {
    ADD_OP(32);
    ADD_OP(64);
    ADD_OP(128);
    ADD_OP(192);
    ADD_OP(256);
    ADD_OP(320);
    ADD_OP(384);
    ADD_OP(448);
    ADD_OP(512);
  }

 private:
  bool Condition(const SkipLayerNormParams<T>* skip_layer_norm_params) override {
    bool condition = (skip_layer_norm_params->ld > 0) && (skip_layer_norm_params->ld % 16 == 0);
    return condition;
  }
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
