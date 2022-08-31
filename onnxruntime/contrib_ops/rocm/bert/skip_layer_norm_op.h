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
                      const int element_count)
      : OpParams(stream), output(output), input(input), skip(skip), gamma(gamma), beta(beta), bias(bias), epsilon(epsilon), ld(ld), element_count(element_count) {}

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
  const float epsilon;
  const int ld;
  const int element_count;
};

template <typename T, int ThreadsPerBlock, int VecSize>
Status SkipLayerNormSmallOp(const SkipLayerNormParams<T>* params) {
  TUNABLE_OP_RETURN_UNSUPPOTED_ARGUMENT_IF(
      !((params->ld <= 1024 && params->ld % VecSize == 0 && params->ld == ThreadsPerBlock * VecSize)));
  hipLaunchKernelGGL((SkipLayerNormKernelSmall<T, ThreadsPerBlock, VecSize>),
                     dim3(CeilingDivision(params->element_count, params->ld)),
                     dim3(ThreadsPerBlock),
                     0, params->stream, params->ld, params->input, params->skip,
                     params->beta, params->gamma, params->bias, maybe2half<T>(params->epsilon), params->output,
                     (params->bias == nullptr) ? false : true);
  auto status = hipGetLastError();
  ORT_RETURN_IF(status != hipSuccess, ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, hipGetErrorName(status)));
  return Status::OK();
}

template <typename T, int ThreadsPerBlock>
Status SkipLayerNormLargeOp(const SkipLayerNormParams<T>* params) {
  TUNABLE_OP_RETURN_UNSUPPOTED_ARGUMENT_IF(
      !((params->ld > 0)));
  hipLaunchKernelGGL((SkipLayerNormKernel<T, ThreadsPerBlock>),
                     dim3(CeilingDivision(params->element_count, params->ld)),
                     dim3(ThreadsPerBlock),
                     0, params->stream, params->ld, params->input, params->skip,
                     params->beta, params->gamma, params->bias, maybe2half<T>(params->epsilon), params->output);
  auto status = hipGetLastError();
  ORT_RETURN_IF(status != hipSuccess, ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, hipGetErrorName(status)));
  return Status::OK();
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
