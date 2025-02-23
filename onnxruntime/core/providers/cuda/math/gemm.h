// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

#ifndef DISABLE_CONTRIB_OPS
#include "contrib_ops/cuda/llm/gemm_runner.h"
#endif

namespace onnxruntime {
namespace cuda {
template <typename T>
class Gemm final : public CudaKernel {
  using Base = CudaKernel;

 public:
  Gemm(const OpKernelInfo& info) : CudaKernel(info) {
    int64_t temp;
    ORT_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    trans_A_ = (temp != 0);

    ORT_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    trans_B_ = (temp != 0);

    ORT_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK());
  }

  Status ComputeInternal(OpKernelContext* context) const override;
  Status ComputeDefault(OpKernelContext* context, int M, int N, int K) const;

 private:
  bool trans_A_;
  bool trans_B_;
  float alpha_;
  float beta_;

#ifndef DISABLE_CONTRIB_OPS
  mutable std::unique_ptr<llm::IGemmRunner> gemm_runner_;
  mutable std::mutex gemm_runner_mutex_;
#endif
};
}  // namespace cuda
}  // namespace onnxruntime
