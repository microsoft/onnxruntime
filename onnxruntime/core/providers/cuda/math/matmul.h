// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/platform/env_var_utils.h"

namespace onnxruntime {
namespace cuda {
template <typename T>
class MatMul final : public CudaKernel {
  using Base = CudaKernel;

 public:
  MatMul(const OpKernelInfo& info)
      : CudaKernel(info),
        alpha_{info.GetAttrOrDefault<float>("alpha", 1.0f)},
        trans_A_{info.GetAttrOrDefault<int64_t>("transA", 0) != 0},
        trans_B_{info.GetAttrOrDefault<int64_t>("transB", 0) != 0},
        trans_batch_a_{info.GetAttrOrDefault<int64_t>("transBatchA", 0) != 0},
        trans_batch_b_{info.GetAttrOrDefault<int64_t>("transBatchB", 0) != 0} {
    if (should_use_proxy_data_ && Node().Name() == "/lm_head/MatMul") {
      cudaMalloc(&data, 768 * 50264 * 2);
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

  ~MatMul() {
    if (data != nullptr) {
      cudaFree(data);
    }
  }

 private:
  void* data = nullptr;
  const float alpha_;
  const bool trans_A_;
  const bool trans_B_;
  const bool trans_batch_a_;
  const bool trans_batch_b_;
  bool should_use_proxy_data_ = ParseEnvironmentVariableWithDefault<bool>("ORT_PROXY_DATA", false);
  bool measure_matmul_perf_ = ParseEnvironmentVariableWithDefault<bool>("ORT_MEASURE_PERF", false);
};
}  // namespace cuda
}  // namespace onnxruntime
