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
    if (should_use_cublas_gemm_) {
      std::cout << "using CublasGemm" << std::endl;
    } else {
      std::cout << "using CublasLtMatmul" << std::endl;
    }

    if (flush_denormals_to_zero_) {
      std::cout << "Denormals to zero" << std::endl;
   
    } else if (flush_all_to_zero_) {
      std::cout << "All to zero" << std::endl;    
    }
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  //bool flush_denormals_to_zero_ = ParseEnvironmentVariableWithDefault<bool>("ORT_FLUSH_DENORMALS", false);
  //bool flush_all_to_zero_ = ParseEnvironmentVariableWithDefault<bool>("ORT_FLUSH_ALL", false);

  bool print_latency_ = ParseEnvironmentVariableWithDefault<bool>("ORT_PRINT_LATENCY", false);
  bool randomize_weights_ = ParseEnvironmentVariableWithDefault<bool>("ORT_RANDOMIZE_WEIGHTS", false);
  bool dump_matmul_inputs_ = ParseEnvironmentVariableWithDefault<bool>("ORT_DUMP_INPUTS", false);


  const bool should_use_cublas_gemm_ = true;
  const float alpha_;
  const bool trans_A_;
  const bool trans_B_;
  const bool trans_batch_a_;
  const bool trans_batch_b_;
};
}  // namespace cuda
}  // namespace onnxruntime
