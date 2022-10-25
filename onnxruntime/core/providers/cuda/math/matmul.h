// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

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
          // cudaMalloc(&left_X_ptr_, (size_t)(ceil(6291456/ 256.)) * 256);

          cudaMalloc(&left_X_ptr_, (size_t)(ceil(6291456/ 256.)) * 256);
          //cudaMalloc(&right_X_ptr_1_, (size_t)(ceil(1179648/ 256.)) * 256);
          //cudaMalloc(&right_X_ptr_2_, (size_t)(ceil(4718592/ 256.)) * 256);


        }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  const float alpha_;
  const bool trans_A_;
  const bool trans_B_;
  const bool trans_batch_a_;
  const bool trans_batch_b_;
  void* left_X_ptr_ = nullptr;
  //void* right_X_ptr_1_ = nullptr;
  //void* right_X_ptr_2_ = nullptr;

};

}  // namespace cuda
}  // namespace onnxruntime
