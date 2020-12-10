// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

template <typename T>
class MatMul final : public OpKernel {
 public:
  MatMul(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override;
};

template <>
class MatMul<float> final : public OpKernel {
 public:
  MatMul(const OpKernelInfo& info) : OpKernel(info) {
    info.GetAttrOrDefault<int64_t>("transA", &trans_a_attr_, 0);
    info.GetAttrOrDefault<int64_t>("transB", &trans_b_attr_, 0);
    info.GetAttrOrDefault<float>("alpha", &alpha_attr_, 1.0);
  }

#if !defined(USE_MKLML_FOR_BLAS)
  Status PrePack(const Tensor& tensor, int input_idx, bool& is_packed) override;
#endif

  Status Compute(OpKernelContext* context) const override;

 private:
  TensorShape b_shape_;
  BufferUniquePtr packed_b_;

  // For FusedMatMul and TransposeMatMul contrib ops
  float alpha_attr_;
  int64_t trans_a_attr_;
  int64_t trans_b_attr_;
};

}  // namespace onnxruntime
