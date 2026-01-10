// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

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
    int64_t trans_batch_a_attr, trans_batch_b_attr;
    info.GetAttrOrDefault<int64_t>("transBatchA", &trans_batch_a_attr, 0);
    info.GetAttrOrDefault<int64_t>("transBatchB", &trans_batch_b_attr, 0);
    trans_batch_a_ = trans_batch_a_attr != 0;
    trans_batch_b_ = trans_batch_b_attr != 0;

#if defined(__aarch64__) && defined(__linux__)
    auto config_ops = info.GetConfigOptions().GetConfigEntry(kOrtSessionOptionsMlasGemmFastMathArm64Bfloat16);
    use_fastmath_mode_ = (config_ops == "1") && MlasBf16AccelerationSupported();
#endif
  }

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers, int input_idx,
                                   /*out*/ bool& used_shared_buffers) override;

  Status Compute(OpKernelContext* context) const override;

 private:
  TensorShape b_shape_;
  IAllocatorUniquePtr<void> packed_b_;

  // For FusedMatMul contrib ops
  float alpha_attr_;
  int64_t trans_a_attr_;
  int64_t trans_b_attr_;
  bool trans_batch_a_;
  bool trans_batch_b_;

#if defined(__aarch64__) && defined(__linux__)
  // fastmath mode state
  bool use_fastmath_mode_;
  // sbgemm kernel is implemented as 8x8 blocks with weights pre-packed to 4 blocks of 4x2
  // so a minimum of 32 elements is defined to outweigh the additional prepacking overhead
  const size_t kFastMathModeKernelsizeThreshold = 32;
#endif
};

}  // namespace onnxruntime
