// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

namespace onnxruntime {

class LayerNormImpl : public OpKernel {
 public:
  LayerNormImpl(const OpKernelInfo& op_kernel_info, bool simplified = false, bool contrib_op = false);
  Status Compute(OpKernelContext* p_op_kernel_context) const override;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override;

  // This method was created so that it can be called directly from `test/onnx/microbenchmark/layer_normalization.cc`.
  template <typename T, typename U>
  Status ComputeWithoutContext(
      const T* X_data,
      const TensorShape& x_shape,
      const T* scale_data,
      size_t scale_size,
      const T* bias_data,
      size_t bias_size,
      T* Y_data,
      U* mean_data,
      U* inv_std_dev,
      onnxruntime::concurrency::ThreadPool* thread_pool,
      int64_t axis,
      float epsilon,
      bool simplified,
      AllocatorPtr alloc) const;

 private:
  template <typename T, typename U>
  Status ComputeImpl(OpKernelContext* p_ctx, int64_t orig_axis, float epsilon, bool simplified) const;

  template <typename T>
  struct SrcDispatcher {
    Status operator()(const LayerNormImpl* p_instance, OpKernelContext* p_ctx, int64_t orig_axis,
                      float epsilon, bool simplified, bool contrib_op) const {
      // the contrib op kernel was always registered with the same type for all constraints.
      // our implementation of the onnx op only supports 'float' as the U constraint.
#if !defined(DISABLE_CONTRIB_OPS)
      if (contrib_op) {
        return p_instance->ComputeImpl<T, T>(p_ctx, orig_axis, epsilon, simplified);
      } else
#else
      ORT_UNUSED_PARAMETER(contrib_op);
#endif
      {
        return p_instance->ComputeImpl<T, float>(p_ctx, orig_axis, epsilon, simplified);
      }
    }
  };

  int64_t axis_;
  float epsilon_;
  const bool simplified_;
  const bool contrib_op_;
  IAllocatorUniquePtr<float> prepacked_scale_fp32_data_;
  size_t prepacked_scale_fp32_size_;
  IAllocatorUniquePtr<float> prepacked_bias_fp32_data_;
  size_t prepacked_bias_fp32_size_;
};

}  // namespace onnxruntime
