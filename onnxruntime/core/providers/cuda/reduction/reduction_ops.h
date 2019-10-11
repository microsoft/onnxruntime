// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/reduction/reduction_ops.h"

namespace onnxruntime {
namespace cuda {

template <bool is_arg_reduce>
class ReduceKernel : public CudaKernel, public ReduceKernelBase<is_arg_reduce> {
 protected:
  ReduceKernel(const OpKernelInfo& info) : CudaKernel(info),
                                           ReduceKernelBase<is_arg_reduce>(info) {}

  // Only Max Min need to set ReduceTensorIndices CUDNN_REDUCE_TENSOR_FLATTENED_INDICES as per cudnn library manual
  // Only Max Min will have indices output, need to set the indices to nullptr for other ops
  template <typename T>
  Status ComputeImpl(OpKernelContext* ctx, cudnnReduceTensorOp_t cudnnReduceOp) const;

  using ReduceKernelBase<is_arg_reduce>::axes_;
  using ReduceKernelBase<is_arg_reduce>::keepdims_;
};

template <typename T>
class ArgMax final : public ReduceKernel<true> {
 public:
  ArgMax(const OpKernelInfo& info) : ReduceKernel<true>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_MAX);
  }
};

template <typename T>
class ArgMin final : public ReduceKernel<true> {
 public:
  ArgMin(const OpKernelInfo& info) : ReduceKernel<true>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_MIN);
  }
};

template <typename T>
class ReduceL1 final : public ReduceKernel<false> {
 public:
  ReduceL1(const OpKernelInfo& info) : ReduceKernel<false>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_NORM1);
  }
};

template <typename T>
class ReduceL2 final : public ReduceKernel<false> {
 public:
  ReduceL2(const OpKernelInfo& info) : ReduceKernel<false>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_NORM2);
  }
};

template <typename T>
class ReduceMax final : public ReduceKernel<false> {
 public:
  ReduceMax(const OpKernelInfo& info) : ReduceKernel<false>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_MAX);
  }
};

template <typename T>
class ReduceMean final : public ReduceKernel<false> {
 public:
  ReduceMean(const OpKernelInfo& info) : ReduceKernel<false>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_AVG);
  }
};

template <typename T>
class ReduceMin final : public ReduceKernel<false> {
 public:
  ReduceMin(const OpKernelInfo& info) : ReduceKernel<false>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_MIN);
  }
};

template <typename T>
class ReduceProd final : public ReduceKernel<false> {
 public:
  ReduceProd(const OpKernelInfo& info) : ReduceKernel<false>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_MUL);
  }
};

template <typename T>
class ReduceSum final : public ReduceKernel<false> {
 public:
  ReduceSum(const OpKernelInfo& info) : ReduceKernel<false>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_ADD);
  }
};

template <typename T>
class ReduceLogSum final : public ReduceKernel<false> {
 public:
  ReduceLogSum(const OpKernelInfo& info) : ReduceKernel<false>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class ReduceSumSquare final : public ReduceKernel<false> {
 public:
  ReduceSumSquare(const OpKernelInfo& info) : ReduceKernel<false>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override;
};

template <typename T>
class ReduceLogSumExp final : public ReduceKernel<false> {
 public:
  ReduceLogSumExp(const OpKernelInfo& info) : ReduceKernel<false>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
