// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/optional.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/reduction/reduction_ops.h"
#include "core/providers/cuda/reduction/reduction_functions.h"

namespace onnxruntime {
namespace cuda {

template <bool allow_multi_axes>
class ReduceKernel : public CudaKernel, public ReduceKernelBase<allow_multi_axes> {
 protected:
  ReduceKernel(
      const OpKernelInfo& info,
      optional<int64_t> keep_dims_override = {})
      : CudaKernel(info),
        ReduceKernelBase<allow_multi_axes>(info, keep_dims_override),
        calculate_log_(false),
        calculate_sqt_(false),
        log_sum_exp_(false),
        fast_reduction_(false) {}

  // Only Max Min need to set ReduceTensorIndices CUDNN_REDUCE_TENSOR_FLATTENED_INDICES as per cudnn library manual
  // Only Max Min will have indices output, need to set the indices to nullptr for other ops
  template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices = CUDNN_REDUCE_TENSOR_NO_INDICES>
  Status ComputeImpl(OpKernelContext* ctx, cudnnReduceTensorOp_t cudnn_reduce_op) const;

  template <typename T, typename OutT, cudnnReduceTensorIndices_t ReduceTensorIndices>
  Status ReduceKernelShared(
      const T* X,
      const TensorShape& input_shape,
      OutT* Y,
      const TensorShape& output_shape,
      cudnnReduceTensorOp_t cudnn_reduce_op,
      std::vector<int64_t>& output_dims) const;

  using ReduceKernelBase<allow_multi_axes>::axes_;
  using ReduceKernelBase<allow_multi_axes>::keepdims_;

  bool calculate_log_;
  bool calculate_sqt_;
  bool log_sum_exp_;
  // Indicates if this reduction can be delegated to our highly-optimized reduction kernels.
  // Those effecient kernels are defined/implemented in reduction_functions.h/.cu.
  bool fast_reduction_;
};

template <typename T>
class ArgMax final : public ReduceKernel<false> {
 public:
  ArgMax(const OpKernelInfo& info) : ReduceKernel<false>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T, CUDNN_REDUCE_TENSOR_FLATTENED_INDICES>(ctx, CUDNN_REDUCE_TENSOR_MAX);
  }
};

template <typename T>
class ArgMin final : public ReduceKernel<false> {
 public:
  ArgMin(const OpKernelInfo& info) : ReduceKernel<false>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T, CUDNN_REDUCE_TENSOR_FLATTENED_INDICES>(ctx, CUDNN_REDUCE_TENSOR_MIN);
  }
};

template <typename T>
class ReduceL1 final : public ReduceKernel<true> {
 public:
  ReduceL1(const OpKernelInfo& info) : ReduceKernel<true>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_NORM1);
  }
};

template <typename T>
class ReduceL2 final : public ReduceKernel<true> {
 public:
  ReduceL2(const OpKernelInfo& info) : ReduceKernel<true>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_NORM2);
  }
};

template <typename T>
class ReduceMax final : public ReduceKernel<true> {
 public:
  ReduceMax(const OpKernelInfo& info) : ReduceKernel<true>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_MAX);
  }
};

template <typename T>
class ReduceMean final : public ReduceKernel<true> {
 public:
  ReduceMean(const OpKernelInfo& info) : ReduceKernel<true>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_AVG);
  }
};

template <typename T>
class ReduceMin final : public ReduceKernel<true> {
 public:
  ReduceMin(const OpKernelInfo& info) : ReduceKernel<true>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_MIN);
  }
};

template <typename T>
class ReduceProd final : public ReduceKernel<true> {
 public:
  ReduceProd(const OpKernelInfo& info) : ReduceKernel<true>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_MUL);
  }
};

template <typename T>
class ReduceSum final : public ReduceKernel<true> {
 public:
  ReduceSum(const OpKernelInfo& info) : ReduceKernel<true>(info) {
    fast_reduction_ = true;
  }

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_ADD);
  }
};

template <typename T>
class ReduceLogSum final : public ReduceKernel<true> {
 public:
  ReduceLogSum(const OpKernelInfo& info) : ReduceKernel<true>(info) {
    ReduceKernel<true>::calculate_log_ = true;
  }

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_ADD);
  }
};

template <typename T>
class ReduceSumSquare final : public ReduceKernel<true> {
 public:
  ReduceSumSquare(const OpKernelInfo& info) : ReduceKernel<true>(info) {
    ReduceKernel<true>::calculate_sqt_ = true;
  }

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_ADD);
  }
};

template <typename T>
class ReduceLogSumExp final : public ReduceKernel<true> {
 public:
  ReduceLogSumExp(const OpKernelInfo& info) : ReduceKernel<true>(info) {
    ReduceKernel<true>::log_sum_exp_ = true;
  }

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, CUDNN_REDUCE_TENSOR_ADD);
  }
};

}  // namespace cuda
}  // namespace onnxruntime
