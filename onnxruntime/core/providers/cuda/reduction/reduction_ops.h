// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/reduction/reduction_ops.h"

namespace onnxruntime {
namespace cuda {

template <bool allow_multi_axes>
class ReduceKernel : public CudaKernel, public ReduceKernelBase<allow_multi_axes> {
 protected:
  ReduceKernel(
      const OpKernelInfo& info,
      std::unique_ptr<int64_t> keep_dims_override = std::unique_ptr<int64_t>{})
      : CudaKernel(info),
        ReduceKernelBase<allow_multi_axes>(info, std::move(keep_dims_override)),
        calculate_log_(false),
        calculate_sqt_(false),
        log_sum_exp_(false) {}

  // Only Max Min need to set ReduceTensorIndices CUDNN_REDUCE_TENSOR_FLATTENED_INDICES as per cudnn library manual
  // Only Max Min will have indices output, need to set the indices to nullptr for other ops
  template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices = CUDNN_REDUCE_TENSOR_NO_INDICES>
  Status ComputeImpl(OpKernelContext* ctx, cudnnReduceTensorOp_t cudnnReduceOp) const;

  template <typename T, typename OutT, cudnnReduceTensorIndices_t ReduceTensorIndices>
  Status ReduceKernelShared(
      const T* X,
      const TensorShape& input_shape,
      OutT* Y,
      const TensorShape& output_shape,
      cudnnReduceTensorOp_t cudnnReduceOp,
      std::vector<int64_t> output_dims) const;

  using ReduceKernelBase<allow_multi_axes>::axes_;
  using ReduceKernelBase<allow_multi_axes>::keepdims_;

  bool calculate_log_;
  bool calculate_sqt_;
  bool log_sum_exp_;
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
  ReduceSum(const OpKernelInfo& info) : ReduceKernel<true>(info) {}

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

// CUDA's reduction descriptor cudnnReduceTensorDescriptor_t is a pointer so
// it's safter to wrap it with automatically memory deleter as CudnnReduceDescriptor.
// An implicit caster from CudnnReduceDescriptor to cudnnReduceTensorDescriptor_t
// is implemented below, so CUDA can seamlessly work.
class CudnnReduceDescriptor final {
 public:
  CudnnReduceDescriptor() : desc_(nullptr) {
  }

  ~CudnnReduceDescriptor() {
    if (desc_ != nullptr) {
      cudnnDestroyReduceTensorDescriptor(desc_);
      desc_ = nullptr;
    }
  }

  Status Set(cudnnReduceTensorOp_t op, cudnnDataType_t type, cudnnReduceTensorIndices_t indices) {
    if (!desc_)
      CUDNN_RETURN_IF_ERROR(cudnnCreateReduceTensorDescriptor(&desc_));

    CUDNN_RETURN_IF_ERROR(cudnnSetReduceTensorDescriptor(
        desc_,
        op,
        type,
        CUDNN_PROPAGATE_NAN,
        indices,
        CUDNN_32BIT_INDICES));  // currently only the 32-bit (unsigned int) type is supported.
    return Status::OK();
  }

  operator cudnnReduceTensorDescriptor_t() const { return desc_; }

 private:
  cudnnReduceTensorDescriptor_t desc_;
};

}  // namespace cuda
}  // namespace onnxruntime
