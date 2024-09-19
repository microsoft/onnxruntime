// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/optional.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cpu/reduction/reduction_kernel_base.h"
#include "core/providers/cuda/reduction/reduction_functions.h"

namespace onnxruntime {
namespace cuda {

namespace ReductionOps {

// Implementation that holds the core logic of reduction op processing
// `input_shape_override` is the input shape for compute purposes (if provided)

template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices = CUDNN_REDUCE_TENSOR_NO_INDICES>
std::unique_ptr<Tensor> ReduceCompute(const AllocatorPtr& gpu_allocator, cudnnReduceTensorOp_t cudnn_reduce_op, AllocatorPtr allocator,
                                      const Tensor& input, gsl::span<const int64_t> axes,
                                      bool keep_dims, bool calculate_log, bool calculate_sqt, bool log_sum_exp,
                                      bool fast_reduction, Stream* stream, const TensorShape* input_shape_override = nullptr);

}  // namespace ReductionOps

// Holds some metadata that will be used during actual reduction op compute time
struct PrepareReduceMetadata {
  int64_t input_count;
  int64_t output_count;
  // This holds the output dims without any reduced dims squeezed (even if keep_dims == 1)
  TensorShapeVector output_dims;
  // This holds the output dims with with reduced dims squeezed (if keep_dims == 1)
  TensorShapeVector squeezed_output_dims;
  TensorShapeVector input_dims_cudnn;
  TensorShapeVector output_dims_cudnn;
};

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
        fast_reduction_(false) {
    cuda_ep_ = static_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider());
  }

  // Only Max Min need to set ReduceTensorIndices CUDNN_REDUCE_TENSOR_FLATTENED_INDICES as per cudnn library manual
  // Only Max Min will have indices output, need to set the indices to nullptr for other ops
  template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices = CUDNN_REDUCE_TENSOR_NO_INDICES>
  Status ComputeImpl(OpKernelContext* ctx, cudnnReduceTensorOp_t cudnn_reduce_op) const;

  // Used by ReduceSumTraining which will have axes as input
  template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices = CUDNN_REDUCE_TENSOR_NO_INDICES>
  Status ComputeImplEx(OpKernelContext* ctx, cudnnReduceTensorOp_t cudnn_reduce_op) const;

  template <typename T, typename OutT, cudnnReduceTensorIndices_t ReduceTensorIndices>
  Status ReduceKernelShared(
      const T* X,
      const TensorShape& input_shape,
      OutT* Y,
      const TensorShape& output_shape,
      cudnnReduceTensorOp_t cudnn_reduce_op,
      cudnnHandle_t cudnn_handle,
      onnxruntime::Stream* stream,
      TensorShapeVector& output_dims) const;

  using ReduceKernelBase<allow_multi_axes>::axes_;
  using ReduceKernelBase<allow_multi_axes>::keepdims_;
  using ReduceKernelBase<allow_multi_axes>::noop_with_empty_axes_;

  bool calculate_log_;
  bool calculate_sqt_;
  bool log_sum_exp_;
  // Indicates if this reduction can be delegated to our highly-optimized reduction kernels.
  // Those efficient kernels are defined/implemented in reduction_functions.h/.cu.
  bool fast_reduction_;

  // We need to access to the CUDA EP instance to get the cudnn handle
  const CUDAExecutionProvider* cuda_ep_;
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
  ReduceMean(const OpKernelInfo& info) : ReduceKernel<true>(info) {
    fast_reduction_ = true;
  }

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
    fast_reduction_ = true;
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
    fast_reduction_ = true;
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

Status PrepareForReduce(const Tensor* X,
                        bool keepdims,
                        gsl::span<const int64_t> axes,
                        PrepareReduceMetadata& prepare_reduce_metadata,
                        const TensorShape* input_shape_override = nullptr);

template <typename T, cudnnReduceTensorIndices_t ReduceTensorIndices>
Status ReduceComputeCore(const AllocatorPtr& allocator, const Tensor& input, PrepareReduceMetadata& prepare_reduce_metadata,
                         /*out*/ Tensor& output, cudnnReduceTensorOp_t cudnn_reduce_op,
                         gsl::span<const int64_t> axes,
                         bool calculate_log, bool calculate_sqt, bool log_sum_exp, bool fast_reduction,
                         Stream* ort_stream,
                         const TensorShape* input_shape_override = nullptr);

// CUDA's reduction descriptor cudnnReduceTensorDescriptor_t is a pointer so
// it's safer to wrap it with automatically memory deleter as CudnnReduceDescriptor.
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

  CudnnReduceDescriptor(const CudnnReduceDescriptor&) = delete;
  CudnnReduceDescriptor& operator=(const CudnnReduceDescriptor&) = delete;

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
