// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/optional.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/cpu/reduction/reduction_ops.h"
#include "core/providers/rocm/reduction/reduction_functions.h"

namespace onnxruntime {
namespace rocm {

namespace ReductionOps {

// Implementation that holds the core logic of reduction op processing
// `input_shape_override` is the input shape for compute purposes (if provided)

template <typename T, miopenReduceTensorIndices_t ReduceTensorIndices = MIOPEN_REDUCE_TENSOR_NO_INDICES>
Tensor ReduceCompute(ROCMExecutionProvider& rocm_ep, miopenReduceTensorOp_t miopen_reduce_op, AllocatorPtr allocator,
                     const Tensor& input, const std::vector<int64_t>& axes,
                     bool keep_dims, bool calculate_log, bool calculate_sqt, bool log_sum_exp,
                     bool fast_reduction, const TensorShape* input_shape_override = nullptr);

}  // namespace ReductionOps

// Holds some metadata that will be used during actual reduction op compute time
struct PrepareReduceMetadata {
  int64_t input_count;
  int64_t output_count;
  // This holds the output dims without any reduced dims squeezed (even if keep_dims == 1)
  std::vector<int64_t> output_dims;
  // This holds the output dims with with reduced dims squeezed (if keep_dims == 1)
  std::vector<int64_t> squeezed_output_dims;
  std::vector<int64_t> input_dims_miopen;
  std::vector<int64_t> output_dims_miopen;
};

template <bool allow_multi_axes>
class ReduceKernel : public RocmKernel, public ReduceKernelBase<allow_multi_axes> {
 protected:
  ReduceKernel(
      const OpKernelInfo& info,
      optional<int64_t> keep_dims_override = {})
      : RocmKernel(info),
        ReduceKernelBase<allow_multi_axes>(info, keep_dims_override),
        calculate_log_(false),
        calculate_sqt_(false),
        log_sum_exp_(false),
        fast_reduction_(false) {
    // We need to cast away the const as PerThreadMiopenHandle() is currently a non-const method
    // TODO: Clean up the ROCMExecutionProvider interface to avoid this
    rocm_ep_ = const_cast<ROCMExecutionProvider*>(static_cast<const ROCMExecutionProvider*>(info.GetExecutionProvider()));
  }

  // Only Max Min need to set ReduceTensorIndices MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES as per miopen library manual
  // Only Max Min will have indices output, need to set the indices to nullptr for other ops
  template <typename T, miopenReduceTensorIndices_t ReduceTensorIndices = MIOPEN_REDUCE_TENSOR_NO_INDICES>
  Status ComputeImpl(OpKernelContext* ctx, miopenReduceTensorOp_t miopen_reduce_op) const;

  // Used by ReduceSumTraining which will have axes as input
  template <typename T, miopenReduceTensorIndices_t ReduceTensorIndices = MIOPEN_REDUCE_TENSOR_NO_INDICES>
  Status ComputeImplEx(OpKernelContext* ctx, miopenReduceTensorOp_t miopen_reduce_op) const;

  template <typename T, typename OutT, miopenReduceTensorIndices_t ReduceTensorIndices>
  Status ReduceKernelShared(
      const T* X,
      const TensorShape& input_shape,
      OutT* Y,
      const TensorShape& output_shape,
      miopenReduceTensorOp_t miopen_reduce_op,
      std::vector<int64_t>& output_dims) const;

  using ReduceKernelBase<allow_multi_axes>::axes_;
  using ReduceKernelBase<allow_multi_axes>::keepdims_;
  using ReduceKernelBase<allow_multi_axes>::noop_with_empty_axes_;

  bool calculate_log_;
  bool calculate_sqt_;
  bool log_sum_exp_;
  // Indicates if this reduction can be delegated to our highly-optimized reduction kernels.
  // Those efficient kernels are defined/implemented in reduction_functions.h/.cu.
  bool fast_reduction_;

  // We need to access to the ROCM EP instance to get the miopen handle
  ROCMExecutionProvider* rocm_ep_;
};

template <typename T>
class ArgMax final : public ReduceKernel<false> {
 public:
  ArgMax(const OpKernelInfo& info) : ReduceKernel<false>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T, MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES>(ctx, MIOPEN_REDUCE_TENSOR_MAX);
  }
};

template <typename T>
class ArgMin final : public ReduceKernel<false> {
 public:
  ArgMin(const OpKernelInfo& info) : ReduceKernel<false>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T, MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES>(ctx, MIOPEN_REDUCE_TENSOR_MIN);
  }
};

template <typename T>
class ReduceL1 final : public ReduceKernel<true> {
 public:
  ReduceL1(const OpKernelInfo& info) : ReduceKernel<true>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    //return ComputeImpl<T>(ctx, MIOPEN_REDUCE_TENSOR_NORM1);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MIOpen does not yet support reduce norm1.");
  }
};

template <typename T>
class ReduceL2 final : public ReduceKernel<true> {
 public:
  ReduceL2(const OpKernelInfo& info) : ReduceKernel<true>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    //return ComputeImpl<T>(ctx, MIOPEN_REDUCE_TENSOR_NORM2);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MIOpen does not yet support reduce norm2.");
  }
};

template <typename T>
class ReduceMax final : public ReduceKernel<true> {
 public:
  ReduceMax(const OpKernelInfo& info) : ReduceKernel<true>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, MIOPEN_REDUCE_TENSOR_MAX);
  }
};

template <typename T>
class ReduceMean final : public ReduceKernel<true> {
 public:
  ReduceMean(const OpKernelInfo& info) : ReduceKernel<true>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, MIOPEN_REDUCE_TENSOR_AVG);
  }
};

template <typename T>
class ReduceMin final : public ReduceKernel<true> {
 public:
  ReduceMin(const OpKernelInfo& info) : ReduceKernel<true>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, MIOPEN_REDUCE_TENSOR_MIN);
  }
};

template <typename T>
class ReduceProd final : public ReduceKernel<true> {
 public:
  ReduceProd(const OpKernelInfo& info) : ReduceKernel<true>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, MIOPEN_REDUCE_TENSOR_MUL);
  }
};

template <typename T>
class ReduceSum final : public ReduceKernel<true> {
 public:
  ReduceSum(const OpKernelInfo& info) : ReduceKernel<true>(info) {
    fast_reduction_ = true;
  }

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, MIOPEN_REDUCE_TENSOR_ADD);
  }
};

template <typename T>
class ReduceLogSum final : public ReduceKernel<true> {
 public:
  ReduceLogSum(const OpKernelInfo& info) : ReduceKernel<true>(info) {
    ReduceKernel<true>::calculate_log_ = true;
  }

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, MIOPEN_REDUCE_TENSOR_ADD);
  }
};

template <typename T>
class ReduceSumSquare final : public ReduceKernel<true> {
 public:
  ReduceSumSquare(const OpKernelInfo& info) : ReduceKernel<true>(info) {
    ReduceKernel<true>::calculate_sqt_ = true;
  }

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, MIOPEN_REDUCE_TENSOR_ADD);
  }
};

template <typename T>
class ReduceLogSumExp final : public ReduceKernel<true> {
 public:
  ReduceLogSumExp(const OpKernelInfo& info) : ReduceKernel<true>(info) {
    ReduceKernel<true>::log_sum_exp_ = true;
  }

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, MIOPEN_REDUCE_TENSOR_ADD);
  }
};

Status PrepareForReduce(const Tensor* X,
                        bool keepdims,
                        const std::vector<int64_t>& axes,
                        PrepareReduceMetadata& prepare_reduce_metadata,
                        const TensorShape* input_shape_override = nullptr);

template <typename T, miopenReduceTensorIndices_t ReduceTensorIndices>
Status ReduceComputeCore(ROCMExecutionProvider& rocm_ep, const Tensor& input, PrepareReduceMetadata& prepare_reduce_metadata,
                         /*out*/ Tensor& output, miopenReduceTensorOp_t miopen_reduce_op,
                         const std::vector<int64_t>& axes,
                         bool calculate_log, bool calculate_sqt, bool log_sum_exp, bool fast_reduction,
                         const TensorShape* input_shape_override = nullptr);

// ROCM's reduction descriptor miopenReduceTensorDescriptor_t is a pointer so
// it's safer to wrap it with automatically memory deleter as MiopenReduceDescriptor.
// An implicit caster from MiopenReduceDescriptor to miopenReduceTensorDescriptor_t
// is implemented below, so ROCM can seamlessly work.
class MiopenReduceDescriptor final {
 public:
  MiopenReduceDescriptor() : desc_(nullptr) {
  }

  ~MiopenReduceDescriptor() {
    if (desc_ != nullptr) {
      miopenDestroyReduceTensorDescriptor(desc_);
      desc_ = nullptr;
    }
  }

  MiopenReduceDescriptor(const MiopenReduceDescriptor&) = delete;
  MiopenReduceDescriptor& operator=(const MiopenReduceDescriptor&) = delete;

  Status Set(miopenReduceTensorOp_t op, miopenDataType_t type, miopenReduceTensorIndices_t indices) {
    if (!desc_)
      MIOPEN_RETURN_IF_ERROR(miopenCreateReduceTensorDescriptor(&desc_));

    MIOPEN_RETURN_IF_ERROR(miopenSetReduceTensorDescriptor(
        desc_,
        op,
        type,
        MIOPEN_PROPAGATE_NAN,
        indices,
        MIOPEN_32BIT_INDICES));  // currently only the 32-bit (unsigned int) type is supported.
    return Status::OK();
  }

  operator miopenReduceTensorDescriptor_t() const { return desc_; }

 private:
  miopenReduceTensorDescriptor_t desc_;
};

}  // namespace rocm
}  // namespace onnxruntime
