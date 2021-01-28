// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/optional.h"
#include "core/providers/cpu/reduction/reduction_ops.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/rocm/reduction/reduction_functions.h"

namespace onnxruntime {
namespace rocm {

enum miopenReduceTensorOp_t {
  MIOPEN_REDUCE_TENSOR_MAX,
  MIOPEN_REDUCE_TENSOR_MIN,
  MIOPEN_REDUCE_TENSOR_NORM1,
  MIOPEN_REDUCE_TENSOR_NORM2,
  MIOPEN_REDUCE_TENSOR_AVG,
  MIOPEN_REDUCE_TENSOR_MUL,
  MIOPEN_REDUCE_TENSOR_ADD
};

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
  int64_t rank;
  int64_t stride;
  bool contiguous_axes;
};

Status PrepareForReduce(const Tensor* X,
                        bool keepdims,
                        const std::vector<int64_t>& axes,
                        PrepareReduceMetadata& prepare_reduce_metadata,
                        const TensorShape* input_shape_override = nullptr);

template <typename T>
Status ReduceComputeCore(const Tensor& input, PrepareReduceMetadata& prepare_reduce_metadata,
                         /*out*/ Tensor& output, miopenReduceTensorOp_t miopen_reduce_op,
                         const std::vector<int64_t>& axes,
                         bool calculate_log, bool calculate_sqt, bool log_sum_exp, bool fast_reduction,
                         const TensorShape* input_shape_override = nullptr);

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
        fast_reduction_(false) {}

  template <typename T>
  Status ComputeImpl(OpKernelContext* ctx, miopenReduceTensorOp_t miopen_reduce_op) const;

  // Used by ReduceSumTraining which will have axes as input
  template <typename T>
  Status ComputeImplEx(OpKernelContext* ctx, miopenReduceTensorOp_t miopen_reduce_op) const;

  template <typename T, typename OutT>
  Status ReduceKernelShared(
      const T* X,
      const TensorShape& input_shape,
      OutT* Y,
      const TensorShape& output_shape,
      miopenReduceTensorOp_t miopen_reduce_op,
      std::vector<int64_t> output_dims) const;

  using ReduceKernelBase<allow_multi_axes>::axes_;
  using ReduceKernelBase<allow_multi_axes>::keepdims_;
  using ReduceKernelBase<allow_multi_axes>::noop_with_empty_axes_;

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
    return ComputeImpl<T>(ctx, MIOPEN_REDUCE_TENSOR_MAX);
  }
};

template <typename T>
class ArgMin final : public ReduceKernel<false> {
 public:
  ArgMin(const OpKernelInfo& info) : ReduceKernel<false>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, MIOPEN_REDUCE_TENSOR_MIN);
  }
};

template <typename T>
class ReduceL1 final : public ReduceKernel<true> {
 public:
  ReduceL1(const OpKernelInfo& info) : ReduceKernel<true>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, MIOPEN_REDUCE_TENSOR_NORM1);
  }
};

template <typename T>
class ReduceL2 final : public ReduceKernel<true> {
 public:
  ReduceL2(const OpKernelInfo& info) : ReduceKernel<true>(info) {}

  Status ComputeInternal(OpKernelContext* ctx) const override {
    return ComputeImpl<T>(ctx, MIOPEN_REDUCE_TENSOR_NORM2);
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

}  // namespace rocm
}  // namespace onnxruntime