// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

struct BinaryElementwisePreparation {
  const Tensor* lhs_tensor = nullptr;
  const Tensor* rhs_tensor = nullptr;
  Tensor* output_tensor = nullptr;
  size_t output_rank_or_simple_broadcast = 0;               // for no_broadcast|left_scalar|right_scalar cases, output_rank uses SimpleBroadcast enums
  CudaKernel::CudaAsyncBuffer<int64_t> lhs_padded_strides;  // for lhs shape == output shape, this is nullptr
  CudaKernel::CudaAsyncBuffer<int64_t> rhs_padded_strides;  // for rhs shape == output shape, this is nullptr
  CudaKernel::CudaAsyncBuffer<fast_divmod> fdm_output_strides;

  // these are for RightPerChannel case
  fast_divmod fdm_H;
  fast_divmod fdm_C;

  BinaryElementwisePreparation(const CudaKernel* op_kernel) : lhs_padded_strides(op_kernel),
                                                              rhs_padded_strides(op_kernel),
                                                              fdm_output_strides(op_kernel) {}

  Status CopyToGpu() {
    ORT_RETURN_IF_ERROR(lhs_padded_strides.CopyToGpu());
    ORT_RETURN_IF_ERROR(rhs_padded_strides.CopyToGpu());
    ORT_RETURN_IF_ERROR(fdm_output_strides.CopyToGpu());
    return Status::OK();
  }

  Status BinaryElementwiseBroadcastPrepareHelper(int device_id, const TensorShape& lhs_shape,
                                                 const TensorShape& rhs_shape,
                                                 const TensorShape& output_shape) {
    size_t lhs_rank = lhs_shape.NumDimensions();
    size_t rhs_rank = rhs_shape.NumDimensions();
    size_t out_rank = std::max(lhs_rank, rhs_rank);

    // early return when shapes match
    if (lhs_shape == rhs_shape) {
      output_rank_or_simple_broadcast = static_cast<size_t>(SimpleBroadcast::NoBroadcast);
      return Status::OK();
    }

    // early return if one operand is scalar
    if (lhs_shape.Size() <= 1 || rhs_shape.Size() <= 1) {
      output_rank_or_simple_broadcast = static_cast<size_t>(lhs_shape.Size() <= 1 ? SimpleBroadcast::LeftScalar : SimpleBroadcast::RightScalar);
      return Status::OK();
    }

    // special case for lhs(N,C,H) and rhs (C,1) which is used in conv bias
    // when N == 1: out[id] = op(lhs[id], rhs[id / H])
    // When N > 1:  out[id] = op(lhs[id], rhs[id / H % C])
    if (lhs_shape == output_shape) {
      const auto& rhs_dims = rhs_shape.GetDims();
      int64_t C = 0;
      if (1 == std::count_if(rhs_dims.begin(), rhs_dims.end(), [&C](int64_t dim) { if (dim > 1) C = dim; return (dim > 1); })) {
        auto dim_C = std::find(rhs_dims.begin(), rhs_dims.end(), C) - rhs_dims.begin() + output_shape.NumDimensions() - rhs_shape.NumDimensions();
        int64_t N = output_shape.SizeToDimension(dim_C);
        int64_t H = (dim_C < out_rank - 1 ? output_shape.SizeFromDimension(dim_C + 1) : 1);

        std::vector<int64_t> new_output_dims;
        if (N == 1) {
          output_rank_or_simple_broadcast = static_cast<size_t>(SimpleBroadcast::RightPerChannelBatch1);
          fdm_H = fast_divmod(gsl::narrow_cast<int>(H));
        } else {
          output_rank_or_simple_broadcast = static_cast<size_t>(SimpleBroadcast::RightPerChannelBatchN);
          fdm_H = fast_divmod(gsl::narrow_cast<int>(H));
          fdm_C = fast_divmod(gsl::narrow_cast<int>(C));
        }
        return Status::OK();
      }
    }

    output_rank_or_simple_broadcast = out_rank;

    if (lhs_shape != output_shape) {
      // compute strides with 1 more dim than out_rank, and use strides[0] == strides[1]
      // to decide if dim0 needs broadcast
      lhs_padded_strides.AllocCpuPtr(device_id, out_rank + 1);
      ORT_RETURN_IF_NOT(TensorPitches::Calculate(lhs_padded_strides.CpuSpan(), lhs_shape.GetDims()));
      if (lhs_shape[0] > 1 && lhs_rank == out_rank)
        lhs_padded_strides.CpuPtr()[0] = 0;
    }

    if (rhs_shape != output_shape) {
      rhs_padded_strides.AllocCpuPtr(device_id, out_rank + 1);
      ORT_RETURN_IF_NOT(TensorPitches::Calculate(rhs_padded_strides.CpuSpan(), rhs_shape.GetDims()));
      if (rhs_shape[0] > 1 && rhs_rank == out_rank)
        rhs_padded_strides.CpuPtr()[0] = 0;
    }

    fdm_output_strides.AllocCpuPtr(device_id, out_rank);
    ORT_RETURN_IF_NOT(CalculateFdmStrides(fdm_output_strides.CpuSpan(), output_shape.GetDims()));
    return Status::OK();
  }
};

// trait classes to indicate if the kernel supports broadcast
class ShouldBroadcast {
};

class ShouldNotBroadcast {
};

template <typename BroadcastTrait>
class BinaryElementwise : public CudaKernel {
 protected:
  typedef BroadcastTrait broadcast_type;

  BinaryElementwise(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext*) const override {
    return Status(common::ONNXRUNTIME, common::FAIL);  // should not reach here
  }
  Status Prepare(OpKernelContext* context, int device_id, BinaryElementwisePreparation* p) const;
};

template <typename T>
class Add final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Add(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Sub final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Sub(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Mul final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Mul(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Div final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Div(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Pow final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Pow(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class And final : public BinaryElementwise<ShouldBroadcast> {
 public:
  And(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Or final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Or(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Xor final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Xor(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

// PRelu is activation function, but it's closer to binary elementwise ops in implementation
template <typename T>
class PRelu final : public BinaryElementwise<ShouldBroadcast> {
 public:
  PRelu(const OpKernelInfo& info) : BinaryElementwise(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

// Sum allows varadic inputs, and it uses binary elementwise Add in implementation
template <typename T>
class Sum final : public CudaKernel {
 public:
  Sum(const OpKernelInfo& info) : CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Greater final : public CudaKernel {
 public:
  Greater(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Max final : public CudaKernel {
 public:
  Max(const OpKernelInfo& info) : CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Min final : public CudaKernel {
 public:
  Min(const OpKernelInfo& info) : CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};
}  // namespace cuda
}  // namespace onnxruntime
