// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

struct BinaryElementwisePreparation {
  const Tensor* lhs_tensor = nullptr;
  const Tensor* rhs_tensor = nullptr;
  Tensor* output_tensor = nullptr;
  int32_t output_rank_or_simple_broadcast = 0;  // for no_broadcast|left_scalar|right_scalar cases, output_rank uses SimpleBroadcast enums

  TArray<int64_t> lhs_padded_strides;
  TArray<int64_t> rhs_padded_strides;
  TArray<fast_divmod> fdm_output_strides;

  // these are for RightPerChannel case
  fast_divmod fdm_H;
  fast_divmod fdm_C;

  BinaryElementwisePreparation() {}

  Status BinaryElementwiseBroadcastPrepareHelper(const TensorShape& lhs_shape, const TensorShape& rhs_shape,
                                                 const TensorShape& output_shape) {
    int32_t lhs_rank = gsl::narrow_cast<int32_t>(lhs_shape.NumDimensions());
    int32_t rhs_rank = gsl::narrow_cast<int32_t>(rhs_shape.NumDimensions());
    int32_t out_rank = std::max(lhs_rank, rhs_rank);

    // early return when shapes match
    if (lhs_shape == rhs_shape) {
      output_rank_or_simple_broadcast = static_cast<int32_t>(SimpleBroadcast::NoBroadcast);
      return Status::OK();
    }

    // early return if one operand is scalar
    if (lhs_shape.Size() == 1 || rhs_shape.Size() == 1) {
      output_rank_or_simple_broadcast =
          static_cast<int32_t>(lhs_shape.Size() == 1 ? SimpleBroadcast::LeftScalar : SimpleBroadcast::RightScalar);
      return Status::OK();
    }

    // Coalesce dimensions.
    std::vector<int64_t> out_shape_vec = output_shape.GetDimsAsVector();
    std::vector<int64_t> lhs_shape_vec(static_cast<size_t>(out_rank), 1);
    std::vector<int64_t> rhs_shape_vec(static_cast<size_t>(out_rank), 1);
    int lhs_offset = out_rank - lhs_rank;
    int rhs_offset = out_rank - rhs_rank;
    for (int i = 0; i < lhs_rank; ++i) {
      lhs_shape_vec[i + lhs_offset] = lhs_shape[i];
    }
    for (int i = 0; i < rhs_rank; ++i) {
      rhs_shape_vec[i + rhs_offset] = rhs_shape[i];
    }

    int prev_dim = 0;
    for (int dim = 1; dim < out_rank; ++dim) {
      if (lhs_shape_vec[prev_dim] == out_shape_vec[prev_dim] && rhs_shape_vec[prev_dim] == out_shape_vec[prev_dim] &&
          lhs_shape_vec[dim] == out_shape_vec[dim] && rhs_shape_vec[dim] == out_shape_vec[dim]) {
        lhs_shape_vec[prev_dim] *= lhs_shape_vec[dim];
        rhs_shape_vec[prev_dim] *= rhs_shape_vec[dim];
        out_shape_vec[prev_dim] *= out_shape_vec[dim];
      } else {
        ++prev_dim;
        if (prev_dim != dim) {
          lhs_shape_vec[prev_dim] = lhs_shape_vec[dim];
          rhs_shape_vec[prev_dim] = rhs_shape_vec[dim];
          out_shape_vec[prev_dim] = out_shape_vec[dim];
        }
      }
    }

    int new_rank = prev_dim + 1;
    lhs_shape_vec.resize(new_rank);
    rhs_shape_vec.resize(new_rank);
    out_shape_vec.resize(new_rank);

    // special case for lhs(N,C,H) and rhs (C,1) which is used in conv bias
    // when N == 1: out[id] = op(lhs[id], rhs[id / H])
    // When N > 1:  out[id] = op(lhs[id], rhs[id / H % C])
    if (lhs_shape == output_shape) {
      int64_t C = 0;
      if (1 == std::count_if(rhs_shape_vec.begin(), rhs_shape_vec.end(), [&C](int64_t dim) {
            if (dim != 1) C = dim;
            return (dim != 1);
          })) {
        int32_t dim_C = gsl::narrow_cast<int32_t>(std::find(rhs_shape_vec.begin(), rhs_shape_vec.end(), C) -
                                                  rhs_shape_vec.begin() + out_shape_vec.size() - rhs_shape_vec.size());
        TensorShape new_out_shape(out_shape_vec);
        int64_t N = new_out_shape.SizeToDimension(dim_C);
        int64_t H = (dim_C < new_rank - 1 ? new_out_shape.SizeFromDimension(dim_C + 1) : 1);
        if (N == 1) {
          output_rank_or_simple_broadcast = static_cast<int32_t>(SimpleBroadcast::RightPerChannelBatch1);
          fdm_H = fast_divmod(gsl::narrow_cast<int>(H));
        } else {
          output_rank_or_simple_broadcast = static_cast<int32_t>(SimpleBroadcast::RightPerChannelBatchN);
          fdm_H = fast_divmod(gsl::narrow_cast<int>(H));
          fdm_C = fast_divmod(gsl::narrow_cast<int>(C));
        }
        return Status::OK();
      }
    }

    output_rank_or_simple_broadcast = new_rank;
    if (lhs_shape != output_shape) {
      TensorPitches original_lhs_padded_strides(lhs_shape_vec);
      lhs_padded_strides.SetSize(new_rank);
      for (int i = 0; i < new_rank; ++i) {
        lhs_padded_strides[i] = lhs_shape_vec[i] == out_shape_vec[i] ? original_lhs_padded_strides[i] : 0;
      }
    }

    if (rhs_shape != output_shape) {
      TensorPitches original_rhs_padded_strides(rhs_shape_vec);
      rhs_padded_strides.SetSize(new_rank);
      for (int i = 0; i < new_rank; ++i) {
        rhs_padded_strides[i] = rhs_shape_vec[i] == out_shape_vec[i] ? original_rhs_padded_strides[i] : 0;
      }
    }

    TensorPitches original_output_strides(out_shape_vec);
    fdm_output_strides.SetSize(new_rank);
    for (int i = 0; i < new_rank; ++i) {
      fdm_output_strides[i] = fast_divmod(gsl::narrow_cast<int>(original_output_strides[i]));
    }

    return Status::OK();
  }
};

Status ComputeOutputShape(
    const std::string& node_name,
    const TensorShape& lhs_shape,
    const TensorShape& rhs_shape,
    TensorShape& out_shape);

Status BinaryElementwiseBroadcastPrepare(
    const Tensor* lhs_tensor,
    const Tensor* rhs_tensor,
    Tensor* output_tensor,
    BinaryElementwisePreparation* p,
    const TensorShape* override_lhs_shape = nullptr,
    const TensorShape* override_rhs_shape = nullptr);

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
  Status Prepare(OpKernelContext* context, BinaryElementwisePreparation* p) const;
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
class Pow_7 final : public BinaryElementwise<ShouldBroadcast> {
 public:
  Pow_7(const OpKernelInfo& info) : BinaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

// Since version 12
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

template <typename T, typename CudaT>
class CompareFunction : public BinaryElementwise<ShouldBroadcast> {
 public:
  CompareFunction(const OpKernelInfo& info) : BinaryElementwise(info) {}

  typedef void (*ImplCompare)(cudaStream_t stream,
                              int32_t output_rank_or_simple_broadcast,
                              const TArray<int64_t>* lhs_padded_strides,
                              const CudaT* lhs_data,
                              const TArray<int64_t>* rhs_padded_strides,
                              const CudaT* rhs_data,
                              const TArray<fast_divmod>* fdm_output_strides,
                              const fast_divmod& fdm_H,
                              const fast_divmod& fdm_C,
                              bool* output_data,
                              size_t count);

  Status CompareMethod(OpKernelContext* context, ImplCompare Impl_Compare) const;
};

template <typename T>
class Greater final : public CompareFunction<T, typename ToCudaType<T>::MappedType> {
 public:
  Greater(const OpKernelInfo& info) : CompareFunction<T, typename ToCudaType<T>::MappedType>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Equal final : public CompareFunction<T, typename ToCudaType<T>::MappedType> {
 public:
  Equal(const OpKernelInfo& info) : CompareFunction<T, typename ToCudaType<T>::MappedType>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class Less final : public CompareFunction<T, typename ToCudaType<T>::MappedType> {
 public:
  Less(const OpKernelInfo& info) : CompareFunction<T, typename ToCudaType<T>::MappedType>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GreaterOrEqual final : public CompareFunction<T, typename ToCudaType<T>::MappedType> {
 public:
  GreaterOrEqual(const OpKernelInfo& info) : CompareFunction<T, typename ToCudaType<T>::MappedType>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class LessOrEqual final : public CompareFunction<T, typename ToCudaType<T>::MappedType> {
 public:
  LessOrEqual(const OpKernelInfo& info) : CompareFunction<T, typename ToCudaType<T>::MappedType>(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
