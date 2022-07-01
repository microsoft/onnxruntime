// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

struct TernaryElementwisePreparation {
  const Tensor* a_tensor = nullptr;
  const Tensor* b_tensor = nullptr;
  const Tensor* c_tensor = nullptr;
  Tensor* output_tensor = nullptr;
  size_t output_rank_or_simple_broadcast = 0;  // for no_broadcast cases, output_rank uses SimpleBroadcast enums
  TArray<int64_t> a_padded_strides;            // for a shape == output shape, this is nullptr
  TArray<int64_t> b_padded_strides;            // for b shape == output shape, this is nullptr
  TArray<int64_t> c_padded_strides;            // for c shape == output shape, this is nullptr
  TArray<fast_divmod> fdm_output_strides;
  BroadcastIndexType a_index_type = BroadcastIndexType::NoBroadcast;
  BroadcastIndexType b_index_type = BroadcastIndexType::NoBroadcast;
  BroadcastIndexType c_index_type = BroadcastIndexType::NoBroadcast;

  TernaryElementwisePreparation() {}

  Status TernaryElementwiseBroadcastPrepareHelper(const TensorShape& a_shape,
                                                  const TensorShape& b_shape,
                                                  const TensorShape& c_shape,
                                                  const TensorShape& output_shape) {
    int32_t a_rank = static_cast<int32_t>(a_shape.NumDimensions());
    int32_t b_rank = static_cast<int32_t>(b_shape.NumDimensions());
    int32_t c_rank = static_cast<int32_t>(c_shape.NumDimensions());
    int32_t out_rank = std::max(std::max(a_rank, b_rank), c_rank);

    // early return when shapes match
    if (a_shape == b_shape && b_shape == c_shape) {
      output_rank_or_simple_broadcast = static_cast<size_t>(SimpleBroadcast::NoBroadcast);
      return Status::OK();
    }

    output_rank_or_simple_broadcast = out_rank;

    auto padder = [out_rank](int32_t rank, const TensorShape& shape, TArray<int64_t>& padded_strides) {
      padded_strides.SetSize(out_rank);
      if (rank > 0) {
        TensorPitches pitches(shape.GetDims());
        auto offset = out_rank - rank;
        for (auto i = offset; i < out_rank; ++i) {
          // the stride for broadcast dimension is kept as 0
          if (shape.GetDims()[i - offset] != 1) {
            padded_strides[i] = pitches[i - offset];
          }
        }
      }
    };

    bool has_need_compute = false;
    if (a_shape.Size() == 1) {
      a_index_type = BroadcastIndexType::Scalar;
    } else if (a_shape != output_shape) {
      padder(a_rank, a_shape, a_padded_strides);
      a_index_type = BroadcastIndexType::NeedCompute;
      has_need_compute = true;
    }

    if (b_shape.Size() == 1) {
      b_index_type = BroadcastIndexType::Scalar;
    } else if (b_shape != output_shape) {
      padder(b_rank, b_shape, b_padded_strides);
      b_index_type = BroadcastIndexType::NeedCompute;
      has_need_compute = true;
    }

    if (c_shape.Size() == 1) {
      c_index_type = BroadcastIndexType::Scalar;
    } else if (c_shape != output_shape) {
      padder(c_rank, c_shape, c_padded_strides);
      c_index_type = BroadcastIndexType::NeedCompute;
      has_need_compute = true;
    }

    if (!has_need_compute) {
      output_rank_or_simple_broadcast = static_cast<size_t>(SimpleBroadcast::NoBroadcast);
      return Status::OK();
    }

    TensorPitches output_pitches(output_shape.GetDims());
    fdm_output_strides.SetSize(out_rank);
    for (auto i = 0; i < out_rank; ++i) {
      fdm_output_strides[i] = fast_divmod(static_cast<int32_t>(output_pitches[i]));
    }

    return Status::OK();
  }
};

// Compute where operator output shape based upon three way broad-casting.
Status ComputeOutputShape(const std::string& node_name, const TensorShape& cond_shape,
                          const TensorShape& x_shape, const TensorShape& y_shape, TensorShape& out_shape);

Status TernaryElementwiseBroadcastPrepare(
    const Tensor* a_tensor,
    const Tensor* b_tensor,
    const Tensor* c_tensor,
    Tensor* output_tensor,
    TernaryElementwisePreparation* p);

// trait classes to indicate if the kernel supports broadcast
class ShouldBroadcast {
};

class ShouldNotBroadcast {
};

template <typename BroadcastTrait>
class TernaryElementwise : public CudaKernel {
 protected:
  typedef BroadcastTrait broadcast_type;

  TernaryElementwise(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext*) const override {
    return Status(common::ONNXRUNTIME, common::FAIL);  // should not reach here
  }
  Status Prepare(OpKernelContext* context, TernaryElementwisePreparation* p) const;
};

template <typename T>
class BiasGeluGrad_dX final : public TernaryElementwise<ShouldBroadcast> {
 public:
  BiasGeluGrad_dX(const OpKernelInfo& info) : TernaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
