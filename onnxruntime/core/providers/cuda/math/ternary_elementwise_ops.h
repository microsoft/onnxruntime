// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

struct TernaryElementwisePreparation {
  const Tensor* x_tensor = nullptr;
  const Tensor* y_tensor = nullptr;
  const Tensor* z_tensor = nullptr;
  Tensor* output_tensor = nullptr;
  size_t output_rank_or_simple_broadcast = 0;  // for no_broadcast cases, output_rank uses SimpleBroadcast enums
  bool only_last_dim_broadcast = false;
  fast_divmod last_dim_fdm;
  TArray<int64_t> x_padded_strides;  // for a shape == output shape, this is nullptr
  TArray<int64_t> y_padded_strides;  // for b shape == output shape, this is nullptr
  TArray<int64_t> z_padded_strides;  // for c shape == output shape, this is nullptr
  TArray<fast_divmod> fdm_output_strides;
  BroadcastIndexType x_index_type = BroadcastIndexType::NoBroadcast;
  BroadcastIndexType y_index_type = BroadcastIndexType::NoBroadcast;
  BroadcastIndexType z_index_type = BroadcastIndexType::NoBroadcast;

  TernaryElementwisePreparation() {}

  Status TernaryElementwiseBroadcastPrepareHelper(
      const TensorShape& x_shape,
      const TensorShape& y_shape,
      const TensorShape& z_shape,
      const TensorShape& output_shape) {
    int32_t x_rank = static_cast<int32_t>(x_shape.NumDimensions());
    int32_t y_rank = static_cast<int32_t>(y_shape.NumDimensions());
    int32_t z_rank = static_cast<int32_t>(z_shape.NumDimensions());
    int32_t out_rank = std::max(std::max(x_rank, y_rank), z_rank);

    // early return when shapes match
    if (x_shape == y_shape && y_shape == z_shape) {
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
    if (x_shape.Size() == 1) {
      x_index_type = BroadcastIndexType::Scalar;
    } else if (x_shape != output_shape) {
      padder(x_rank, x_shape, x_padded_strides);
      x_index_type = BroadcastIndexType::NeedCompute;
      has_need_compute = true;
    }

    if (y_shape.Size() == 1) {
      y_index_type = BroadcastIndexType::Scalar;
    } else if (y_shape != output_shape) {
      padder(y_rank, y_shape, y_padded_strides);
      y_index_type = BroadcastIndexType::NeedCompute;
      has_need_compute = true;
    }

    if (z_shape.Size() == 1) {
      z_index_type = BroadcastIndexType::Scalar;
    } else if (z_shape != output_shape) {
      padder(z_rank, z_shape, z_padded_strides);
      z_index_type = BroadcastIndexType::NeedCompute;
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

    int last_dim = output_shape[out_rank - 1];
    last_dim_fdm = fast_divmod(last_dim);
    only_last_dim_broadcast =
        IsNoBroadcastOr1DBroadcast(x_index_type, x_rank, x_shape[x_rank - 1], out_rank, last_dim);
    only_last_dim_broadcast = only_last_dim_broadcast &&
                              IsNoBroadcastOr1DBroadcast(y_index_type, y_rank, y_shape[y_rank - 1],
                                                         out_rank, last_dim);
    only_last_dim_broadcast = only_last_dim_broadcast &&
                              IsNoBroadcastOr1DBroadcast(z_index_type, z_rank, z_shape[z_rank - 1],
                                                         out_rank, last_dim);

    std::cout << "only_last_dim_broadcast: " << only_last_dim_broadcast
              << "|" << x_rank << "," << x_shape[x_rank - 1] << "," << last_dim
              << "|"
              << y_rank << "," << y_shape[y_rank - 1]
              << "|"
              << z_rank << "," << z_shape[z_rank - 1] << std::endl;
    return Status::OK();
  }

 private:
  bool IsNoBroadcastOr1DBroadcast(BroadcastIndexType index_type, int32_t rank, int32_t last_dim,
                                  int32_t output_rank, int32_t output_last_dim) {
    if (index_type != BroadcastIndexType::NeedCompute and index_type != BroadcastIndexType::NoBroadcast) {
      return false;
    }

    return ((rank == 1 || rank == output_rank) && last_dim == output_last_dim);
  }
};

// Compute output shape based upon three way broad-casting.
Status ComputeOutputShape(const std::string& node_name, const TensorShape& cond_shape,
                          const TensorShape& x_shape, const TensorShape& y_shape, TensorShape& out_shape);

Status TernaryElementwiseBroadcastPrepare(const Tensor* x_tensor, const Tensor* y_tensor, const Tensor* z_tensor,
                                          Tensor* output_tensor, TernaryElementwisePreparation* p);

class TernaryElementwise : public CudaKernel {
 protected:
  TernaryElementwise(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext*) const override {
    return Status(common::ONNXRUNTIME, common::FAIL);  // should not reach here
  }
  Status Prepare(OpKernelContext* context, TernaryElementwisePreparation* p) const;
};

template <typename T>
class BiasGeluGrad_dX final : public TernaryElementwise {
 public:
  BiasGeluGrad_dX(const OpKernelInfo& info) : TernaryElementwise(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
