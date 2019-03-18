// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include <unordered_set>
#include "core\util\eigen_common_wrapper.h"

namespace onnxruntime {

// utility helpers specific to Reverse
inline bool axes_has_dupes(const std::vector<int64_t>& axes) {
  if (axes.size() == 0)
    return false;

  std::unordered_set<int64_t> elements;
  for (const auto& axis : axes) {
    if (elements.find(axis) != elements.end())
      return true;
  }

  return false;
}

template <int rank>
Eigen::array<bool, rank> vector_to_eigen_array(const std::vector<int64_t>& reverse_axes) {
  Eigen::array<bool, rank> eigen_reverse_axes;

  // default axes - reverse all axes as per spec
  if (reverse_axes.size() == 0) {
    for (int i = 0; i < rank; ++i) {
      eigen_reverse_axes[i] = true;
    }
    return eigen_reverse_axes;
  }

  // explicit axes given
  eigen_reverse_axes.fill(false);
  for (int i = 0; i < rank; ++i) {
    const auto& dim = reverse_axes[i];
    eigen_reverse_axes[dim >= 0 ? dim : dim + rank] = true;
  }
  return eigen_reverse_axes;
}

// some templated aliases
template <typename T, int rank>
using EigenTensorMap = Eigen::TensorMap<Eigen::Tensor<T, rank, Eigen::RowMajor>>;

template <typename T, int rank>
using ConstEigenTensorMap = Eigen::TensorMap<Eigen::Tensor<const T, rank, Eigen::RowMajor>>;

template <typename T, int rank>
using EigenTensorMapPair = std::pair<ConstEigenTensorMap<T, rank>, EigenTensorMap<T, rank>>;

template <typename T, int rank>
inline EigenTensorMapPair<T, rank> buffers_to_eigen_tensormaps(const T* input_buffer, T* output_buffer, const std::vector<int64_t>& dims) {
  if (rank == 1)
    return std::make_pair(
        ConstEigenTensorMap<T, rank>(input_buffer, static_cast<size_t>(dims[0])),
        EigenTensorMap<T, rank>(output_buffer, static_cast<size_t>(dims[0])));
  else if (rank == 2)
    return std::make_pair(
        ConstEigenTensorMap<T, rank>(input_buffer, static_cast<size_t>(dims[0], dims[1])),
        EigenTensorMap<T, rank>(output_buffer, static_cast<size_t>(dims[0], dims[1])));
  else if (rank == 3)
    return std::make_pair(
        ConstEigenTensorMap<T, rank>(input_buffer, static_cast<size_t>(dims[0], dims[1], dims[2])),
        EigenTensorMap<T, rank>(output_buffer, static_cast<size_t>(dims[0], dims[1], dims[2])));
  else if (rank == 4)
    return std::make_pair(
        ConstEigenTensorMap<T, rank>(input_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3])),
        EigenTensorMap<T, rank>(output_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3])));
  else if (rank == 5)
    return std::make_pair(
        ConstEigenTensorMap<T, rank>(input_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3], dims[4])),
        EigenTensorMap<T, rank>(output_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3], dims[4])));
  else if (rank == 6)
    return std::make_pair(
        ConstEigenTensorMap<T, rank>(input_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5])),
        EigenTensorMap<T, rank>(output_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5])));
  else if (rank == 7)
    return std::make_pair(
        ConstEigenTensorMap<T, rank>(input_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6])),
        EigenTensorMap<T, rank>(output_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6])));
  else if (rank == 8)
    return std::make_pair(
        ConstEigenTensorMap<T, rank>(input_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], dims[7])),
        EigenTensorMap<T, rank>(output_buffer, static_cast<size_t>(dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], dims[7])));
  else
    ORT_THROW("unsupported conversion from raw buffers to Eigen tensors");
}

// Reverse kernel
class Reverse final : public OpKernel {
 public:
  explicit Reverse(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
    auto has_axes = op_kernel_info.GetAttrs("axes", attr_axes_).IsOK();
    ORT_ENFORCE(!has_axes || !axes_has_dupes(attr_axes_), "axes attribute has duplicates, this is not accordance with Reverse op spec");
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

  std::vector<int64_t> attr_axes_;
};
}  // namespace onnxruntime
