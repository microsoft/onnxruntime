// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include <unordered_set>
#include "core\util\eigen_common_wrapper.h"

namespace onnxruntime {
namespace contrib {

// some templated aliases
template <typename T, int rank>
using EigenTensorMap = Eigen::TensorMap<Eigen::Tensor<T, rank, Eigen::RowMajor, Eigen::DenseIndex>>;

template <typename T, int rank>
using ConstEigenTensorMap = Eigen::TensorMap<Eigen::Tensor<const T, rank, Eigen::RowMajor, Eigen::DenseIndex>>;

// utility helpers specific to Reverse
// explicit dupes - (e.g.) axes contains 0 and another 0
inline bool axes_has_explicit_dupes(const std::vector<int64_t>& axes) {
  if (axes.size() == 0)
    return false;

  std::unordered_set<int64_t> elements;
  for (const auto& axis : axes) {
    if (elements.find(axis) != elements.end())
      return true;
    elements.insert(axis);
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

template <int rank>
Eigen::DSizes<Eigen::DenseIndex, rank> dims_as_eigen_dsizes(const std::vector<int64_t> dims) {
  Eigen::DSizes<Eigen::DenseIndex, rank> eigen_dsizes;
  for (int i = 0; i < rank; ++i) {
    eigen_dsizes[i] = static_cast<Eigen::DenseIndex>(dims[i]);
  }
  return eigen_dsizes;
}

template <typename T, int rank>
EigenTensorMap<T, rank> buffer_as_eigen_tensor(T* buffer, const std::vector<int64_t>& dims) {
  return EigenTensorMap<T, rank>(buffer, dims_as_eigen_dsizes<rank>(dims));
}

template <typename T, int rank>
ConstEigenTensorMap<T, rank> buffer_as_const_eigen_tensor(const T* buffer, const std::vector<int64_t>& dims) {
  return ConstEigenTensorMap<T, rank>(buffer, dims_as_eigen_dsizes<rank>(dims));
}

// Reverse kernel
class Reverse final : public OpKernel {
 public:
  explicit Reverse(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
    auto has_axes = op_kernel_info.GetAttrs("axes", attr_axes_).IsOK();
    ORT_ENFORCE(!has_axes || !axes_has_explicit_dupes(attr_axes_), "axes attribute has duplicate values in Reverse operator");
  }

  Status Compute(OpKernelContext* p_op_kernel_context) const override;

  std::vector<int64_t> attr_axes_;
};
}  // namespace contrib
}  // namespace onnxruntime