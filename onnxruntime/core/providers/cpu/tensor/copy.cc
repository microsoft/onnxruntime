//
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/common/common.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/copy.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {

std::vector<int64_t> StridesForTensor(const Tensor& tensor) {
  auto shape = tensor.Shape();
  auto strides = std::vector<int64_t>(shape.NumDimensions());
  int64_t running_size = 1;
  for (auto i = shape.NumDimensions(); i > 0; i--) {
    strides[i - 1] = running_size;
    running_size *= shape[i - 1];
  }

  return strides;
}

namespace {
/*
    Check if we can coalesce dim with dim + 1.

    We can do this if:
      * either of the dims have shape 1
      * strides[dim + 1] * shape[dim + 1] = strides[dim] (for all tensors)
*/
inline bool CanCoalesce(
    std::initializer_list<std::reference_wrapper<std::vector<int64_t>>>& tensors_strides,
    const std::vector<int64_t>& shape,
    std::size_t dim,
    std::size_t ndim) {
  auto shape_dim = shape[dim];
  auto shape_ndim = shape[ndim];
  if (shape_dim == 1 || shape_ndim == 1) {
    return true;
  }

  for (const auto& cur_stride : tensors_strides) {
    std::vector<int64_t>& strides = cur_stride.get();
    if (shape_ndim * strides[ndim] != strides[dim]) {
      return false;
    }
  }
  return true;
}

/*
    Copy the stride from ndim to dim in all tensors.
*/
inline void CopyStride(
    std::initializer_list<std::reference_wrapper<std::vector<int64_t>>>& tensors_strides,
    std::size_t dim, std::size_t ndim) {
  for (const auto& cur_stride : tensors_strides) {
    std::vector<int64_t>& strides = cur_stride.get();
    strides[dim] = strides[ndim];
  }
}

}  // namespace

/*
    Coalesce contiguous dimensions in the tensors. Operates inplace on the function arguments.
*/
void CoalesceDimensions(std::initializer_list<std::reference_wrapper<std::vector<int64_t>>>&& tensors_strides,
                        std::vector<int64_t>& shape) {
  const std::size_t dims = shape.size();

  // the current dimension is the one we are attempting to "coalesce onto"
  std::size_t current_dim = 0;

  for (std::size_t dim = 1; dim < dims; dim++) {
    // check if dim can be coalesced with current_dim
    if (CanCoalesce(tensors_strides, shape, current_dim, dim)) {
      if (shape[dim] != 1) {
        CopyStride(tensors_strides, current_dim, dim);
      }
      shape[current_dim] *= shape[dim];
    } else {
      current_dim++;

      if (current_dim != dim) {
        // we have coalesced at least one value before this: bump forward the values into the correct place
        CopyStride(tensors_strides, current_dim, dim);
        shape[current_dim] = shape[dim];
      }
    }
  }

  shape.resize(current_dim + 1);
  for (const auto& cur_stride : tensors_strides) {
    std::vector<int64_t>& strides = cur_stride.get();
    strides.resize(current_dim + 1);
  }
}

}  // namespace onnxruntime
