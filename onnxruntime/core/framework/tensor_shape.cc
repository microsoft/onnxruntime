// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor_shape.h"
#include <iostream>
#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {

TensorShape::TensorShape(const int64_t* dimension_sizes, size_t dimension_count)
    : std::vector<int64_t>(dimension_count) {
  for (size_t i = 0; i < dimension_count; ++i) {
    (*this)[i] = dimension_sizes[i];
  }
}

TensorShape::TensorShape(const std::vector<int64_t>& dims, size_t start, size_t end) {
  assign(dims.begin() + start, dims.begin() + end);
}

/**
 * Return the total number of elements. Returns 1 for an empty (rank 0) TensorShape.
 */
int64_t TensorShape::Size() const {
  size_t arraySize = size();
  int64_t size = SizeHelper(0, arraySize);
  //should we cache the size? as multiple operation may be expensive.
  return size;
}

int64_t TensorShape::SizeToDimension(size_t dimension) const {
  const size_t num_dims = size();
  ORT_ENFORCE(dimension <= num_dims,
              "Invalid dimension of ", dimension, " for SizeFromDimension. Tensor has ",
              num_dims, " dimensions.");

  int64_t size = SizeHelper(0, dimension);
  return size;
}

int64_t TensorShape::SizeFromDimension(size_t dimension) const {
  const size_t num_dims = size();
  ORT_ENFORCE(dimension <= num_dims,
              "Invalid dimension of ", dimension, " for SizeFromDimension. Tensor has ",
              num_dims, " dimensions.");

  int64_t size = SizeHelper(dimension, num_dims);
  return size;
}

TensorShape TensorShape::Slice(size_t dimstart, size_t dimend) const {
  ORT_ENFORCE(dimstart <= dimend && dimend <= size(),
              "Invalid tensor shape slice argument.");
  return TensorShape(*this, dimstart, dimend);
}

TensorShape TensorShape::Slice(size_t dimstart) const {
  return Slice(dimstart, size());
}

// output dimensions
std::string TensorShape::ToString() const {
  std::string result;

  result.append("{");
  bool first = true;
  for (auto dim : (*this)) {
    if (!first) {
      result.append(",");
    }

    result.append(std::to_string(dim));
    first = false;
  }
  result.append("}");

  return result;
}

int64_t TensorShape::SizeHelper(size_t start, size_t end) const {
  // Must return 1 for an empty sequence
  SafeInt<int64_t> size = 1;  // this is used to calculate the size, which is used for memory allocations, so validate no overflow
  for (size_t i = start; i < end; i++) {
    if ((*this)[i] < 0) return -1;
    size *= (*this)[i];
  }
  return size;
}

// operator<< to nicely output to a stream
std::ostream& operator<<(std::ostream& out, const ::onnxruntime::TensorShape& shape) {
  return (out << shape.ToString());
}

}  // namespace onnxruntime
