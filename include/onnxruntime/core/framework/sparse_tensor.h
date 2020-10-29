#pragma once

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/data_types.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensor.h"

namespace onnxruntime {

/**
 * @brief This class implements SparseTensor. 
 * We represent a SparseTensor as a triple <values, indices, shape>. "values" and "indices" themselves
 * are implemented as Tensors. 
 * We follow the Tensor design for memory ownership/management: a sparse-tensor does not own the "value"
 * or "indices" tensors.
 */

class SparseTensor final {
 public:
  SparseTensor(MLDataType elt_type,
               const TensorShape& shape,
               size_t nnz,
               void* values_data,
               void* indices_data,
               const OrtMemoryInfo& memory_info);

  SparseTensor(MLDataType elt_type,
               const TensorShape& shape,
               size_t nnz,
               std::shared_ptr<IAllocator> allocator);

  ~SparseTensor() = default;

  // For now, disallow all copy, assignment, and move.
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SparseTensor);

  // Returns the number of entries in the values tensor (aka "NNZ" or "number of nonzero values")
  size_t NumValues() const { return static_cast<size_t>(values_.Shape().Size()); }

  const Tensor& Values() const {
    return values_;
  }

  const Tensor& Indices() const {
    return indices_;
  }

  const TensorShape& Shape() const {
    return shape_;
  }

  Tensor& MutableValues() {
    return values_;
  }

  Tensor& MutableIndices() {
    return indices_;
  }

  //TensorShape& MutableShape() {
  //  return shape_;
  //}

 private:
  Tensor values_;
  Tensor indices_;
  TensorShape shape_;  // The shape of corresponding dense-tensor.
};

}  // namespace onnxruntime
