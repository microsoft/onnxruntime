// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {
namespace webgpu {

/**
 * Returns the maximum number of components `N` to be used as `vecN` for the given size.
 */
inline int GetMaxComponents(int64_t size) {
  if (size % 4 == 0) {
    return 4;
  } else if (size % 2 == 0) {
    return 2;
  }
  return 1;
}

/**
 * Returns a string representing a WGSL expression that sums the components of a value T.
 *
 * T can be a scalar S, vec2<S> or vec4<S>.
 */
inline std::string SumVector(std::string x, int components) {
  switch (components) {
    case 1:
      return x;
    case 2:
      return "(" + x + ".x + " + x + ".y" + ")";
    case 4:
      return "(" + x + ".x + " + x + ".y + " + x + ".z + " + x + ".w" + ")";
    default:
      ORT_THROW("Unsupported number of components: ", components);
  }
}

inline std::string MakeScalarOrVectorType(int components, std::string_view data_type) {
  switch (components) {
    case 1:
      return std::string{data_type};
    case 2:
      return MakeStringWithClassicLocale("vec2<", data_type, ">");
    case 3:
      return MakeStringWithClassicLocale("vec3<", data_type, ">");
    case 4:
      return MakeStringWithClassicLocale("vec4<", data_type, ">");
    default:
      ORT_THROW("Unsupported number of components: ", components);
  }
}

TensorShape ReduceShapeByComponents(const TensorShape& shape, int64_t components);

/**
 * Create a reshaped tensor from an existing tensor.
 *
 * The specified new shape must have the same number of elements as the original tensor.
 *
 * The new tensor is a "view" of the original tensor. It uses the same data of the original tensor.
 * The new tensor does not take or share ownership of the underlying data. The original tensor must outlive the new tensor.
 */
inline Tensor CreateTensorView(const Tensor& tensor, const TensorShape& new_shape) {
  ORT_ENFORCE(tensor.Shape().Size() == new_shape.Size(), "Cannot reshape tensor ", tensor.Shape().ToString(), " to ", new_shape.ToString());
  return {tensor.DataType(), new_shape, const_cast<void*>(tensor.DataRaw()), tensor.Location()};
}

/**
 * Create a reinterpreted tensor from an existing tensor with a new data type and shape.
 *
 * The new data type and shape must match the original tensor's storage size.
 *
 * The new tensor is a "view" of the original tensor. It uses the same data of the original tensor.
 * The new tensor does not take or share ownership of the underlying data. The original tensor must outlive the new tensor.
 */
inline Tensor CreateTensorView(const Tensor& tensor, MLDataType new_data_type, const TensorShape& new_shape) {
  auto byte_size = Tensor::CalculateTensorStorageSize(tensor.DataType(), tensor.Shape());
  auto new_byte_size = Tensor::CalculateTensorStorageSize(new_data_type, new_shape);
  ORT_ENFORCE(byte_size == new_byte_size,
              "Cannot reshape tensor ", tensor.Shape().ToString(), " to ", new_shape.ToString(),
              " with data type ", DataTypeImpl::ToString(new_data_type), ". The byte size of the original tensor is ",
              byte_size, " and the byte size of the new tensor is ", new_byte_size);
  return {new_data_type, new_shape, const_cast<void*>(tensor.DataRaw()), tensor.Location()};
}

}  // namespace webgpu
}  // namespace onnxruntime
