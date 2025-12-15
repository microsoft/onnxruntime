// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/webgpu/webgpu_external_header.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {

class ShaderVariableHelper;

template <typename T>
inline T CeilDiv(T numerator, T denominator) {
  return (numerator + denominator - 1) / denominator;
}

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

/**
 * Configuration for Split-K optimization (Conv|MatMul).
 */
class SplitKConfig {
 public:
  explicit SplitKConfig(const wgpu::AdapterInfo& adapter_info);

  bool UseSplitK(
      bool is_vec4, ActivationKind activation_kind, uint64_t batch_size,
      bool is_channels_last, uint32_t dim_a_outer,
      uint32_t dim_b_outer, uint32_t dim_inner) const;

  uint32_t GetSplitDimInner() const;

 private:
  bool enable_split_k_ = false;
  uint32_t split_dim_inner_ = 0;
  uint32_t min_dim_inner_with_split_k_ = 0;
  uint32_t max_dim_inner_with_split_k_ = 0;
  float max_dim_a_outer_multiplies_dim_b_outer_divides_dim_inner_ = 0.0f;
};

/**
 * Generates WGSL (WebGPU Shading Language) code for performing an atomic add operation
 * on a non-integer value (e.g., floating-point) in a shader.
 *
 * Since WGSL natively supports atomic operations only on integer types, this function
 * generates code that emulates atomic addition for non-integer types using a compare-and-swap loop.
 *
 * @param output        A reference to the ShaderVariableHelper representing the atomic variable
 *                      to be updated. This encapsulates the variable's name and access logic.
 * @param offset        The offset or index within the atomic variable where the operation is applied.
 * @param output_type   The WGSL type of the value being added (e.g., "f32").
 * @param add_value     The expression or variable representing the value to add.
 * @return              A string containing the generated WGSL code for the atomic add operation.
 */
std::string GenerateAtomicAddNonIntegerCode(const ShaderVariableHelper& output, const std::string& offset, const std::string& output_type, const std::string& add_value);

}  // namespace webgpu
}  // namespace onnxruntime
