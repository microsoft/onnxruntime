// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>

#include "core/common/status.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime::webgpu::util {

/**
 * Q or DQ op quantization granularity.
 */
enum class QuantizationType {
  PerTensor = 0,
  PerAxis,
  Blocked,
};

/**
 * Validates the Q or DQ op input shapes and attributes and detects the quantization type.
 * For inout parameters `axis` and `block_size`, the input is the attribute value and the output is the effective
 * value to use, if applicable.
 * `quantization_type` will be set to the detected quantization type.
 *
 * @param input_shape The shape of the input data tensor.
 * @param scale_shape The shape of the scale tensor.
 * @param zero_point_shape The shape of the zero point tensor, if present.
 * @param[inout] axis The axis value.
 *                    The input value may be negative.
                      It will be normalized to a non-negative value for `PerAxis` and `Blocked` quantization.
 * @param[inout] block_size The block size value.
 *                          The input value may be zero, meaning unspecified.
 *                          If unspecified, it will be set to the actual block size for `Blocked` quantization.
 * @param[out] quantization_type The detected quantization type.
 *
 * @return Status indicating success.
 */
Status ValidateAndDetectQuantizationType(const TensorShape& input_shape,
                                         const TensorShape& scale_shape,
                                         const TensorShape* zero_point_shape,
                                         int64_t& axis,
                                         int64_t& block_size,
                                         QuantizationType& quantization_type);

/**
 * Determines whether the specified ONNX element data type is signed.
 */
bool IsOnnxElementDataTypeSigned(int32_t data_type);

/**
 * Modes for packing values into a single u32.
 */
enum class U32PackingMode {
  None = 0,
  Pack8bx4,  // pack 4 8-bit values per u32
  Pack4bx8,  // pack 8 4-bit values per u32
};

/**
 * Determines the packing mode to use for the specified ONNX data element type.
 */
U32PackingMode GetOnnxTensorElementDataTypeU32PackingMode(int32_t data_type);

/**
 * Returns the number of elements packed into a single u32 for the given packing mode, if applicable.
 */
std::optional<int> GetU32PackingModeNumComponents(U32PackingMode packing_mode);

}  // namespace onnxruntime::webgpu::util
