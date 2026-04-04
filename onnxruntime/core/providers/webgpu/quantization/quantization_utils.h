// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime::webgpu::util {

enum class QuantizationType {
  PerTensor,
  PerAxis,
  Blocked,
};

/**
 * Validates the parameters and detects the Q or DQ op quantization type.
 * The `axis` and `block_size` parameters will be updated if applicable.
 *
 * @param input_shape The shape of the input data tensor.
 * @param scale_shape The shape of the scale tensor.
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
Status DetectQuantizationType(const TensorShape& input_shape, const TensorShape& scale_shape,
                              int64_t& axis, int64_t& block_size,
                              QuantizationType& quantization_type);

}  // namespace onnxruntime::webgpu::util
