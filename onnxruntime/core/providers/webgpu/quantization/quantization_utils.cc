// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/quantization/quantization_utils.h"

#include "core/providers/common.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime::webgpu::util {

Status DetectQuantizationType(const TensorShape& input_shape, const TensorShape& scale_shape,
                              int64_t& axis, int64_t& block_size,
                              QuantizationType& quantization_type) {
  if (IsScalarOr1ElementVector(scale_shape)) {
    // PerTensor

    quantization_type = QuantizationType::PerTensor;
    return Status::OK();
  }

  // The axis is used for PerAxis and Blocked quantization.
  int64_t normalized_axis{};
  ORT_RETURN_IF_ERROR(HandleNegativeAxis(axis, input_shape.NumDimensions(), normalized_axis));

  if (scale_shape.NumDimensions() == 1 && scale_shape[0] == input_shape[normalized_axis]) {
    // PerAxis

    quantization_type = QuantizationType::PerAxis;
    axis = normalized_axis;
    return Status::OK();
  }

  const auto is_blocked = [&input_shape, &scale_shape, normalized_axis]() -> bool {
    if (scale_shape.NumDimensions() != input_shape.NumDimensions()) {
      return false;
    }

    for (size_t i = 0; i < scale_shape.NumDimensions(); ++i) {
      if (i != static_cast<size_t>(normalized_axis) && scale_shape[i] != input_shape[i]) {
        return false;
      }
    }
    return true;
  };

  if (is_blocked()) {
    // Blocked

    const auto input_dim = input_shape[normalized_axis];
    const auto scale_dim = scale_shape[normalized_axis];

    ORT_RETURN_IF_NOT(input_dim > 0 && scale_dim > 0 && input_dim >= scale_dim,
                      "Input block dimension (", input_dim, ") and scale block dimension (", scale_dim,
                      ") must be greater than 0. The input dimension must be greater than or equal to the scale "
                      "dimension.");

    ORT_RETURN_IF_NOT(block_size >= 0, "block_size must be non-negative.");

    int64_t actual_block_size{};
    if (block_size == 0) {
      // block_size is unspecified. Try to detect it.
      ORT_RETURN_IF_NOT(input_dim % scale_dim != 0,
                        "Automatic detection of block size requires input dimension (", input_dim,
                        ") to be a multiple of scale dimension (", scale_dim, ").");
      actual_block_size = input_dim / scale_dim;
    } else {
      const auto min_block_size = CeilDiv(input_dim, scale_dim);
      ORT_RETURN_IF(block_size < min_block_size,
                    "block_size (", block_size, ") must be at least ", min_block_size);

      if (scale_dim > 1) {
        const auto max_block_size = CeilDiv(input_dim, (scale_dim - 1)) - 1;
        ORT_RETURN_IF(block_size > max_block_size,
                      "block_size (", block_size, ") must be at most ", max_block_size);
      }

      actual_block_size = block_size;
    }

    quantization_type = QuantizationType::Blocked;
    axis = normalized_axis;
    block_size = actual_block_size;
    return Status::OK();
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unable to detect quantization type.");
}

}  // namespace onnxruntime::webgpu::util