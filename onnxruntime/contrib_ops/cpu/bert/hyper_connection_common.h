// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <vector>

#include "contrib_ops/cpu/bert/deepseek_v4_compression_common.h"

namespace onnxruntime {
namespace contrib {
namespace deepseek_v4_attention_impl {

inline Status ValidateHyperParameters(const Tensor& hidden, const Tensor& weight, const Tensor& bias,
                                      const Tensor& scale, bool is_head, int64_t& rows, int64_t& streams,
                                      int64_t& hidden_size) {
  ORT_RETURN_IF_NOT(hidden.Shape().NumDimensions() == 4 && weight.Shape().NumDimensions() == 2 &&
                        bias.Shape().NumDimensions() == 1 && scale.Shape().NumDimensions() == 1,
                    "HyperConnection inputs have invalid ranks.");
  streams = hidden.Shape()[2];
  hidden_size = hidden.Shape()[3];
  rows = hidden.Shape()[0] * hidden.Shape()[1];
  const int64_t output_size = is_head ? streams : (2 + streams) * streams;
  const int64_t expected_scales = is_head ? 1 : 3;
  ORT_RETURN_IF_NOT(streams > 0 && hidden_size > 0 && weight.Shape()[0] == output_size &&
                        weight.Shape()[1] == streams * hidden_size && bias.Shape().Size() == output_size &&
                        scale.Shape().Size() == expected_scales,
                    "HyperConnection parameter shapes are inconsistent with hidden_streams.");
  ORT_RETURN_IF_NOT(weight.IsDataType<float>() && bias.IsDataType<float>() && scale.IsDataType<float>(),
                    "HyperConnection parameters must use float32.");
  return Status::OK();
}

inline void NormalizeHyperInput(const std::vector<float>& hidden, int64_t rows, int64_t width,
                                float epsilon, std::vector<float>& normalized) {
  normalized.resize(hidden.size());
  for (int64_t row = 0; row < rows; ++row) {
    float square_sum = 0.0f;
    for (int64_t index = 0; index < width; ++index) {
      const float value = hidden[static_cast<size_t>(row * width + index)];
      square_sum += value * value;
    }
    const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(width) + epsilon);
    for (int64_t index = 0; index < width; ++index) {
      normalized[static_cast<size_t>(row * width + index)] =
          hidden[static_cast<size_t>(row * width + index)] * inverse_rms;
    }
  }
}

inline void ProjectHyperInput(const std::vector<float>& normalized, const float* weight,
                              int64_t rows, int64_t input_size, int64_t output_size,
                              std::vector<float>& projected) {
  projected.resize(static_cast<size_t>(rows * output_size));
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t output = 0; output < output_size; ++output) {
      float value = 0.0f;
      for (int64_t input = 0; input < input_size; ++input) {
        value += normalized[static_cast<size_t>(row * input_size + input)] *
                 weight[output * input_size + input];
      }
      projected[static_cast<size_t>(row * output_size + output)] = value;
    }
  }
}

}  // namespace deepseek_v4_attention_impl
}  // namespace contrib
}  // namespace onnxruntime