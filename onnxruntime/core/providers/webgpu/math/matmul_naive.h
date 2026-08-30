// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>

#include "core/framework/tensor_shape.h"

namespace onnxruntime {
namespace webgpu {

struct MatMulNaiveProgramInfo {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t components;
  uint32_t a_components;
  uint32_t output_number;
  uint32_t output_size;
  uint32_t bias_components;
  size_t output_rank;
  TensorShape a_program_shape;
  TensorShape b_program_shape;
  TensorShape output_program_shape;
  TensorShape outer_dims;
};

std::optional<MatMulNaiveProgramInfo> AnalyzeMatMulNaiveProgram(
    const TensorShape& a_shape,
    const TensorShape& b_shape,
    bool is_channels_last);

}  // namespace webgpu
}  // namespace onnxruntime
