// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

// Helper that creates a new TensorShape for the intermediate result of MatMul
// The new shape is created by appending the two dimensions dim1 and dim2 / components to the original shape
inline TensorShape CreateMatMulIntermediateShape(const TensorShape& shape, const int64_t dim1, const int64_t dim2, const int components) {
  TensorShapeVector shape_vec = shape.AsShapeVector();
  shape_vec.push_back(dim1);
  shape_vec.push_back(dim2 / components);
  return TensorShape(shape_vec);
}

// Helper that convert output batch indices to input batch indices using only the rank and
// the shape information in uniform
inline std::string ConvertOutputBatchIndicesToInputBatchIndices(const std::string& name, const ShaderVariableHelper& input, int input_batch_rank, int output_batch_rank, const std::string& batch_indices) {
  std::ostringstream oss;
  const std::string input_shape = "uniforms." + name + "_shape";
  const std::string input_indices = name + "_indices";
  int extending_input_rank = output_batch_rank - input_batch_rank;
  for (int i = 0; i < input_batch_rank; ++i) {
    oss << "if (" << GetElementAt(input_shape, i, input.Rank()) << " != 1) {\n"
        << input.IndicesSet(input_indices, i, GetElementAt(batch_indices, i + extending_input_rank, output_batch_rank)) << "\n"
        << "} else {\n"
        << input.IndicesSet(input_indices, i, 0) << "\n"
        << "}\n";
  }
  return oss.str();
}

}  // namespace webgpu
}  // namespace onnxruntime
