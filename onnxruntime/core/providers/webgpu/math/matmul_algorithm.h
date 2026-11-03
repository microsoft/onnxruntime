// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>

namespace onnxruntime {
namespace webgpu {

enum class MatMulAlgorithm {
  SubgroupMatrix,
  Gemv,
  Naive,
  Subgroup,
  Packed,
  PackedSplitK,
};

std::string_view MatMulAlgorithmName(MatMulAlgorithm algorithm);

}  // namespace webgpu
}  // namespace onnxruntime
