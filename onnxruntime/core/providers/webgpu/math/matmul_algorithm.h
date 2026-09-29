// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>
#include <string_view>

namespace onnxruntime {
namespace webgpu {

enum class MatMulAlgorithm {
  SubgroupMatrix,
  Naive,
  Subgroup,
  Packed,
  PackedSplitK,
};

std::optional<MatMulAlgorithm> ParseMatMulAlgorithm(std::string_view name);

std::string_view MatMulAlgorithmName(MatMulAlgorithm algorithm);

}  // namespace webgpu
}  // namespace onnxruntime
