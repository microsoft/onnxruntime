// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace cuda {

enum class ScatterNDReduction : int {
  None = 0,
  Add = 1,
  Mul = 2,
  Min = 3,
  Max = 4,
};

}  // namespace cuda
}  // namespace onnxruntime
