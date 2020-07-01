// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace training {

struct GradientGraphConfiguration {
  bool use_invertible_layernorm_grad{false};
};

}  // namespace training
}  // namespace onnxruntime
