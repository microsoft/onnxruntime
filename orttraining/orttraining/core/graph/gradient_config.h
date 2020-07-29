// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace training {

struct GradientGraphConfiguration {
  // Layernorm gradient can be computed based on either input or output of layernorm.
  // That is to say, either input or output needs to be stashed for layernorm gradient.
  // To save memory, ideally, only one(input vs output) should be stashed rather than both.
  // By default, the input based algorithm is used. This flag is to enable the output based algorithm.
  bool use_invertible_layernorm_grad{false};
};

}  // namespace training
}  // namespace onnxruntime
