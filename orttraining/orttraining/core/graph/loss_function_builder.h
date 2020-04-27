// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>
#include <utility>
#include "core/graph/graph.h"
#include "orttraining/core/graph/loss_function_registry.h"

namespace onnxruntime {
namespace training {

class LossFunctionBuilder {
 public:
  static std::unique_ptr<ILossFunction> Build(const std::string& loss_func_name);
};
}  // namespace training
}  // namespace onnxruntime
