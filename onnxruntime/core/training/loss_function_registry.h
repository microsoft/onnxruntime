// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include "core/training/graph_augmenter.h"
#include "core/training/loss_func/loss_func_common.h"

namespace onnxruntime {
namespace training {

typedef std::function<GraphAugmenter::GraphDefs(const LossFunctionInfo&)> LossFunction;

class LossFunctionRegistry {
 public:
  void RegisterCustomLossFunction(const std::string& loss_func_name);

  const LossFunction* GetLossFunction(const std::string& loss_func_name) const;

  static LossFunctionRegistry& GetInstance();

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LossFunctionRegistry);

  LossFunctionRegistry();

  void RegisterStandardLossFunction(const std::string& loss_func_name, const LossFunction& loss_func);

  std::unordered_map<std::string, LossFunction> loss_function_map_;
};
}  // namespace training
}  // namespace onnxruntime
