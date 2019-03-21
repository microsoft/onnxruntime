// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <string>
#include <vector>
#include <functional>
#include <unordered_map>
#include "core/training/graph_augmenter.h"
#include "core/training/generic_registry.h"
#include "core/training/loss_func/loss_func_common.h"

namespace onnxruntime {
namespace training {

class LossFunctionRegistry : public GenericRegistry<ILossFunction> {
 public:
  // Register a list of non-operator loss functions stacitally.
  void RegisterNonOperatorLossFunctions();

  // Register a operator loss function.
  void RegisterOperatorLossFunction(const std::string& op_name);

  static LossFunctionRegistry& GetInstance() {
    static LossFunctionRegistry instance;
    return instance;
  }

 private:
  LossFunctionRegistry() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(LossFunctionRegistry);
};
}  // namespace training
}  // namespace onnxruntime
