// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "orttraining/core/graph/generic_registry.h"
#include "orttraining/core/graph/optimizer_graph_builder.h"

namespace onnxruntime {
namespace training {

class OptimizerGraphBuilderRegistry : public GenericRegistry<OptimizerGraphBuilder,
                                                             const OptimizerBuilderRegistry&,
                                                             const OptimizerGraphConfig&,
                                                             const std::unordered_map<std::string, OptimizerNodeConfig>&> {
 public:
  // Register optimizer graph builders.
  void RegisterGraphBuilders();

  std::string GetNameFromConfig(const OptimizerGraphConfig& config) const;

  static OptimizerGraphBuilderRegistry& GetInstance() {
    static OptimizerGraphBuilderRegistry instance;
    return instance;
  }

 private:
  OptimizerGraphBuilderRegistry() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OptimizerGraphBuilderRegistry);
};

}  // namespace training
}  // namespace onnxruntime
