// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <functional>
#include "gradient_builder_base.h"
#include "generic_registry.h"

namespace onnxruntime {
namespace training {

typedef GenericRegistry<GradientBuilderBase,
                        const GradientGraphConfiguration&,
                        Graph*&,                                 // graph
                        const Node*&,                            // node
                        const std::unordered_set<std::string>&,  // gradient_inputs
                        const std::unordered_set<std::string>&,  // gradient_outputs
                        const logging::Logger&,
                        std::unordered_set<std::string>&>
    GradientRegistryType;

class GradientBuilderRegistry : public GradientRegistryType {
 public:
  void RegisterGradientBuilders();

  static GradientBuilderRegistry& GetInstance() {
    static GradientBuilderRegistry instance;
    return instance;
  }

 private:
  GradientBuilderRegistry() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GradientBuilderRegistry);
};

GradientDef GetGradientForOp(const GradientGraphConfiguration& gradient_graph_config,
                             Graph* graph,
                             const Node* node,
                             const std::unordered_set<std::string>& output_args_need_grad,
                             const std::unordered_set<std::string>& input_args_need_grad,
                             const logging::Logger& logger,
                             std::unordered_set<std::string>& stashed_tensors);

}  // namespace training
}  // namespace onnxruntime
