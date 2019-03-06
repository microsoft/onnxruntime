// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/training/loss_function_builder.h"
#include "core/training/training_session.h"

namespace onnxruntime {
namespace training {

GraphAugmenter::GraphDefs LossFunctionBuilder::Build(const Graph& graph, const LossFunctionInfo& loss_func_info) {
  const auto* loss_func = LossFunctionRegistry::GetInstance().GetLossFunction(loss_func_info.loss_func_name_);
  ORT_ENFORCE(loss_func != nullptr, "The loss function has not been registered:", loss_func_info.loss_func_name_);

  // Invoke the creator to generate GraphDefs, without worrying about label's TypeProto.
  auto graph_defs = (*loss_func)(loss_func_info);

  // Now let's set label's TypeProto from pridiction's.
  // TODO: see if we want to relax this.
  const auto* prediction_type_proto = graph.GetNodeArg(loss_func_info.prediction_name_)->TypeAsProto();

  for (auto& node : graph_defs.NodeDefs()) {
    for (auto& arg_def : node.input_args) {
      if (arg_def.name == loss_func_info.label_name_) {
        arg_def.type_proto = prediction_type_proto;
      }
    }
  }

  return graph_defs;
}
}  // namespace training
}  // namespace onnxruntime
