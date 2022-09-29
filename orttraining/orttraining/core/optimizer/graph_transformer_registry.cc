// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/optimizer/graph_transformer_registry.h"

namespace onnxruntime {
namespace training {

#ifdef ORT_TRAINING_EXTERNAL_GRAPH_TRANSFORMERS
void RegisterTrainingExternalTransformers();
#endif 

void GraphTransformerRegistry::RegisterExternalGraphTransformers() {
#ifdef ORT_TRAINING_EXTERNAL_GRAPH_TRANSFORMERS
  RegisterTrainingExternalTransformers();
#endif
}

void GenerateExternalTransformers(
    TransformerLevel level,
    bool before_gradient_builder,
    const InlinedHashSet<std::string_view>& ep_list,
    std::vector<std::unique_ptr<GraphTransformer>>& output) {
  auto& registered_transformers = GraphTransformerRegistry::GetInstance().GetAllRegisteredTransformers();
  for (auto& [k, v] : registered_transformers) {
    if (v.before_gradient_builder != before_gradient_builder || v.level != level)
      continue;
    output.push_back(GraphTransformerRegistry::GetInstance().CreateTransformer(k, ep_list));
  }
}

}  // namespace training
}  // namespace onnxruntime
