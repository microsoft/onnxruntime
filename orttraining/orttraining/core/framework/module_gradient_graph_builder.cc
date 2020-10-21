// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/model.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "orttraining/core/framework/module_gradient_graph_builder.h"
#include "orttraining/core/framework/gradient_graph_builder.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/core/optimizer/graph_transformer_utils.h"

namespace onnxruntime {
namespace training {

std::string ModuleGradientGraphBuilder::Build(std::istream& model_istream, const ModuleGradientGraphBuilderConfiguration& config) {
  const logging::Logger& logger = logging::LoggingManager::DefaultLogger();  // use default logger for now.
  ONNX_NAMESPACE::ModelProto mp;
  Model::Load(model_istream, &mp);
  Model model(mp, nullptr, logger);
  model.MainGraph().Resolve();

  const TrainingSession::TrainingConfiguration::GraphTransformerConfiguration graph_transformer_config{};
  GraphTransformerManager graph_transformation_mgr{2};
  std::unique_ptr<CPUExecutionProvider> cpu_execution_provider =
      onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());

  auto add_transformers = [&](TransformerLevel level) {
    auto transformers_to_register = transformer_utils::GeneratePreTrainingTransformers(
        level, config.weight_names_to_train, graph_transformer_config, *cpu_execution_provider, {});
    for (auto& entry : transformers_to_register) {
      graph_transformation_mgr.Register(std::move(entry), level);
    }
  };

  for (int i = static_cast<int>(TransformerLevel::Level1); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    TransformerLevel level = static_cast<TransformerLevel>(i);
    if (TransformerLevel::MaxLevel >= level) {
      add_transformers(level);
    }
  }

  // apply transformers
  Graph& graph = model.MainGraph();
  for (int i = static_cast<int>(TransformerLevel::Level1); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    graph_transformation_mgr.ApplyTransformers(graph, static_cast<TransformerLevel>(i), logger);
  }

  // TODO: mixed precision transformer.
  
  GradientGraphConfiguration gradient_graph_config{};
  gradient_graph_config.use_invertible_layernorm_grad = config.use_invertible_layernorm_grad;
  gradient_graph_config.set_gradients_as_graph_outputs = config.set_gradients_as_graph_outputs;
  GradientGraphBuilder grad_graph_builder(&model.MainGraph(),
                                          config.output_names,
                                          config.weight_names_to_train,
                                          "", // not support loss name for now.
                                          gradient_graph_config,
                                          logger);
  grad_graph_builder.Build();

  std::string str;
  model.ToProto().SerializeToString(&str);
  return str;
}

}  // namespace training
}  // namespace onnxruntime
