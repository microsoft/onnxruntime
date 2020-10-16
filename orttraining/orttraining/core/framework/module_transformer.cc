// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/model.h"
#include "orttraining/core/framework/module_transformer.h"
#include "orttraining/core/framework/gradient_graph_builder.h"

namespace onnxruntime {
namespace training {

std::string ModuleTransformer::Transform(std::istream& model_istream,
                                         const std::unordered_set<std::string>& weights_to_train,
                                         const std::unordered_set<std::string>& output_names) {
  ONNX_NAMESPACE::ModelProto mp;
  Model::Load(model_istream, &mp);
  Model model(mp, nullptr, logging::LoggingManager::DefaultLogger());
  model.MainGraph().Resolve();
  
  GradientGraphBuilder grad_graph_builder(&model.MainGraph(),
                                          output_names,
                                          weights_to_train,
                                          "",
                                          GradientGraphConfiguration(),
                                          logging::LoggingManager::DefaultLogger());
  grad_graph_builder.Build();
  std::string str;
  model.ToProto().SerializeToString(&str);
  return str;
}

}  // namespace training
}  // namespace onnxruntime
