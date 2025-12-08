// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

#undef ERROR
#undef OPTIONAL

#include "GraphTransformerHelpers.h"

#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/graph_transformer_level.h"
#include "dynamic_quantize_convinteger_fusion.h"

namespace GraphTransformerHelpers {
std::vector<std::pair<std::unique_ptr<onnxruntime::GraphTransformer>, onnxruntime::TransformerLevel>> GetGraphTransformers() {
  // Register QNN EP graph transformers
  //
  std::vector<std::pair<std::unique_ptr<onnxruntime::GraphTransformer>, onnxruntime::TransformerLevel>> graphTransformers;
  std::unique_ptr<onnxruntime::RuleBasedGraphTransformer> rule_transformer =
      std::make_unique<onnxruntime::RuleBasedGraphTransformer>("QnnEpRuleTransformer");
  ORT_THROW_IF_ERROR(rule_transformer->Register(std::make_unique<onnxruntime::DynamicQuantizeConvIntegerFusion>()));
  graphTransformers.push_back(std::make_pair(std::move(rule_transformer),
                                             onnxruntime::TransformerLevel::Level1));
  return graphTransformers;
}
}  // namespace GraphTransformerHelpers
