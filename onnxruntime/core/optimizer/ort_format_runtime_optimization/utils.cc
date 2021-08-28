// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/ort_format_runtime_optimization/utils.h"

#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.h"

namespace onnxruntime::optimizer_utils {

std::vector<std::unique_ptr<GraphTransformer>> GenerateOrtFormatRuntimeTransformers() {
  std::vector<std::unique_ptr<GraphTransformer>> transformers{};
  transformers.emplace_back(std::make_unique<QDQSelectorActionTransformer>());
  return transformers;
}

}  // namespace onnxruntime::optimizer_utils
