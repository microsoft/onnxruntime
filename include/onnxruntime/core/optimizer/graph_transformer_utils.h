// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

namespace transformerutils {
std::vector<std::unique_ptr<RewriteRule>> GenerateRewriteRules(const TransformerLevel& level, const std::vector<std::string>& custom_list);

std::vector<std::pair<std::unique_ptr<GraphTransformer>, std::vector<std::string>>> GenerateTransformers(const TransformerLevel& level, const std::vector<std::string>& custom_list);

}  // namespace utils
}  // namespace onnxruntime
