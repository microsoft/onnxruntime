// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

namespace transformer_utils {

/* Generates all predefined rules for this level
*  If rules_to_enable is not empty returns intersection of predefined rules and rules_to_enable 
*/
std::vector<std::unique_ptr<RewriteRule>> GenerateRewriteRules(const TransformerLevel& level, 
                                                               const std::vector<std::string>* rules_to_enable = nullptr);

/* Generates all predefined transformers for this level
*  If transformers_to_enable is not empty returns intersection of predefined transformers and transformers_to_enable 
*/
using TransformerProviderSet = std::pair<std::unique_ptr<GraphTransformer>, std::vector<std::string>>;
std::vector<TransformerProviderSet> GenerateTransformers(const TransformerLevel& level, 
                                                           const std::vector<std::string>* transformers_to_enable = nullptr);

}  // namespace transformerutils
}  // namespace onnxruntime
