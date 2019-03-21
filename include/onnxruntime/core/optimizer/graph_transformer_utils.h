// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

namespace transformer_utils {

/** Generates all predefined rules for this level.
   If rules_to_enable is not empty, it returns the intersection of predefined rules and rules_to_enable. */
std::vector<std::unique_ptr<RewriteRule>> GenerateRewriteRules(const TransformerLevel& level,
                                                               const std::vector<std::string>* rules_to_enable = nullptr);

/** Generates all predefined rules for a specific level, and then creates a rule-based graph transformer and adds the rules to it.
    If rules_to_enable is not empty, adds to the transformer the intersection of predefined rules and rules_to_enable. */
std::unique_ptr<RuleBasedGraphTransformer> GenerateRuleBasedGraphTransformer(const TransformerLevel& level,
                                                                             const std::vector<std::string>* rules_to_enable,
                                                                             std::string t_name);

/** A graph transformer along with the set of execution providers for which it will be applied. */
using TransformerProviderSet = std::pair<std::unique_ptr<GraphTransformer>, std::vector<std::string>>;

/** Generates all predefined transformers for this level.
    If transformers_to_enable is not empty, it returns the intersection of predefined transformers and transformers_to_enable. */
std::vector<TransformerProviderSet> GenerateTransformers(const TransformerLevel& level,
                                                         const std::vector<std::string>* transformers_to_enable = nullptr);

}  // namespace transformer_utils
}  // namespace onnxruntime
