// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <unordered_set>
#include <vector>

#include "core/framework/session_options.h"
#include "core/optimizer/graph_transformer.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/rewrite_rule.h"
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_REPLAY_IN_MINIMAL_BUILD)
#include "core/optimizer/selectors_actions/selector_action_transformer_apply_contexts.h"
#endif

namespace onnxruntime {
class IExecutionProvider;

namespace optimizer_utils {

#if !defined(ORT_MINIMAL_BUILD)

/** Generates all predefined rules for this level.
   If rules_to_enable is not empty, it returns the intersection of predefined rules and rules_to_enable.
   TODO: This is visible for testing at the moment, but we should rather make it private. */
std::vector<std::unique_ptr<RewriteRule>> GenerateRewriteRules(
    TransformerLevel level,
    const std::unordered_set<std::string>& rules_to_disable = {});

/** Given a TransformerLevel, this method generates a name for the rule-based graph transformer of that level. */
std::string GenerateRuleBasedTransformerName(TransformerLevel level);

/** Generates all rule-based transformers for this level. */
std::unique_ptr<RuleBasedGraphTransformer> GenerateRuleBasedGraphTransformer(
    TransformerLevel level,
    const std::unordered_set<std::string>& rules_to_disable,
    const std::unordered_set<std::string>& compatible_execution_providers);

/** Generates all predefined (both rule-based and non-rule-based) transformers for this level.
    Any transformers or rewrite rules named in rules_and_transformers_to_disable will be excluded. */
std::vector<std::unique_ptr<GraphTransformer>> GenerateTransformers(
    TransformerLevel level,
    const SessionOptions& session_options,
    const IExecutionProvider& execution_provider /*required by constant folding*/,
    const std::unordered_set<std::string>& rules_and_transformers_to_disable = {});

#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_REPLAY_IN_MINIMAL_BUILD)

/** Generates all predefined transformers which support runtime optimizations for this level.
    Any transformers or rewrite rules named in rules_and_transformers_to_disable will be excluded.

    This is a distinct function from GenerateTransformers() because:
    - Runtime optimizations are used in a different scenario than normal graph optimization.
    - The set of transformers which support runtime optimizations is different. */
std::vector<std::unique_ptr<GraphTransformer>> GenerateTransformersForRuntimeOptimizations(
    TransformerLevel level,
    const SessionOptions& session_options,
    const SatApplyContextVariant& apply_context,
    const std::unordered_set<std::string>& rules_and_transformers_to_disable = {});

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_REPLAY_IN_MINIMAL_BUILD)

}  // namespace optimizer_utils
}  // namespace onnxruntime
