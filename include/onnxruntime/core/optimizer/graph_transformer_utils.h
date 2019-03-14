// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/rewrite_rule.h"

namespace onnxruntime {

namespace transformerutils {
/* Validates whether level can be mapped to a valid TransformerLevel enum
*/
Status ValidateTransformerLevel(unsigned int level);

/* Sets bit mask for transformers enabled based on the level provided
*  Also populates a convenience list of all transformers  
*/
void SetTransformerContext(const uint32_t& level, uint32_t& levels_enabled, 
                           std::vector<TransformerLevel>* all_levels = nullptr);

/* Generates rules listed in custom list for the given level.
*  If custom list is empty returns all pre-defined rules for this level
*/
std::vector<std::unique_ptr<RewriteRule>> GenerateRewriteRules(const TransformerLevel& level, 
                                                               const std::vector<std::string>* custom_list = nullptr);

using TransformerProviderSet = std::pair<std::unique_ptr<GraphTransformer>, std::vector<std::string>>;
std::vector<TransformerProviderSet> GenerateTransformers(const TransformerLevel& level, 
                                                           const std::vector<std::string>* custom_list = nullptr);

}  // namespace transformerutils
}  // namespace onnxruntime
