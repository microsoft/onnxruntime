// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <queue>
#include <string>

#include "core/optimizer/graph_transformer.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/compute_optimizer/shared_utils.h"

namespace onnxruntime {

/**
@Class PaddingElimination


*/
class PaddingElimination : public GraphTransformer {
 public:
  PaddingElimination(const InlinedHashSet<std::string_view>& compatible_execution_providers = {},
                     const std::vector<std::string>& sparse_embedding_input_names = {}) noexcept
      : GraphTransformer("PaddingElimination", compatible_execution_providers),
        sparse_embedding_input_names_{sparse_embedding_input_names} {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
private:
  std::vector<std::string> sparse_embedding_input_names_;
};

}  // namespace onnxruntime
