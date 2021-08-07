// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
    @Class QDQPropagationTransformer

    Propagate Q backward, DQ forward and remove DQ/Q pair
*/
class QDQPropagationTransformer : public GraphTransformer {
 public:
  QDQPropagationTransformer(const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("QDQPropagationTransformer", compatible_execution_providers) {
  }

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
  bool PropagateDQForward(Graph& graph) const;
  bool PropagateQBackward(Graph& graph) const;
};

}  // namespace onnxruntime
