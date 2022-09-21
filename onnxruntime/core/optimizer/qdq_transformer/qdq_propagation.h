// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
    @Class QDQPropagationTransformer

    Propagate Q backward, DQ forward. This is done by inserting Q/DQ pairs that match the starting Q or DQ past each
    subsequent (previous for Q, next for DQ) op supporting propagation.
*/
class QDQPropagationTransformer : public GraphTransformer {
 public:
  QDQPropagationTransformer(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("QDQPropagationTransformer", compatible_execution_providers) {
  }

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
