// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class MatMulAddFusion : public GraphTransformer {
 public:
  MatMulAddFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept 
      : GraphTransformer("MatMulAddFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
