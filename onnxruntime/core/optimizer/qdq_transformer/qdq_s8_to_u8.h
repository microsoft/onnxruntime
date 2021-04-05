// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
    @Class QDQS8ToU8Transformer

    Convert QuantizeLinear and DequantizeLinear pair with type int8_t to type uint8_t
    */
class QDQS8ToU8Transformer : public GraphTransformer {
 public:
  QDQS8ToU8Transformer(const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("QDQS8ToU8Transformer", compatible_execution_providers) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
