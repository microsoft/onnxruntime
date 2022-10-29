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
  QDQS8ToU8Transformer(bool weights_to_u8, const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("QDQS8ToU8Transformer", compatible_execution_providers), weights_to_u8_(weights_to_u8) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
  bool weights_to_u8_;
};

}  // namespace onnxruntime
