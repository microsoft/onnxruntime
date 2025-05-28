// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class CastElimination
The transform that will try to find the longest chain of the type Cast where the 'to' attribute has the same data type as the input of the first Cast node in the chain.
E.g.
A ('float32') -> Cast (to='float16') ->  Cast (to='int4') ->  Cast (to='float32') -> Cast (to='float16') -> B
will reduce to
 A ('float32') -> Cast (to='float16') -> B

All the Cast nodes throughout the path need to have one input and one output to be considered for the fusion.
*/
class CastChainElimination : public GraphTransformer {
 public:
  explicit CastChainElimination(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("CastChainElimination", compatible_execution_providers) {
  }

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
