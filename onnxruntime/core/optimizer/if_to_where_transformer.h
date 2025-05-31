#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

// Transformer that lowers dynamic If nodes into Where ops by inlining both branches.
class IfToWhereTransformer final : public GraphTransformer {
 public:
  IfToWhereTransformer(const InlinedHashSet<std::string_view>& compatible_execution_providers = {kQnnExecutionProvider}) noexcept
      : GraphTransformer("IfToWhereTransformer", compatible_execution_providers) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const;
};

}  // namespace onnxruntime