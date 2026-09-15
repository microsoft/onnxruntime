#pragma once
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class AttentionMemOpt : public GraphTransformer {
 public:
  AttentionMemOpt(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("AttentionMemOpt", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
