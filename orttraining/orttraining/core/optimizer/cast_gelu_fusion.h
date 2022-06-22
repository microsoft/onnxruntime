// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class CastGeluFusion

If there is Cast(to:float) before Gelu/GeluGrad and Cast(to:half) after,
remove all these Cast nodes for EPs when the actual compute inside the corresponding kernel implementation is
on float type, i.e., the kernels cast input half data to float, compute, and cast back output to half.

*/
class CastGeluFusion : public GraphTransformer {
 public:
  explicit CastGeluFusion(
      const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("CastGeluFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
