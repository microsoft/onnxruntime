// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class GemmActivationFusion : public onnxruntime::GraphTransformer {
 public:
  GemmActivationFusion() noexcept
      : onnxruntime::GraphTransformer("GemmActivationFusion", "Fusing Activation into Gemm") {}

  Status ApplyImpl(Graph& graph, bool& modified, 
                   const std::vector<std::string>& compatible_provider_types, int graph_level) const override;
};

}  // namespace onnxruntime
