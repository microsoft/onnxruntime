// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class ConvActivationFusion : public onnxruntime::GraphTransformer {
 public:
  ConvActivationFusion() noexcept : onnxruntime::GraphTransformer("ConvActivationFusion", "Fusing Activation into Conv", TransformerLevel::Optional_L2, 
      std::vector<std::string>{onnxruntime::kCpuExecutionProvider}) {}

 private:
  Status ApplyImpl(onnxruntime::Graph& graph, bool& modified, int graph_level) const override;  
};

}  // namespace onnxruntime
