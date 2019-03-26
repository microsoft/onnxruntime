// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class InsertOutputTransformer : public GraphTransformer {
 public:
  InsertOutputTransformer(const std::string& name)
      : onnxruntime::GraphTransformer(name, "Transformer to insert addtional outputs to some node to facilitate training") {
  }

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override;
};
}  // namespace onnxruntime
