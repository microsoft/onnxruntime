// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <deque>
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_mgr.h"

namespace onnxruntime {

class NchwcTransformer : public onnxruntime::GraphTransformer {
 public:
  NchwcTransformer() noexcept;

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override;

  GraphTransformerManager stage1_graph_transformer_mgr_;
  GraphTransformerManager stage2_graph_transformer_mgr_;
};

}  // namespace onnxruntime
