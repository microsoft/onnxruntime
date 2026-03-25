// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class NchwcTransformer

Transformer that optimizes the graph by using NCHWc nodes instead of NCHW nodes
and inserts nodes to reorder tensors as needed.
*/
class NchwcTransformer : public GraphTransformer {
 public:
  explicit NchwcTransformer(bool enable_pointwise_avx512_activation_fusion = true) noexcept
      : GraphTransformer("NchwcTransformer"),
        enable_pointwise_avx512_activation_fusion_(enable_pointwise_avx512_activation_fusion) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  const bool enable_pointwise_avx512_activation_fusion_;
};

}  // namespace onnxruntime
