// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/quantized_embed_layer_norm_fusion.h"

namespace onnxruntime {

Status QuantizedEmbedLayerNormFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  return Status::OK();
}

}  // namespace onnxruntime
