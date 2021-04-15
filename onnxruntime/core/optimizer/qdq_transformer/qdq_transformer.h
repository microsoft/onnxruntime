// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class QDQTransformer

Transformer that fuse QDQ and fp32 op to quantized op.
*/
class QDQTransformer : public GraphTransformer {
 public:
  QDQTransformer() noexcept : GraphTransformer("QDQTransformer") {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
