// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class NchwcTransformer : public onnxruntime::GraphTransformer {
 public:
  NchwcTransformer() noexcept : onnxruntime::GraphTransformer("NchwcTransformer") {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override;
};

}  // namespace onnxruntime
