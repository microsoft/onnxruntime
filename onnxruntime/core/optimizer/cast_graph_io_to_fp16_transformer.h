// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class CastGraphIOToFp16Transformer : public GraphTransformer {
 public:
  CastGraphIOToFp16Transformer(const InlinedHashSet<std::string_view>& compatible_execution_providers = {})
      : GraphTransformer("CastGraphIOToFp16Transformer", compatible_execution_providers) {}

 private:
  Status ApplyImpl(
      Graph& graph,
      bool& modified,
      int graph_level,
      const logging::Logger& logger) const override;
};
}  // namespace onnxruntime
