// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/constants.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

// Removes casts around CUDA SimplifiedLayerNormalization without changing FP32 arithmetic.
class BFloat16NormCastFusion : public GraphTransformer {
 public:
  BFloat16NormCastFusion(const InlinedHashSet<std::string>& excluded_initializers = {}) noexcept
      : GraphTransformer("BFloat16NormCastFusion", {kCudaExecutionProvider}),
        excluded_initializers_(excluded_initializers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  const InlinedHashSet<std::string> excluded_initializers_;
};

}  // namespace onnxruntime
