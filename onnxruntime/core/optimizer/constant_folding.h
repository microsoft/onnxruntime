// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/framework/ort_value.h"
#include <memory>
#include "core/framework/execution_provider.h"

namespace onnxruntime {

/**
@class ConstantFolding

Transformer that traverses the graph top-down and performs constant folding, i.e.,
it statically computes parts of the graph that rely only on constant initializers.
*/
class ConstantFolding : public GraphTransformer {
 public:
  /*! Constant folding will not be applied to nodes that have one of initializers from excluded_initializers as input.
      For pre-training, the trainable weights are those initializers to be excluded.
      \param execution_provider Execution provider instance to execute constant folding.
  */
  ConstantFolding(const IExecutionProvider& execution_provider,
                  bool skip_dequantize_linear,
                  const InlinedHashSet<std::string_view>& compatible_execution_providers = {},
                  const InlinedHashSet<std::string>& excluded_initializers = {}) noexcept;

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  bool skip_dequantize_linear_;
  const InlinedHashSet<std::string> excluded_initializers_;
  const IExecutionProvider& execution_provider_;
};

}  // namespace onnxruntime
