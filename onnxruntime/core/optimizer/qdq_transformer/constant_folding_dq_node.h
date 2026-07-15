// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/constant_folding.h"
#include "core/framework/ort_value.h"
#include <memory>
#include "core/framework/execution_provider.h"

namespace onnxruntime {

/**
@class ConstantFoldingDQ

It's the derived class from ConstantFolding.
*/
class ConstantFoldingDQ : public ConstantFolding {
 public:
  /*! Constant folding will not be applied to nodes that have one of initializers from excluded_initializers as input.
      \param execution_provider Execution provider instance to execute constant folding.
  */
  ConstantFoldingDQ(const IExecutionProvider& execution_provider,
                    bool skip_dequantize_linear,
                    const ConfigOptions& config_options,
                    const InlinedHashSet<NodeIndex>& node_index_set,
                    const InlinedHashSet<std::string_view>& compatible_execution_providers = {},
                    const InlinedHashSet<std::string>& excluded_initializers = {}) noexcept;

  bool AllowConstantFolding(const Node& node) const override;

 private:
  InlinedHashSet<NodeIndex> node_index_set_;
};

}  // namespace onnxruntime
