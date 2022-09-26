// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class BiasSoftmaxDropoutFusion
Fuse (x)->BiasSoftmax->Dropout->(y)    (dy)->DropoutGrad->SoftmaxGrad->(dx)
                 |                                           |
                 ---------------------------------------------
to
Fuse (x)->BiasSoftmaxDropout->(y)    (dy)->SoftmaxDropoutGrad->(dx)
                        |                     |
                        -----------------------
*/
class BiasSoftmaxDropoutFusion : public GraphTransformer {
 public:
  BiasSoftmaxDropoutFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("BiasSoftmaxDropoutFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
