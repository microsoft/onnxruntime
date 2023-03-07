// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@class ShapeOptimizer

Transformer that traverses the graph top-down and performs shape optimizations.

Try the best effort to constant fold the shape related to Shape node outputs:
  1. Shape generate 1D tensor [12, 128, 512] -> Slice(start=0,end=3), we can constant fold the Shape->Slice to an
  initializer including 1D tensor values [12, 128, 512]. (Some logic of ConstantFolding also does the same thing.)

  2. Shape generate 1D tensor [batch_size, 128, 512] -> Slice(start=1,end=3), we can constant fold the Shape->Slice to
  an initializer including 1D tensor values [128, 512].

  3. Shape generate 1D tensor [batch_size, 128, 512] -> Gather(axes=[0], index=[2]), we can constant fold the
  Shape->Gather to an initializer including 1D tensor values [512].

  4. Shape 15 takes input of shape [batch_size, 128, 512], slicing from 1 to 2(exclusive), we can constant fold the
  Shape15(start=1,end=2) to an initializer including 1D tensor values [128].

This would help clean up the graph, combined with ConstantFolding, the graph would be much more simplified.

*/
class ShapeOptimizer : public GraphTransformer {
 public:
  ShapeOptimizer(
      const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("ShapeOptimizer", compatible_execution_providers) {
  }

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
