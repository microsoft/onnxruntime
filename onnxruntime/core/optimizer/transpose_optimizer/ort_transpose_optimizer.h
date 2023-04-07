// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class TransposeOptimizer
Push transposes through ops and eliminate them.
*/
class TransposeOptimizer : public GraphTransformer {
 private:
  AllocatorPtr cpu_allocator_;

 public:
  explicit TransposeOptimizer(AllocatorPtr cpu_allocator) noexcept
      : GraphTransformer("TransposeOptimizer"), cpu_allocator_(std::move(cpu_allocator)) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  // One run should be sufficient.
  // The second phase of optimization may swap a DequantizeLinear -> Transpose back, so multiple runs would
  // keep swapping the order of the nodes in the first and second phases, leading to always returning true for
  // modified.
  // see https://github.com/microsoft/onnxruntime/blob/e3a2d5cca8bcefe064f83d57e46ea51ddb2b16e8/onnxruntime/core/optimizer/transpose_optimizer/transpose_optimizer.cc#L1917-L1921
  bool ShouldOnlyApplyOnce() const override { return true; }
};

}  // namespace onnxruntime
