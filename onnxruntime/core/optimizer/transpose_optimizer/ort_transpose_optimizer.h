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
};

}  // namespace onnxruntime
