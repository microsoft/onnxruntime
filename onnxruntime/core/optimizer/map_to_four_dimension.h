// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/framework/ort_value.h"
#include <memory>
#include "core/framework/execution_provider.h"

namespace onnxruntime {

/**
 * Map 2D, 3D, 5D and 6D tensor to 4D tensor as some hardware platform only supports 4D tensor operations.
 * 
 * This transformer mainly has two graph transformations that apply to specific patterns: (Note: Not a very generic solution)
 *   1. Replace 2D Gemms with Transpose/Reshape and 1x1 Conv.
 *   2. Replace Reshape and ReduceSum with two Slices, two ReduceSums and one Concat.
 * 
 */
class MapToFourDimensions : public GraphTransformer {
 public:
  MapToFourDimensions() noexcept;

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
