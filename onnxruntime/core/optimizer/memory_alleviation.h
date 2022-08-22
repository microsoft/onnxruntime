// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

namespace memory_alleviation {

enum AlleviationStratagy {
  None = 0,
  Recompute = 1,
  OffloadToCPUMemory = 2,
};

AlleviationStratagy ParseFromIntFlag(const int32_t& flag_int);

}  // namespace memory_alleviation

/**
@Class MemoryAlleviation

Set priority for recomputed nodes for example: Gelu/BiasGelu/FastGelu.

*/
class MemoryAlleviation : public GraphTransformer {
 public:
  MemoryAlleviation(const int32_t enable_gelu_recompute,
                             const int32_t enable_dropout_recompute,
                             const int32_t enable_tile_recompute) noexcept
      : GraphTransformer("MemoryAlleviation"),
        enable_gelu_recompute_{enable_gelu_recompute},
        enable_dropout_recompute_{enable_dropout_recompute},
        enable_tile_recompute_{enable_tile_recompute} {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  bool ShouldOnlyApplyOnce() const override { return true; }

 private:
  int32_t enable_gelu_recompute_;
  int32_t enable_dropout_recompute_;
  int32_t enable_tile_recompute_;
};

}  // namespace onnxruntime
