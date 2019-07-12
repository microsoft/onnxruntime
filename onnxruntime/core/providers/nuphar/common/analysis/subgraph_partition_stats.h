// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/codegen/common/common.h"
#include "core/providers/nuphar/common/analysis/graph_stats.h"
#include "core/providers/nuphar/compiler/traverse_shape_infer.h"

namespace onnxruntime {
namespace nuphar {

// TODO: rename class name to more target-specific in the tvm refactoring
// Maybe GraphPartitionStatsX86
class SubgraphPartitionStats : public OrtGraphStats {
 public:
  SubgraphPartitionStats()
      : OrtGraphStats("SubgraphPartitionStats") {}

  ~SubgraphPartitionStats() = default;

  void SetShapeInference(const std::shared_ptr<ShapeExprContext>& shape_infernece);

  int NodeUseCount(const onnxruntime::Node* node) const;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SubgraphPartitionStats);
};

}  // namespace nuphar
}  // namespace onnxruntime
