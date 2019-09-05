// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/common/analysis/subgraph_partition_stats.h"

#include "core/providers/nuphar/common/analysis/use_count_analysis.h"

namespace onnxruntime {
namespace nuphar {

// TODO: Add memory analysis
// SubgraphPartitionStats has one analysis pass
// The first pass, offset as 0,  is UseCountAnalysis
constexpr int UseCountAnalysisOffset = 0;

void SubgraphPartitionStats::SetShapeInference(
    const std::shared_ptr<ShapeExprContext>& shape_infernece) {
  passes_.clear();
  passes_.emplace_back(std::make_shared<OrtUseCountAnalysis>(shape_infernece));
}

int SubgraphPartitionStats::NodeUseCount(const onnxruntime::Node* node) const {
  ORT_ENFORCE(passes_.size() > UseCountAnalysisOffset);
  return Promote<OrtUseCountAnalysis>(passes_[UseCountAnalysisOffset])->NodeUseCount(node);
}

}  // namespace nuphar
}  // namespace onnxruntime
