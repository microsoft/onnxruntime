// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "graph_partition_stats.h"
#include "use_count_analysis.h"

namespace onnxruntime {
namespace codegen {

// This file contains old contents from old GraphStats.
// It will be removed after refactoring step 13
// So no need to do detailed review.

// TODO: Add memory analysis

// GraphPartitionStats has one analysis pass
// The first pass, offset as 0,  is UseCountAnalysis
constexpr int UseCountAnalysisOffset = 0;

void GraphPartitionStats::SetShapeInference(
    const std::shared_ptr<ShapeExprContext>& shape_infernece) {
  passes_.clear();
  auto use_count_pass = std::make_shared<UseCountAnalysis>(shape_infernece);
  passes_.push_back(use_count_pass);
}

int GraphPartitionStats::NodeUseCount(const onnxruntime::Node* node) const {
  ORT_ENFORCE(passes_.size() > UseCountAnalysisOffset);
  return Promote<UseCountAnalysis>(passes_[UseCountAnalysisOffset])->NodeUseCount(node);
}

}  // namespace codegen
}  // namespace onnxruntime
