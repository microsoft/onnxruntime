// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/nuphar/common/analysis/graph_stats.h"
#include "use_count_analysis.h"
#include "core/codegen/common/common.h"

namespace onnxruntime {
namespace codegen {

// TODO: rename class name to more target-specific in the tvm refactoring
// Maybe SubGraphStatsX86
class SubGraphStats : public OrtGraphStats {
 public:
  SubGraphStats(const std::shared_ptr<ShapeExprContext>& shape_infernece);

  ~SubGraphStats() = default;

  void EvaluateSingleNode(const onnxruntime::Node& node);

  int NodeUseCount(const onnxruntime::Node* node) const;

  bool IsOutputNode(const onnxruntime::Node* node) const;

  bool IsOutputAlias(const onnxruntime::Node* node) const;

  const onnxruntime::NodeArg* SourceDefOfOutputAlias(const onnxruntime::NodeArg* node) const;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SubGraphStats);
};

}  // namespace codegen
}  // namespace onnxruntime
