// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/common/common.h"
#include "core/providers/nuphar/common/analysis/graph_stats.h"
#include "core/providers/nuphar/common/analysis/use_count_analysis.h"
#include "core/providers/nuphar/common/nuphar_subgraph.h"

namespace onnxruntime {
namespace nuphar {

class CodeGenUnitStats : public NupharSubgraphUnitStats {
 public:
  CodeGenUnitStats(const std::shared_ptr<ShapeExprContext>& shape_infernece);

  ~CodeGenUnitStats() = default;

  int NodeUseCount(const onnxruntime::Node* node) const;

  bool IsCheapNodeReuse(const onnxruntime::Node* node) const;

  bool IsOutputNode(const onnxruntime::Node* node) const;

  bool IsOutputAlias(const onnxruntime::Node* node) const;

  const onnxruntime::NodeArg* SourceDefOfOutputAlias(const onnxruntime::NodeArg* node) const;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CodeGenUnitStats);
};

}  // namespace nuphar
}  // namespace onnxruntime
