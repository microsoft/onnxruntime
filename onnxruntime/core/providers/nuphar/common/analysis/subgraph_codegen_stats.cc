// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/common/analysis/subgraph_codegen_stats.h"

#include "core/providers/nuphar/common/analysis/output_alias_analysis.h"
#include "core/providers/nuphar/common/analysis/use_count_analysis.h"

namespace onnxruntime {
namespace nuphar {

// CodeGenUnitStats has two analysis passes
// The first pass, offset as 0,  is UseCountAnalysis
// The second pass, offset as 1,  is OutputAliasAnalysis
constexpr int UseCountAnalysisOffset = 0;
constexpr int OutputAliasAnalysisOffset = 1;

// True reuse count for cheap Op
constexpr int CheapNodeTrueReuseCount = 2;

// Constructor
CodeGenUnitStats::CodeGenUnitStats(
    const std::shared_ptr<ShapeExprContext>& shape_infernece)
    : NupharSubgraphUnitStats("CodeGenUnitStats") {
  auto use_count_pass = std::make_shared<NupharUseCountAnalysis>(shape_infernece);
  passes_.push_back(use_count_pass);

  auto output_alias_pass = std::make_shared<OutputAliasAnalysis>();
  passes_.push_back(output_alias_pass);
}

int CodeGenUnitStats::NodeUseCount(const onnxruntime::Node* node) const {
  ORT_ENFORCE(passes_.size() > UseCountAnalysisOffset);
  return Promote<NupharUseCountAnalysis>(passes_[UseCountAnalysisOffset])->NodeUseCount(node);
}

bool CodeGenUnitStats::IsCheapNodeReuse(const onnxruntime::Node* node) const {
  ORT_ENFORCE(passes_.size() > UseCountAnalysisOffset);

  // always inline(not reuse) alias node
  if (IsAliasNode(*node))
    return false;

  // always inline tensor ops that do not involve computation
  static const std::unordered_set<std::string> trivial_tensor_ops = {
      "Cast",
      "Concat",
      "Crop",
      "Expand",
      "Gather",
      "GatherElements",
      "Pad",
      "Shape",
      "Slice",
      "Split",
      "Tile",
      "Transpose",
  };
  if (trivial_tensor_ops.count(node->OpType()) > 0)
    return false;

  // always inline cheap compute ops if reuse count is low
  static const std::unordered_set<std::string> trivial_compute_ops = {
      "Abs",
      "Add",
      "Ceil",
      "Clip",
      "Equal",
      "Floor",
      "Greater",
      "Less",
      "Mul",
      "Neg",
      "Relu",
      "Sub",
      "Where",
  };
  if (trivial_compute_ops.count(node->OpType()) > 0)
    return Promote<NupharUseCountAnalysis>(passes_[UseCountAnalysisOffset])->NodeUseCount(node) > CheapNodeTrueReuseCount;

  // Otherwise return true and use count is determined by NodeUseCount
  return true;
}

bool CodeGenUnitStats::IsOutputNode(const onnxruntime::Node* node) const {
  ORT_ENFORCE(passes_.size() > OutputAliasAnalysisOffset);
  return Promote<OutputAliasAnalysis>(passes_[OutputAliasAnalysisOffset])->IsOutputNode(node);
}

bool CodeGenUnitStats::IsOutputAlias(const onnxruntime::Node* node) const {
  ORT_ENFORCE(passes_.size() > OutputAliasAnalysisOffset);
  return Promote<OutputAliasAnalysis>(passes_[OutputAliasAnalysisOffset])->IsOutputAlias(node);
}

const onnxruntime::NodeArg* CodeGenUnitStats::SourceDefOfOutputAlias(const onnxruntime::NodeArg* node) const {
  ORT_ENFORCE(passes_.size() > OutputAliasAnalysisOffset);
  return Promote<OutputAliasAnalysis>(passes_[OutputAliasAnalysisOffset])->SourceDefOfOutputAlias(node);
}

}  // namespace nuphar
}  // namespace onnxruntime
