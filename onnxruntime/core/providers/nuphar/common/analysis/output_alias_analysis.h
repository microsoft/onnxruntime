// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/codegen/common/common.h"
#include "core/providers/nuphar/common/analysis/analysis.h"
#include "core/graph/graph.h"

namespace onnxruntime {
namespace codegen {

class OutputAliasAnalysis : public OrtAnalysis {
 public:
  OutputAliasAnalysis()
      : OrtAnalysis("OutputAliasAnalysis") {}

  ~OutputAliasAnalysis() = default;

  void Evaluate(const onnxruntime::GraphViewer& graph) override;

  void EvaluateSingleNode(const onnxruntime::Node& node);

  bool IsOutputNode(const onnxruntime::Node* node) const;

  bool IsOutputAlias(const onnxruntime::Node* node) const;

  const onnxruntime::NodeArg* SourceDefOfOutputAlias(const onnxruntime::NodeArg* node) const;

 private:
  // a set for output nodes
  std::set<NodeKey> output_nodes_;
  // a map from an output alias to its input
  std::map<NodeKey, const onnxruntime::NodeArg*> alias_use_defs_;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OutputAliasAnalysis);
};

}  // namespace codegen
}  // namespace onnxruntime
