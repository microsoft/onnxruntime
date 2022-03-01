// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/codegen/common/common.h"
#include "core/graph/graph.h"
#include "core/providers/nuphar/common/analysis/analysis.h"
#include <gsl/gsl>

namespace onnxruntime {
namespace nuphar {

class OutputAliasAnalysis : public NupharAnalysis {
 public:
  OutputAliasAnalysis()
      : NupharAnalysis("OutputAliasAnalysis") {}

  ~OutputAliasAnalysis() = default;

  void Evaluate(const onnxruntime::nuphar::NupharSubgraphUnit& graph) override;

  bool IsOutputNode(const onnxruntime::Node* node) const;

  bool IsOutputAlias(const onnxruntime::Node* node) const;

  const onnxruntime::NodeArg* SourceDefOfOutputAlias(const onnxruntime::NodeArg* node) const;

 private:
  // a set for output nodes
  std::set<NodeKey> output_nodes_;
  // a map from an output alias to its input
  std::map<NodeKey, const onnxruntime::NodeArg*> alias_use_defs_;

  void Traverse(gsl::span<const Node* const> nodes,
                const InlinedHashSet<std::string_view>& graph_inputs,
                const InlinedHashSet<std::string_view>& graph_outputs);

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OutputAliasAnalysis);
};

}  // namespace nuphar
}  // namespace onnxruntime
