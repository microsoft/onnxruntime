// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "output_alias_analysis.h"

namespace onnxruntime {
namespace codegen {

// This file contains old contents from old GraphStats.
// It will be removed after refactoring step 13
// So no need to do detailed review.

// TODO: please reimplement output alias using the right algorithm
// Currently we only copy it from old graph_stats, which is still wrong one
void OutputAliasAnalysis::Evaluate(const onnxruntime::GraphViewer& graph) {
  // outputs names
  std::vector<std::string> graph_outputs;
  for (const auto& def : graph.GetOutputs())
    if (def)
      graph_outputs.push_back(def->Name());

  using TraverseFunc = std::function<void(const onnxruntime::GraphViewer&)>;
  TraverseFunc traverse = [&](const onnxruntime::GraphViewer& g) {
    for (auto& node : g.Nodes()) {
      if (node.NodeType() == Node::Type::Fused) {
        // unboxing of fused node
        traverse(GraphViewer(node.GetFunctionBody()->Body()));
      } else {
        // TODO: change identity to other alias
        bool is_identity = (node.OpType() == "Identity");
        node.ForEachWithIndex(
            node.OutputDefs(),
            [this, &graph_outputs, &node, &is_identity](const NodeArg& def, size_t) {
              if (std::find(graph_outputs.begin(), graph_outputs.end(), def.Name()) != graph_outputs.end()) {
                NodeKey key = GetKey(&node);
                output_nodes_.insert(key);
                if (is_identity) {
                  auto input_def = node.InputDefs()[0];
                  alias_use_defs_.insert(std::make_pair(key, input_def));
                  NodeKey input_key = GetKey(input_def);
                  output_nodes_.insert(input_key);
                }
              }
              return Status::OK();
            });
      }
    }
  };
  traverse(graph);
}

void OutputAliasAnalysis::EvaluateSingleNode(const onnxruntime::Node& node) {
  NodeKey key = GetKey(&node);
  output_nodes_.insert(key);
}

bool OutputAliasAnalysis::IsOutputNode(const onnxruntime::Node* node) const {
  return output_nodes_.count(GetKey(node)) != 0;
}

bool OutputAliasAnalysis::IsOutputAlias(const onnxruntime::Node* node) const {
  auto key = GetKey(node);
  return alias_use_defs_.count(key) != 0;
}

const onnxruntime::NodeArg*
OutputAliasAnalysis::SourceDefOfOutputAlias(const onnxruntime::NodeArg* node) const {
  auto iter = alias_use_defs_.find(GetKey(node));
  if (iter != alias_use_defs_.end())
    return iter->second;
  return nullptr;
}

}  // namespace codegen
}  // namespace onnxruntime
