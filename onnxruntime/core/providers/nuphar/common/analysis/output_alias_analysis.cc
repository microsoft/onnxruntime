// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/common/analysis/output_alias_analysis.h"

#include "core/codegen/common/common.h"

namespace onnxruntime {
namespace nuphar {

void OutputAliasAnalysis::Traverse(gsl::span<const Node* const> nodes,
                                   const InlinedHashSet<std::string_view>& graph_inputs,
                                   const InlinedHashSet<std::string_view>& graph_outputs) {
  for (auto& node : nodes) {
    if (node->NodeType() == Node::Type::Fused) {
      // unboxing of fused node
      const auto& func_body = GraphViewer(node->GetFunctionBody()->Body());
      Traverse(ConvertGraphNodesToNodePtrs(func_body.Nodes()), graph_inputs, graph_outputs);
    } else {
      // TODO: change identity to other alias
      bool is_identity = (node->OpType() == "Identity");
      ORT_THROW_IF_ERROR(node->ForEachWithIndex(
          node->OutputDefs(),
          [&](const NodeArg& def, size_t) {
            if (graph_outputs.count(def.Name()) > 0) {
              NodeKey key = GetKey(node);
              output_nodes_.insert(key);
              if (is_identity) {
                auto input_def = node->InputDefs()[0];
                // regard as aliased if input_def is not graph input
                // otherwise, we still generate Identity ops in TVM
                // TODO: remove once we have a better solution for alias optimization
                if (graph_inputs.count(input_def->Name()) == 0) {
                  alias_use_defs_.insert(std::make_pair(key, input_def));
                  NodeKey input_key = GetKey(input_def);
                  output_nodes_.insert(input_key);
                }
              }
            }
            return Status::OK();
          }));
    }
  }
}

// TODO: please reimplement output alias using the right algorithm.
// Currently we only copy it from old graph_stats, which is still wrong one
void OutputAliasAnalysis::Evaluate(const onnxruntime::nuphar::NupharSubgraphUnit& graph) {
  if (graph.IsSingleNode()) {
    const Node* node = graph.nodes.front();
    auto subgraph = GetSubgraph(*node);

    if (nullptr != subgraph) {
      const auto& graph_viewer = GraphViewer(*subgraph);
      InlinedHashSet<std::string_view> graph_inputs;
      InlinedHashSet<std::string_view> graph_outputs;
      graph_inputs.reserve(graph_viewer.GetInputs().size());
      graph_outputs.reserve(graph_viewer.GetOutputs().size());
      for (const auto* def : graph_viewer.GetInputs()) {
        if (nullptr != def) {
          graph_inputs.insert(def->Name());
        }
      }
      for (const auto* def : graph_viewer.GetOutputs()) {
        if (nullptr != def) {
          graph_outputs.insert(def->Name());
        }
      }
      Traverse(ConvertGraphNodesToNodePtrs(graph_viewer.Nodes()), graph_inputs, graph_outputs);
    } else {
      NodeKey key = GetKey(node);
      output_nodes_.insert(key);
    }
  } else {
    // outputs names
    InlinedHashSet<std::string_view> graph_inputs;
    InlinedHashSet<std::string_view> graph_outputs;
    graph_inputs.reserve(graph.inputs.size());
    graph_outputs.reserve(graph.outputs.size());
    for (const auto* def : graph.inputs) {
      if (nullptr != def) {
        graph_inputs.insert(def->Name());
      }
    }
    for (const auto* def : graph.outputs) {
      if (nullptr != def) {
        graph_outputs.insert(def->Name());
      }
    }
    Traverse(graph.nodes, graph_inputs, graph_outputs);
  }
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
  if (iter != alias_use_defs_.end()) {
    return iter->second;
  }
  return nullptr;
}

}  // namespace nuphar
}  // namespace onnxruntime
