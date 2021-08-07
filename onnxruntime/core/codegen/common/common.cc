// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/common/common.h"

#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph.h"
#include "core/graph/schema_registry.h"
#include <algorithm>
#include <unordered_set>

namespace onnxruntime {

NodeKey GetKey(const onnxruntime::Node* node) {
  ORT_ENFORCE(nullptr != node);
  ORT_ENFORCE(node->OutputDefs().size() > 0);
  return node->OutputDefs()[0]->Name();
}

NodeKey GetKey(const onnxruntime::Node& node) {
  ORT_ENFORCE(node.OutputDefs().size() > 0);
  return node.OutputDefs()[0]->Name();
}

NodeKey GetKey(const onnxruntime::NodeArg* def) {
  // NodeArg's name is unique.
  ORT_ENFORCE(nullptr != def);
  return def->Name();
}

bool IsRecurrentNode(const onnxruntime::Node& node) {
  auto op_type = node.OpType();
  return (op_type == "LSTM" || op_type == "RNN" || op_type == "GRU" ||
          op_type == "Scan" || op_type == "Loop");
}

bool IsAliasNode(const onnxruntime::Node& node) {
  auto op_type = node.OpType();
  if (op_type == "Transpose") {
    // Treat Transpose (1,N) -> (N,1) as Alias
    const auto shape = node.OutputDefs()[0]->Shape();
    if (shape != nullptr && shape->dim_size() == 2) {
      for (int i = 0; i < 2; ++i) {
        if (shape->dim(i).has_dim_value() && shape->dim(i).dim_value() == 1) {
          return true;
        }
      }
    }
    return false;
  }

  return (op_type == "Flatten" || op_type == "Identity" || op_type == "Reshape" ||
          op_type == "Squeeze" || op_type == "Unsqueeze");
}

std::string NormalizeCppName(const std::string& name) {
  std::string normalized_name = name;
  for (char c : {'.', ' ', '+', '-', '*', '/', '\\', '='})
    std::replace(normalized_name.begin(), normalized_name.end(), c, '_');
  return normalized_name;
}

std::string NormalizeNodeArgName(const NodeArg* def) {
  return NormalizeCppName(def->Name());
}

bool IsFusedNode(const Node& node) {
  if (node.NodeType() == Node::Type::Fused) {
    return true;
  }
  return false;
}

// A unified API to get Subgraph
const Graph* GetSubgraph(const Node& node) {
  if (node.NodeType() == Node::Type::Fused) {
    return &(node.GetFunctionBody()->Body());
  } else if (node.OpType() == "Scan") {
    return node.GetGraphAttribute("body");
  }
  // return nullptr implying no subgraph
  return nullptr;
}

bool HasLoop(const Node& node) {
  auto op_type = node.OpType();
  if (op_type == "LSTM" ||
      op_type == "GRU" ||
      op_type == "RNN" ||
      op_type == "Scan") {
    return true;
  }
  return false;
}

// Return the corresponding input node for the NodeArg of the given node
const onnxruntime::Node* GetInputNode(const Node& node, const NodeArg* def) {
  const auto& input_name = def->Name();
  const onnxruntime::Node* input_node = nullptr;
  // search input node set to see if input_name is in their outputs (weights are not from node)
  for (auto iter = node.InputNodesBegin(); iter != node.InputNodesEnd(); ++iter) {
    const onnxruntime::Node& p = *iter;
    bool found = false;
    p.ForEachWithIndex(
        p.OutputDefs(),
        [&found, &input_name](const onnxruntime::NodeArg& out_def, size_t) {
          if (input_name == out_def.Name()) {
            found = true;
          }
          return Status::OK();
        });
    if (found)
      input_node = &p;
  }
  return input_node;
}

// create capacity from subgraph
std::unique_ptr<ComputeCapability> ToCapacity(const onnxruntime::GraphViewer& graph,
                                              int fused_count,
                                              std::unique_ptr<IndexedSubGraph>& subgraph) {
  auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
  meta_def->name = "Fuse" + std::to_string(fused_count);
  meta_def->domain = "Fuse";

  std::set<NodeIndex> node_indices(subgraph->nodes.begin(), subgraph->nodes.end());

  const auto& start_node_index = subgraph->nodes.front();
  const auto& start_node = *graph.GetNode(start_node_index);
  const auto& end_node_index = subgraph->nodes.back();
  const auto& end_node = *graph.GetNode(end_node_index);
  meta_def->name += start_node.OpType() + std::to_string(start_node_index);
  meta_def->name += "_With" + std::to_string(subgraph->nodes.size()) + "Nodes_";
  meta_def->name += end_node.OpType() + std::to_string(end_node_index);

  std::unordered_set<std::string> real_output_names;
  for (const auto* def : graph.GetOutputs()) {
    real_output_names.insert(def->Name());
  }

  for (const auto& node_index : subgraph->nodes) {
    const auto& node = *graph.GetNode(node_index);
    auto process_input_fn =
        [&meta_def, &node, &node_indices](const onnxruntime::NodeArg& def, size_t) {
          const onnxruntime::Node* input_node = GetInputNode(node, &def);
          bool input_from_subgraph = (input_node && node_indices.count(input_node->Index()));
          if (!input_from_subgraph) {
            // input is from weights or outside of graph
            meta_def->inputs.push_back(def.Name());
          }
          return Status::OK();
        };
    // handle current graph's inputs
    node.ForEachWithIndex(node.InputDefs(), process_input_fn);
    // nodes' implicit inputs also need to be collected. They need to
    // be promoted to being explicit inputs for everything to work.
    node.ForEachWithIndex(node.ImplicitInputDefs(), process_input_fn);

    // Handle outouts
    // two cases are considerd as outputs
    // 1. Output NodeArg is not used by any Node
    // 2. Output NodeArg is used by at least one Node out of this subgraph.
    //    Note a NodeArg can be used by Nodes in and out of the subgraph at the same time.
    // 3. Output NodeArg is one of real outputs of an Ort graph.

    auto InsertOutputToSubgraph = [&meta_def](const NodeArg* def) {
      if (std::find(meta_def->outputs.begin(), meta_def->outputs.end(), def->Name()) ==
          meta_def->outputs.end()) {
        meta_def->outputs.push_back(def->Name());
      }
    };

    std::unordered_set<std::string> input_names_from_the_output_node;

    for (auto o_iter = node.OutputEdgesBegin(); o_iter != node.OutputEdgesEnd(); ++o_iter) {
      const auto& p = *o_iter;
      const Node& out_node = p.GetNode();

      // preprocess for the case 1
      out_node.ForEachWithIndex(
          out_node.InputDefs(),
          [&input_names_from_the_output_node](const onnxruntime::NodeArg& in_def, size_t) {
            input_names_from_the_output_node.insert(in_def.Name());
            return Status::OK();
          });

      // handle the case 2
      if (node_indices.count(out_node.Index()) == 0) {
        const NodeArg* def = node.OutputDefs()[p.GetSrcArgIndex()];
        InsertOutputToSubgraph(def);
      }
    }

    // handle case 1 and 3
    node.ForEachWithIndex(
        node.OutputDefs(),
        [&](const onnxruntime::NodeArg& def, size_t) {
          if (input_names_from_the_output_node.count(def.Name()) == 0 ||
              real_output_names.count(def.Name()) > 0) {
            InsertOutputToSubgraph(&def);
          }
          return Status::OK();
        });
  }

  // Handle subgraph's initializers
  const auto& all_initializers = graph.GetAllInitializedTensors();
  for (const auto& node_index : subgraph->nodes) {
    const auto& node = *graph.GetNode(node_index);
    // check whether it is an immediate nested subgraph
    auto immediate_nested_subgraph = GetSubgraph(node);
    // If so, copy the immediate nested subgraph's initializers to meta_def->inputs.
    // Note we don't need recursion here, since Ort did recursion for us by handling subgraph early than the current graph.
    // Therefore, the all inner nested subgraph's initializers should be already in the immediate nested subgraph's inputs.
    if (nullptr != immediate_nested_subgraph) {
      for (auto& n : immediate_nested_subgraph->Nodes()) {
        auto add_input_fn =
            [&meta_def, &all_initializers](const onnxruntime::NodeArg& def, size_t) {
              auto iter = all_initializers.find(def.Name());
              if (iter != all_initializers.end()) {
                meta_def->inputs.push_back(def.Name());
              }
              return Status::OK();
            };
        n.ForEachWithIndex(n.InputDefs(), add_input_fn);
        n.ForEachWithIndex(n.ImplicitInputDefs(), add_input_fn);
      }
    }
  }

  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  std::unique_ptr<IndexedSubGraph> finished_subgraph(subgraph.release());
  finished_subgraph->SetMetaDef(std::move(meta_def));
  return std::make_unique<ComputeCapability>(std::move(finished_subgraph));
}

int64_t ShapeRank(const NodeArg* def) {
  ORT_ENFORCE_DEBUG(nullptr != def);
  return gsl::narrow_cast<int64_t>(def->Shape()->dim_size());
}

bool ShapeHasValue(const NodeArg* def, int i) {
  ORT_ENFORCE_DEBUG(nullptr != def);
  ORT_ENFORCE_DEBUG(i >= 0);
  ORT_ENFORCE_DEBUG(i < def->Shape()->dim_size());
  return utils::HasDimValue(def->Shape()->dim(i));
}

bool ShapeHasSymbol(const NodeArg* def, int i) {
  ORT_ENFORCE_DEBUG(nullptr != def);
  ORT_ENFORCE_DEBUG(i >= 0);
  ORT_ENFORCE_DEBUG(i < def->Shape()->dim_size());
  return utils::HasDimParam(def->Shape()->dim(i));
}

int64_t ShapeValue(const NodeArg* def, int i) {
  ORT_ENFORCE_DEBUG(ShapeHasValue(def, i));
  return def->Shape()->dim(i).dim_value();
}

const std::string& ShapeSymbol(const NodeArg* def, int i) {
  ORT_ENFORCE_DEBUG(ShapeHasSymbol(def, i));
  return def->Shape()->dim(i).dim_param();
}

ONNX_NAMESPACE::TensorProto_DataType TensorProtoDataType(const NodeArg* def) {
  ORT_ENFORCE_DEBUG(nullptr != def);
  return static_cast<ONNX_NAMESPACE::TensorProto_DataType>(def->TypeAsProto()->tensor_type().elem_type());
}

// Convert GraphNodes to internal NodePtrs without check lifetime.
// Please use it only locally when GraphNodes still exist
std::vector<const Node*> ConvertGraphNodesToNodePtrs(const ConstGraphNodes& graph_nodes) {
  std::vector<const Node*> nodes;
  for (auto& node : graph_nodes) {
    nodes.push_back(&node);
  }
  return nodes;
}

}  // namespace onnxruntime
