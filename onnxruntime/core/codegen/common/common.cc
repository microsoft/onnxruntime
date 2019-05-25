// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/common/common.h"

#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph.h"
#include "core/graph/schema_registry.h"
#include <algorithm>

namespace onnxruntime {

NodeKey GetKey(const onnxruntime::Node* node) {
  ORT_ENFORCE(nullptr != node);
  ORT_ENFORCE(node->OutputDefs().size() > 0);
  return node->OutputDefs()[0]->Name();
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
                                              std::unique_ptr<IndexedSubGraph>& subgraph) {
  auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
  static int fuse_count = 0;
  meta_def->name = "Fuse" + std::to_string(fuse_count++);
  meta_def->domain = "Fuse";

  std::set<NodeIndex> node_indices(subgraph->nodes.begin(), subgraph->nodes.end());

  const auto& start_node_index = subgraph->nodes.front();
  const auto& start_node = *graph.GetNode(start_node_index);
  const auto& end_node_index = subgraph->nodes.back();
  const auto& end_node = *graph.GetNode(end_node_index);
  meta_def->name += start_node.OpType() + std::to_string(start_node_index);
  meta_def->name += "_With" + std::to_string(subgraph->nodes.size()) + "Nodes_";
  meta_def->name += end_node.OpType() + std::to_string(end_node_index);

  for (const auto& node_index : subgraph->nodes) {
    const auto& node = *graph.GetNode(node_index);
    node.ForEachWithIndex(
        node.InputDefs(),
        [&meta_def, &node, &node_indices](const onnxruntime::NodeArg& def, size_t) {
          const onnxruntime::Node* input_node = GetInputNode(node, &def);
          bool input_from_subgraph = (input_node && node_indices.count(input_node->Index()));
          if (!input_from_subgraph) {
            // input is from weights or outside of graph
            meta_def->inputs.push_back(def.Name());
          }
          return Status::OK();
        });

    // add outputs to nodes outside of subgraph to meta_def and tvm_outputs
    node.ForEachWithIndex(
        node.OutputDefs(),
        [&meta_def, &node, &node_indices](const onnxruntime::NodeArg& def, size_t) {
          const auto& output_name = def.Name();
          // search output node set to see if output_name is in their inputs
          const onnxruntime::Node* output_node = nullptr;
          for (auto iter = node.OutputNodesBegin(); iter != node.OutputNodesEnd(); ++iter) {
            const onnxruntime::Node& p = *iter;
            bool found = false;
            p.ForEachWithIndex(
                p.InputDefs(),
                [&found, &output_name](const onnxruntime::NodeArg& out_def, size_t) {
                  if (output_name == out_def.Name()) {
                    found = true;
                  }
                  return Status::OK();
                });
            if (found)
              output_node = &p;
          }
          bool output_to_subgraph = output_node && node_indices.count(output_node->Index());
          if (!output_to_subgraph) {
            meta_def->outputs.push_back(def.Name());
          }
          return Status::OK();
        });
  }
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  std::unique_ptr<IndexedSubGraph> finished_subgraph(subgraph.release());
  finished_subgraph->SetMetaDef(meta_def);
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
  return def->Shape()->dim(i).has_dim_value();
}

bool ShapeHasSymbol(const NodeArg* def, int i) {
  ORT_ENFORCE_DEBUG(nullptr != def);
  ORT_ENFORCE_DEBUG(i >= 0);
  ORT_ENFORCE_DEBUG(i < def->Shape()->dim_size());
  return def->Shape()->dim(i).has_dim_param();
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

}  // namespace onnxruntime
