// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/quantize_dequantize_linear_fusion.h"

#include <algorithm>
#include <array>
#include <optional>
#include <string_view>

#include "core/graph/constants.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"

namespace onnxruntime {
namespace {

// Ops that only move or select elements of their first input, so they give the same values on dequantized data.
constexpr std::array<std::string_view, 13> kDataMovementOps = {
    "DepthToSpace", "Expand", "Flatten", "Gather", "Identity", "Reshape", "Slice",
    "SpaceToDepth", "Split", "Squeeze", "Tile", "Transpose", "Unsqueeze"};

int32_t ElemType(const NodeArg& arg) {
  const auto* type = arg.TypeAsProto();
  return type != nullptr && type->has_tensor_type() ? type->tensor_type().elem_type()
                                                    : ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
}

bool HasNoBlockSize(const Node& node) {
  const auto* block_size = graph_utils::GetNodeAttribute(node, "block_size");
  return block_size == nullptr || block_size->i() == 0;
}

bool IsDataMovementOp(const Node& node) {
  return node.Domain() == kOnnxDomain &&
         std::find(kDataMovementOps.begin(), kDataMovementOps.end(), node.OpType()) != kDataMovementOps.end();
}

bool IsFusableQ(const Graph& graph, const Node& q_node, const QDQ::GetConstantInitializerFn& get_constant_initializer,
                const InlinedHashSet<std::string_view>& compatible_eps) {
  if (!QDQ::MatchQNode(q_node) || !graph_utils::IsSupportedProvider(q_node, compatible_eps) ||
      !HasNoBlockSize(q_node) || q_node.GetOutputEdgesCount() == 0 || graph.NodeProducesGraphOutput(q_node)) {
    return false;
  }

  const auto& inputs = q_node.InputDefs();
  if (inputs.size() != QDQ::InputIndex::TOTAL_COUNT ||
      ElemType(*inputs[QDQ::InputIndex::INPUT_ID]) != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
      ElemType(*inputs[QDQ::InputIndex::SCALE_ID]) != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    return false;
  }

  const int32_t quantized_type = ElemType(*inputs[QDQ::InputIndex::ZERO_POINT_ID]);
  if (quantized_type != ONNX_NAMESPACE::TensorProto_DataType_UINT16 &&
      quantized_type != ONNX_NAMESPACE::TensorProto_DataType_INT16) {
    return false;
  }

  // A positive scale that compares equal to the DQ scale is also bitwise equal to it.
  return QDQ::IsQOrDQScalePositiveConstantScalar(graph, q_node, get_constant_initializer, graph.ModelPath());
}

struct DQLeaf {
  Node* dq_node;
  NodeIndex producer;
  int producer_output;
};

struct QuantizedTree {
  // The DQ that produces a graph output, if any, is first.
  InlinedVector<DQLeaf> dq_leaves;
  // Each node's quantized input comes from the Q node or an earlier entry.
  InlinedVector<Node*> data_movement_nodes;
};

// Succeeds if every use of the Q output reaches a DQ with the same scale and zero point, directly or through
// data-movement ops. Only a direct DQ may produce a graph output, since the fused node takes over its output.
std::optional<QuantizedTree> CollectQuantizedTree(Graph& graph, const Node& q_node,
                                                  const QDQ::GetConstantInitializerFn& get_constant_initializer) {
  const int32_t quantized_type = ElemType(*q_node.InputDefs()[QDQ::InputIndex::ZERO_POINT_ID]);
  QuantizedTree tree;
  bool has_graph_output_dq = false;
  InlinedVector<const Node*> producers{&q_node};
  for (size_t i = 0; i < producers.size(); ++i) {
    const Node& producer = *producers[i];
    for (auto it = producer.OutputEdgesBegin(), end = producer.OutputEdgesEnd(); it != end; ++it) {
      Node* consumer = graph.GetNode(it->GetNode().Index());
      if (it->GetDstArgIndex() != 0 || consumer->GetExecutionProviderType() != q_node.GetExecutionProviderType()) {
        return std::nullopt;
      }

      if (QDQ::MatchDQNode(*consumer)) {
        if (!HasNoBlockSize(*consumer) ||
            ElemType(*consumer->OutputDefs()[0]) != ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
            !QDQ::IsQDQPairSupported(graph, q_node, *consumer, get_constant_initializer, graph.ModelPath(), false)) {
          return std::nullopt;
        }

        const DQLeaf leaf{consumer, producer.Index(), it->GetSrcArgIndex()};
        if (graph.NodeProducesGraphOutput(*consumer)) {
          if (&producer != &q_node || has_graph_output_dq) {
            return std::nullopt;
          }
          has_graph_output_dq = true;
          tree.dq_leaves.insert(tree.dq_leaves.begin(), leaf);
        } else {
          tree.dq_leaves.push_back(leaf);
        }
        continue;
      }

      if (!IsDataMovementOp(*consumer) || graph.NodeProducesGraphOutput(*consumer)) {
        return std::nullopt;
      }
      for (const NodeArg* output : consumer->OutputDefs()) {
        if (output->Exists() && ElemType(*output) != quantized_type) {
          return std::nullopt;
        }
      }
      tree.data_movement_nodes.push_back(consumer);
      producers.push_back(consumer);
    }
  }

  return tree;
}

NodeArg& MakeFloatNodeArg(Graph& graph, const NodeArg& like) {
  ONNX_NAMESPACE::TypeProto type = *like.TypeAsProto();
  type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  return graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(like.Name()), &type);
}

}  // namespace

Status QuantizeDequantizeLinearFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                                 const logging::Logger& logger) const {
  const auto get_constant_initializer = [&graph](const std::string& name) {
    return graph.GetConstantInitializer(name, true);
  };

  GraphViewer graph_viewer(graph);
  for (NodeIndex node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    Node* q_node = graph.GetNode(node_index);
    if (q_node == nullptr) {
      continue;
    }

    ORT_RETURN_IF_ERROR(Recurse(*q_node, modified, graph_level, logger));

    if (!IsFusableQ(graph, *q_node, get_constant_initializer, GetCompatibleExecutionProviders())) {
      continue;
    }

    std::optional<QuantizedTree> tree = CollectQuantizedTree(graph, *q_node, get_constant_initializer);
    if (!tree.has_value() || tree->dq_leaves.empty()) {
      continue;
    }

    // Reuse the output of a direct DQ (the graph output, if any) so its consumers keep their input.
    const DQLeaf& first = tree->dq_leaves.front();
    auto& q_inputs = q_node->MutableInputDefs();
    NodeArg& fused_output = first.producer == q_node->Index() ? *first.dq_node->MutableOutputDefs()[0]
                                                              : MakeFloatNodeArg(graph, *q_inputs[0]);
    Node& fused = graph.AddNode(graph.GenerateNodeName(q_node->Name() + "_QuantizeDequantizeLinear"),
                                "QuantizeDequantizeLinear", "Fused QuantizeLinear and DequantizeLinear",
                                {q_inputs[0], q_inputs[1], q_inputs[2]}, {&fused_output}, nullptr, kMSDomain);
    fused.SetExecutionProviderType(q_node->GetExecutionProviderType());

    // Disconnect the Q and DQ nodes while the edges still match the quantized node args.
    InlinedVector<NodeIndex> source_nodes{q_node->Index()};
    graph_utils::MoveAllNodeInputEdges(graph, *q_node, fused);
    graph_utils::RemoveNodeOutputEdges(graph, *q_node);
    for (const DQLeaf& leaf : tree->dq_leaves) {
      if (leaf.producer != q_node->Index()) {
        graph.RemoveEdge(leaf.producer, leaf.dq_node->Index(), leaf.producer_output, 0);
      }
      source_nodes.push_back(leaf.dq_node->Index());
    }

    // The data-movement nodes now run on the dequantized float values.
    InlinedHashMap<const NodeArg*, NodeArg*> float_args{{q_node->OutputDefs()[0], &fused_output}};
    for (Node* node : tree->data_movement_nodes) {
      NodeArg*& input = node->MutableInputDefs()[0];
      const bool reads_q_output = input == q_node->OutputDefs()[0];
      input = float_args.at(input);
      for (NodeArg*& output : node->MutableOutputDefs()) {
        if (output->Exists()) {
          NodeArg& float_output = MakeFloatNodeArg(graph, *output);
          float_args.emplace(output, &float_output);
          output = &float_output;
        }
      }
      if (reads_q_output) {
        graph.AddEdge(fused.Index(), node->Index(), 0, 0);
      }
    }

    for (const DQLeaf& leaf : tree->dq_leaves) {
      Node& producer = leaf.producer == q_node->Index() ? fused : *graph.GetNode(leaf.producer);
      graph_utils::ReplaceDownstreamNodeInput(graph, *leaf.dq_node, 0, producer, leaf.producer_output);
    }

    for (NodeIndex index : source_nodes) {
      graph.RemoveNode(index);
    }
    graph.NotifyNodeReplacement(source_nodes, fused.Index());

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
