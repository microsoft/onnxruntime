// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/matmul_transpose_fusion.h"
#include "core/graph/graph_utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

static bool GetTransposePerms(const Node& transpose_node, std::vector<int64_t>& perms) {
  ORT_ENFORCE(transpose_node.InputDefs().size() == 1);

  // use perms if present
  const auto perm_attr = transpose_node.GetAttributes().find("perm");
  if (perm_attr != transpose_node.GetAttributes().end()) {
    perms = RetrieveValues<int64_t>(perm_attr->second);
    return true;
  }

  // otherwise, reverse dimensions
  const NodeArg& input = *transpose_node.InputDefs()[0];
  const TensorShapeProto* shape = input.Shape();
  if (!shape) {
    return false;
  }

  perms.resize(shape->dim_size());
  std::iota(perms.rbegin(), perms.rend(), 0);
  return true;
}

static Node* GetTransposeNodeFromOutput(Graph& graph, NodeArg& node_arg) {
  Node* trans_node = graph.GetMutableProducerNode(node_arg.Name());
  if (trans_node == nullptr || trans_node->OpType() != "Transpose") {
    return nullptr;
  }

  // if the node has Graph output, skip it too
  if (!graph.GetNodeOutputsInGraphOutputs(*trans_node).empty()) {
    return nullptr;
  }

  std::vector<int64_t> perms;
  if (!GetTransposePerms(*trans_node, perms)) {
    return nullptr;
  }

  int64_t rank = perms.size();
  if (rank < 2) {
    return nullptr;
  }

  bool is_trans_on_last_two_dims = true;
  for (int64_t i = 0; i < rank - 2; i++) {
    if (perms[static_cast<size_t>(i)] != i) {
      is_trans_on_last_two_dims = false;
      break;
    }
  }

  if (is_trans_on_last_two_dims) {
    // rank is atleast 2 (checked above) and so it is safe to cast (rank - 2) and (rank - 1) to size_t
    is_trans_on_last_two_dims = perms[static_cast<size_t>(rank - 2)] == rank - 1 && perms[static_cast<size_t>(rank - 1)] == rank - 2;
  }

  if (!is_trans_on_last_two_dims) {
    return nullptr;
  }

  return trans_node;
}

static size_t UpdateConsumerCount(Graph& graph, NodeArg* target, std::unordered_map<NodeArg*, size_t>& count_map) {
  const auto& node_consumers = graph.GetConsumerNodes(target->Name());
  ORT_ENFORCE(!node_consumers.empty());
  auto it = count_map.find(target);
  if (it == count_map.end()) {
    count_map.insert({target, node_consumers.size() - 1});
    return node_consumers.size() - 1;
  } else {
    count_map[target] -= 1;
    return count_map[target];
  }
}

/* ReorderCastAndTranspose:
*  Interchange Cast and Transpose nodes in the graph and return the new Transpose node if possible else nullptr.
*
*
*  Transform the following pattern
*                              |
*                         _____|______
*                         |Transpose |
*                         |__________|
*                              |
*                              |
*                         _____V______
*                         |  Cast    |
*                         |__________|
*                              |
*                              V
*
*  to
*                              |
*                         _____|______
*                         |  Cast    |
*                         |__________|
*                              |
*                              |
*                         _____V______
*                         | Transpose|
*                         |__________|
*                              |
*                              V
*/
static Node* ReorderCastAndTranspose(Graph& graph, Node* cast,
                                    std::unordered_map<NodeArg*, size_t>& consumer_count,
                                    std::deque<onnxruntime::NodeIndex>& removed_nodes) {

  ORT_ENFORCE(cast != nullptr);
  auto transpose = GetTransposeNodeFromOutput(graph, *cast->MutableInputDefs()[0]);
  if (transpose == nullptr) {
    return nullptr;
  }
  NodeArg* cast_output = cast->MutableOutputDefs()[0];
  NodeArg* transpose_input = transpose->MutableInputDefs()[0];

  // Create a new NodeArg to feed the output from the new Cast to the new Transpose.
  // The shape of the new NodeArg is same as the original input to Transport but type
  // should match that of the output from the original Cast.

  auto new_cast_output_type_proto = *transpose_input->TypeAsProto();
  const ONNX_NAMESPACE::TensorProto_DataType element_type =
      static_cast<ONNX_NAMESPACE::TensorProto_DataType>(cast_output->TypeAsProto()->tensor_type().elem_type());
  new_cast_output_type_proto.mutable_tensor_type()->set_elem_type(element_type);
  auto& new_cast_output = graph.GetOrCreateNodeArg(cast_output->Name() + "_transformed", &new_cast_output_type_proto);

  const std::vector<NodeArg*> new_cast_input_defs {transpose_input};
  const std::vector<NodeArg*> new_cast_output_defs {&new_cast_output};
  const std::vector<NodeArg*> new_transpose_input_defs = {&new_cast_output};
  const std::vector<NodeArg*> new_transpose_output_defs = {cast_output};

  (void) graph.AddNode(graph.GenerateNodeName(cast->Name() + "_transformed"),
                                 cast->OpType(),
                                 "Created a new Cast node to interchange Cast and Transpose nodes",
                                 new_cast_input_defs,
                                 new_cast_output_defs,
                                 &cast->GetAttributes(),
                                 cast->Domain());

  Node& new_transpose = graph.AddNode(graph.GenerateNodeName(transpose->Name() + "_transformed"),
                                      transpose->OpType(),
                                      "Created a new Transpose node to interchange Cast and Transpose nodes",
                                      new_transpose_input_defs,
                                      new_transpose_output_defs,
                                      &transpose->GetAttributes(),
                                      transpose->Domain());

  size_t consumers = UpdateConsumerCount(graph, transpose->MutableOutputDefs()[0], consumer_count);
  graph_utils::RemoveNodeOutputEdges(graph, *cast);
  graph.RemoveNode(cast->Index());
  if (consumers == 0) {
    removed_nodes.push_front(transpose->Index());
  }
  return &new_transpose;
}

// Check whether the element_type is an allowed FusedMatMul data type or not.
static bool IsAllowedFusedMatMulDataType(ONNX_NAMESPACE::TensorProto_DataType element_type)
{
  return element_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT ||
         element_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ||
         element_type == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE ||
         element_type == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
}

/*********************************************************************************************

Case I: The followin is a scenario where Transpose output feeds MatMul. The Transpose input can be either on the left or right.
   The input graph
                         __________                             __________
                         | input0 |                             | input1 |
                         |________|                             |________|
                              |                                      |
                              |                                      |
                              |                                      |
                         _____V______                                |
                         |Transpose |                                |
                         |__________|                                |
                              |                                      |
                              |                                      |
                              |______________           _____________|
                                            |           |
                                            |           |
                                            |           |
                                          __V___________V__
                                          |    MatMul     |
                                          |_______________|
                                                  |
                                                  V
    is transformed to the following

                         __________                             __________
                         | input0 |                             | input1 |
                         |________|                             |________|
                              |                                      |
                              |                                      |
                              |                                      |
                              |_____________            _____________|
                                            |           |
                                            |           |
                                            |           |
                                          __V___________V__
                                          |  FusedMatMul  |
                                          |_______________|
                                                  |
                                                  V

Case II: The output of Tanspose feeds Cast and the output from the Cast feeds MatMul
   The input graph
                         __________                             __________
                         | input0 |                             | input1 |
                         |________|                             |________|
                              |                                      |
                              |                                      |
                         _____V______                                |
                         |Transpose |                                |
                         |__________|                                |
                              |                                      |
                              |                                      |
                         _____V______                                |
                         |  Cast    |                                |
                         |__________|                                |
                              |                                      |
                              |______________           _____________|
                                            |           |
                                            |           |
                                            |           |
                                          __V___________V__
                                          |    MatMul     |
                                          |_______________|
                                                  |
                                                  V
    is transformed to the following

                         __________                             __________
                         | input0 |                             | input1 |
                         |________|                             |________|
                              |                                      |
                              |                                      |
                              |                                      |
                         _____V______                                |
                         |  Cast    |                                |
                         |__________|                                |
                              |                                      |
                              |______________           _____________|
                                            |           |
                                            |           |
                                            |           |
                                          __V___________V__
                                          |  FusedMatMul  |
                                          |_______________|
                                                  |
                                                  V

********************************************************************************************************************/

Status MatmulTransposeFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::deque<onnxruntime::NodeIndex> removed_nodes;

  std::unordered_map<NodeArg*, size_t> consumer_count;

  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if ((!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", {9, 13}) &&
         !graph_utils::IsSupportedOptypeVersionAndDomain(node, "FusedMatMul", {1}, kMSDomain)) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
      continue;
    }

    NodeArg* left_input = node.MutableInputDefs()[0];
    auto left_type = left_input->TypeAsProto()->tensor_type().elem_type();
    if (!IsAllowedFusedMatMulDataType(static_cast<ONNX_NAMESPACE::TensorProto_DataType>(left_type))) {
      continue;
    }
    auto left = GetTransposeNodeFromOutput(graph, *left_input);

    NodeArg* right_input = node.MutableInputDefs()[1];
    auto right_type = right_input->TypeAsProto()->tensor_type().elem_type();
    if (!IsAllowedFusedMatMulDataType(static_cast<ONNX_NAMESPACE::TensorProto_DataType>(right_type))) {
      continue;
    }
    auto right = GetTransposeNodeFromOutput(graph, *right_input);

    if (!left) {
      Node* left_node = graph.GetMutableProducerNode(left_input->Name());
      if (left_node && left_node->OpType() == "Cast") {
        left = ReorderCastAndTranspose(graph, left_node, consumer_count, removed_nodes);
      }
    }

    if (!right) {
      Node* right_node = graph.GetMutableProducerNode(right_input->Name());
      if (right_node && right_node->OpType() == "Cast") {
        right = ReorderCastAndTranspose(graph, right_node, consumer_count, removed_nodes);
      }
    }

    if (!left && !right) {
      continue;
    }

    if (left) {
      size_t left_consumers = UpdateConsumerCount(graph, left_input, consumer_count);
      if (left_consumers == 0)
        removed_nodes.push_front(left->Index());
      left_input = left->MutableInputDefs()[0];
    }

    if (right) {
      size_t right_consumers = UpdateConsumerCount(graph, right_input, consumer_count);
      if (right_consumers == 0)
        removed_nodes.push_front(right->Index());
      right_input = right->MutableInputDefs()[0];
    }

    const std::vector<NodeArg*> input_defs{left_input, right_input};
    const std::vector<NodeArg*> output_defs{node.MutableOutputDefs()[0]};

    Node& matmul_node = graph.AddNode(graph.GenerateNodeName("MatMul_With_Transpose"),
                                      "FusedMatMul",
                                      "fused MatMul and Transpose ",
                                      input_defs,
                                      output_defs, {}, kMSDomain);
    bool transpose_left = (left != nullptr);
    bool transpose_right = (right != nullptr);
    float alpha = 1.0f;
    if (node.OpType() == "FusedMatMul") {
      transpose_left ^= static_cast<bool>(node.GetAttributes().at("transA").i());
      transpose_right ^= static_cast<bool>(node.GetAttributes().at("transB").i());
      alpha = node.GetAttributes().at("alpha").f();
    }
    matmul_node.AddAttribute("transA", static_cast<int64_t>(transpose_left));
    matmul_node.AddAttribute("transB", static_cast<int64_t>(transpose_right));
    matmul_node.AddAttribute("alpha", alpha);
    // Assign provider to this new node. Provider should be same as the provider for old node.
    matmul_node.SetExecutionProviderType(node.GetExecutionProviderType());

    graph_utils::FinalizeNodeFusion(graph, matmul_node, node);

    modified = true;
  }

  // Have to remove node in reversed order for now to walk around the issue in RemoveNode
  for (onnxruntime::NodeIndex removed_node : removed_nodes) {
    graph.RemoveNode(removed_node);
  }

  return Status::OK();
}
}  // namespace onnxruntime
