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

// is_trans is whether to transpose the 2 dims used to MatMul.
// is_trans_batch is whether to transpose 1st dim and batch dims.
// Batch here has the same meaning in CUDA's GemmStridedBatched (other than the training batch concept),
// i.e., all dims except the 2 dims used to MatMul, including from dim[0] to dim[rank-3].
// For example (take lhs input as example):
// is_trans=False, is_trans_batch=False:
//   the input tensor is in shape [b0,...,bn,M,K], no Transpose (no fuse for this case)
// is_trans=False, is_trans_batch=True:
//   the input tensor is in shape [M,b0,...,bn,K], Transpose perm=[1,...,rank-2,0,rank-1]
// is_trans=True , is_trans_batch=False:
//   the input tensor is in shape [b0,...,bn,K,M], Transpose perm=[0,...,rank-3,rank-1,rank-2]
// is_trans=True , is_trans_batch=True:
//   the input tensor is in shape [K,b0,...,bn,M], Transpose perm=[1,...,rank-2,rank-1,0]
static Node* GetTransposeNodeFromOutput(Graph& graph, NodeArg& node_arg, bool& is_trans, bool& is_trans_batch) {
  is_trans = is_trans_batch = false;

  // Skip if not a Transpose node or it has graph output.
  Node* trans_node = graph.GetMutableProducerNode(node_arg.Name());
  if (trans_node == nullptr || trans_node->OpType() != "Transpose" || graph.NodeProducesGraphOutput(*trans_node)) {
    return nullptr;
  }

  std::vector<int64_t> perms;
  if (!GetTransposePerms(*trans_node, perms)) {
    return nullptr;
  }

  size_t rank = perms.size();
  if (rank < 2) {
    return nullptr;
  }

  // We can fuse the Transpose node only when the last axis of original tensor is within the last two dims after transpose.
  int64_t last_axis = static_cast<int64_t>(rank) - 1;
  size_t last_axis_index = perms[rank - 1] == last_axis ? rank - 1 : perms[rank - 2] == last_axis ? rank - 2 : rank;
  if (last_axis_index == rank) {
    return nullptr;
  }

  // Transpose node can be fused to MatMul when the batch dims keep same relative orders before and after transpose.
  // But if they are not contiguous, after the fusion, we can only use GemmBatched instead of GemmStridedBatched,
  // which may have perf issue. To keep it simple, we will fuse only when batch dimensions are contiguous.
  if (rank >= 3) {
    if (perms[0] != 0 && perms[0] != 1) {
      return nullptr;
    }
    for (size_t i = 0; i < rank - 3; ++i) {
      if (perms[i] + 1 != perms[i + 1]) {
        return nullptr;
      }
    }
  }

  is_trans = last_axis_index == rank - 2;
  is_trans_batch = rank >= 3 && perms[0] == 1;
  return trans_node;
}

static size_t UpdateConsumerCount(Graph& graph, NodeArg* target, InlinedHashMap<NodeArg*, size_t>& count_map) {
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
                                     InlinedHashMap<NodeArg*, size_t>& consumer_count,
                                     std::deque<onnxruntime::NodeIndex>& removed_nodes,
                                     bool& is_trans, bool& is_trans_batch) {
  ORT_ENFORCE(cast != nullptr);
  auto transpose = GetTransposeNodeFromOutput(graph, *cast->MutableInputDefs()[0], is_trans, is_trans_batch);
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

  const std::array new_cast_input_defs{transpose_input};
  const std::array new_cast_output_defs{&new_cast_output};
  const std::array new_transpose_input_defs = {&new_cast_output};
  const std::array new_transpose_output_defs = {cast_output};

  Node& new_cast = graph.AddNode(graph.GenerateNodeName(cast->Name() + "_transformed"),
                      cast->OpType(),
                      "Created a new Cast node to interchange Cast and Transpose nodes",
                      new_cast_input_defs,
                      new_cast_output_defs,
                      &cast->GetAttributes(),
                      cast->Domain());
  new_cast.SetExecutionProviderType(cast->GetExecutionProviderType());

  Node& new_transpose = graph.AddNode(graph.GenerateNodeName(transpose->Name() + "_transformed"),
                                      transpose->OpType(),
                                      "Created a new Transpose node to interchange Cast and Transpose nodes",
                                      new_transpose_input_defs,
                                      new_transpose_output_defs,
                                      &transpose->GetAttributes(),
                                      transpose->Domain());
  new_transpose.SetExecutionProviderType(transpose->GetExecutionProviderType());

  size_t consumers = UpdateConsumerCount(graph, transpose->MutableOutputDefs()[0], consumer_count);
  graph_utils::RemoveNodeOutputEdges(graph, *cast);
  graph.RemoveNode(cast->Index());
  if (consumers == 0) {
    removed_nodes.push_front(transpose->Index());
  }
  return &new_transpose;
}

// Check whether the element_type is an allowed FusedMatMul data type or not.
constexpr static bool IsAllowedFusedMatMulDataType(ONNX_NAMESPACE::TensorProto_DataType element_type) {
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

  InlinedHashMap<NodeArg*, size_t> consumer_count;

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

    bool is_trans_left = false;
    bool is_trans_batch_left = false;
    Node* left = nullptr;
    // If it's already a FusedMatMul with transBatchA is true, don't fuse it.
    if (node.OpType() != "FusedMatMul" || node.GetAttributes().at("transBatchA").i() == 0) {
      left = GetTransposeNodeFromOutput(graph, *left_input, is_trans_left, is_trans_batch_left);
      if (!left) {
        Node* left_node = graph.GetMutableProducerNode(left_input->Name());
        if (left_node && left_node->OpType() == "Cast") {
          left = ReorderCastAndTranspose(graph, left_node, consumer_count, removed_nodes, is_trans_left,
                                         is_trans_batch_left);
        }
      }
    }

    NodeArg* right_input = node.MutableInputDefs()[1];
    auto right_type = right_input->TypeAsProto()->tensor_type().elem_type();
    if (!IsAllowedFusedMatMulDataType(static_cast<ONNX_NAMESPACE::TensorProto_DataType>(right_type))) {
      continue;
    }

    bool is_trans_right = false;
    bool is_trans_batch_right = false;
    Node* right = nullptr;
    // If it's already a FusedMatMul with transBatchB is true, don't fuse it.
    if (node.OpType() != "FusedMatMul" || node.GetAttributes().at("transBatchB").i() == 0) {
      right = GetTransposeNodeFromOutput(graph, *right_input, is_trans_right, is_trans_batch_right);
      if (!right) {
        Node* right_node = graph.GetMutableProducerNode(right_input->Name());
        if (right_node && right_node->OpType() == "Cast") {
          right = ReorderCastAndTranspose(graph, right_node, consumer_count, removed_nodes, is_trans_right,
                                          is_trans_batch_right);
        }
      }
    }

    // When the rank of two inputs are not equal, we need to pad 1 to one of them for MutMul.
    // For example, if padding 1 is the "M" dim for left input, set is_trans_batch to true is not correct logically.
    // To keep it simple, if any side of is_trans_batch is true, we require both side has same rank.
    if (is_trans_batch_left || is_trans_batch_right) {
      auto shape_left = left_input->Shape();
      auto shape_right = right_input->Shape();
      if (!shape_left || !shape_right || shape_left->dim_size() != shape_right->dim_size()) {
        if (is_trans_batch_left) {
          is_trans_left = is_trans_batch_left = false;
          left = nullptr;
        }
        if (is_trans_batch_right) {
          is_trans_right = is_trans_batch_right = false;
          right= nullptr;
        }
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

    const std::array input_defs{left_input, right_input};
    const std::array output_defs{node.MutableOutputDefs()[0]};

    Node& matmul_node = graph.AddNode(graph.GenerateNodeName("MatMul_With_Transpose"),
                                      "FusedMatMul",
                                      "fused MatMul and Transpose ",
                                      input_defs,
                                      output_defs, {}, kMSDomain);
    float alpha = 1.0f;
    if (node.OpType() == "FusedMatMul") {
      is_trans_left ^= static_cast<bool>(node.GetAttributes().at("transA").i());
      is_trans_right ^= static_cast<bool>(node.GetAttributes().at("transB").i());
      is_trans_batch_left ^= static_cast<bool>(node.GetAttributes().at("transBatchA").i());
      is_trans_batch_right ^= static_cast<bool>(node.GetAttributes().at("transBatchB").i());
      alpha = node.GetAttributes().at("alpha").f();
    }
    matmul_node.AddAttribute("transA", static_cast<int64_t>(is_trans_left));
    matmul_node.AddAttribute("transB", static_cast<int64_t>(is_trans_right));
    matmul_node.AddAttribute("transBatchA", static_cast<int64_t>(is_trans_batch_left));
    matmul_node.AddAttribute("transBatchB", static_cast<int64_t>(is_trans_batch_right));
    matmul_node.AddAttribute("alpha", alpha);
    // Assign provider to this new node. Provider should be same as the provider for old node.
    matmul_node.SetExecutionProviderType(node.GetExecutionProviderType());
#ifdef USE_ROCM
    // forward the __backwardpass, if present
    auto& attrs = node.GetAttributes();
    if (attrs.count("__backwardpass")) {
      matmul_node.AddAttribute("__backwardpass", static_cast<int64_t>(attrs.at("__backwardpass").i()));
    }
#endif

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
