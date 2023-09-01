// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/bias_softmax_fusion.h"

#include <deque>

#include "core/common/logging/logging.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/providers/common.h"
#include "core/graph/graph_utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
using namespace onnxruntime;

namespace {

// helper to check dimensions match on concrete or symbolic value
bool operator==(const ONNX_NAMESPACE::TensorShapeProto_Dimension& lhs, int value) {
  return utils::HasDimValue(lhs) && lhs.dim_value() == value;
}

bool operator!=(const ONNX_NAMESPACE::TensorShapeProto_Dimension& lhs, int value) {
  return !(lhs == value);
}

void select_input_on_lhs_condition(bool lhs_condition, Node& add_node, NodeArg*& input, NodeArg*& mask) {
  if (lhs_condition) {
    input = add_node.MutableInputDefs()[0];
    mask = add_node.MutableInputDefs()[1];
  } else {
    input = add_node.MutableInputDefs()[1];
    mask = add_node.MutableInputDefs()[0];
  }
}

// pattern match on dropout(softmax(input + mask)) subgraph
bool TryBiasSoftmaxSubgraphMatch(Graph& graph, Node& start, Node*& add, Node*& softmax) {
  Node& node = start;
  add = softmax = nullptr;

  // check node is add and has single output
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Add", {7, 13, 14}) ||
      !graph_utils::IsSupportedProvider(node, {kCudaExecutionProvider, kRocmExecutionProvider}) ||
      !optimizer_utils::CheckOutputEdges(graph, node, 1)) {
    return false;
  }

  // check shape information is not available for both add inputs
  Node& add_node = node;
  NodeArg* input1 = add_node.MutableInputDefs()[0];
  NodeArg* input2 = add_node.MutableInputDefs()[1];
  const TensorShapeProto* S1 = input1->Shape();
  const TensorShapeProto* S2 = input2->Shape();
  if (S1 == nullptr || S2 == nullptr || S1->dim_size() < 1 || S2->dim_size() < 1) {
    return false;
  }

  // BiasSoftmax supports only float/float16/double - see ./onnxruntime/core/graph/contrib_ops/contrib_defs.cc
  auto type_allowed = [](NodeArg* input) {
    auto data_type = input->TypeAsProto()->tensor_type().elem_type();
    if (data_type != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE &&
        data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
        data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      return false;
    }
    return true;
  };
  if (!type_allowed(input1) || !type_allowed(input2)) {
    return false;
  }

  // check add is only consumed by softmax with matching exec provider
  Node& softmax_node = *graph.GetNode(add_node.OutputNodesBegin()->Index());
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(softmax_node, "Softmax", {1, 11, 13}) ||
      softmax_node.GetExecutionProviderType() != add_node.GetExecutionProviderType()) {
    return false;
  }

  // pattern match succeeded
  add = &add_node;
  softmax = &softmax_node;
  return true;
}

/**
 * Here we check input and mask dimensions are as expected.
 * Let
 *     input shape = [ a0 ... a(B-1) aB a(B+1) ... a(k-1) ak a(k+1) ... a(N-1) ]
 * with rank = N and softmax axis = k.
 *
 * Then the mask shape should be:
 *     mask shape  = [                   ...              ak a(k+1) ... a(N-1) ]
 * with dimensions from k to N - 1 same as input shape.
 *
 * We support two types of broadcasting:
 * The first type is broadcasting the inner dimensions before dim-k:
 *     mask shape  = [ a0 ... a(B-1) 1    1    ...   1    ak a(k+1) ... a(N-1) ]
 * The second type is broadcasting the outer dimensions before dim-k:
 *     mask shape  = [  ...  1,  1,  aB a(B+1) ... a(k-1) ak a(k+1) ... a(N-1) ]
 * For second case, the rank of mask shape can be smaller than N.
 *
 * If for mask shape the dimensions before dim-k are all 1 or are all same as input shape,
 * either type (inner broadcast or outer broadcast) is OK.
 *
 * In the BERT case scores shape = [batch_size, num_heads, seq_length, seq_length]
 *       and sequence mask shape = [batch_size,     1,         1,      seq_length]
 *
 * NOTE that the axis attribute for Softmax in OpSet-11 and OpSet-13 are different. For OpSet-11, dim ak to dim a(N-1)
 * are in same batch. But since OpSet-13, only ak is in a batch. Above fusion logic is for OpSet-11 or before.
 * Since OpSet-13, to compute Softmax, we will first transpose the axis dim to the last dim before the real Softmax
 * computation if axis is not the last dim. Fusing Add+Softmax to BiasSoftmax would require extra transpose for bias,
 * and bring complex checking condition. So since OpSet-13, we will apply the fusion only when axis is the last dim.
 */
bool TrySelectInputAndBiasWithAlignment(Node& add_node, Node& softmax_node, NodeArg*& input, NodeArg*& mask,
                                        int& new_axis, bool& is_inner_broadcast) {
  NodeArg* input1 = add_node.MutableInputDefs()[0];
  NodeArg* input2 = add_node.MutableInputDefs()[1];

  // default axis = -1 if opset >= 13
  bool is_since_opset_13 = !graph_utils::MatchesOpSinceVersion(softmax_node, {1, 11});
  int axis = is_since_opset_13 ? -1 : 1;
  auto& softmax_attr = softmax_node.GetAttributes();
  if (softmax_attr.find("axis") != softmax_attr.end()) {
    auto& axis_attr = softmax_attr.at("axis");
    axis = utils::HasInt(axis_attr) ? (int)axis_attr.i() : axis;
  }

  int N1 = input1->Shape()->dim_size();
  int N2 = input2->Shape()->dim_size();
  int rank = std::max({N1, N2});
  new_axis = (int)HandleNegativeAxis(axis, rank);

  // The axis attribute for Softmax in OpSet-11 and OpSet-13 are different.
  // Details in function documentatin.
  if (is_since_opset_13 && new_axis != rank - 1) return false;

  int singlebatch_rank = rank - new_axis;
  if (singlebatch_rank > N1 || singlebatch_rank > N2) {
    return false;
  }

  // All dims from k to N-1 should be same.
  for (int i = 1; i <= singlebatch_rank; i++) {
    if (input1->Shape()->dim(N1 - i) != input2->Shape()->dim(N2 - i)) {
      return false;
    }
  }

  // Inner broadcasting check.
  if (N1 == N2) {
    // discover B (first axis where shapes don't match)
    int pos = 0;
    while (pos < new_axis && input1->Shape()->dim(pos) == input2->Shape()->dim(pos)) {
      pos++;
    }

    // use B dimension to distinguish input and mask
    select_input_on_lhs_condition(pos == new_axis || input1->Shape()->dim(pos) != 1, add_node, input, mask);

    // confirm mask dimensions are ones on broadcast axes B to (k-1)
    is_inner_broadcast = true;
    for (int i = pos; i < new_axis; i++) {
      if (mask->Shape()->dim(i) != 1) {
        is_inner_broadcast = false;
        break;
      }
    }

    // If is_inner_broadcast is true, it's the inner broadcasting type.
    // Otherwise, we will check the outer broadcasting type below.
    if (is_inner_broadcast) return true;
  }

  // Outer broadcasting check.
  int pos1 = N1 - singlebatch_rank - 1;
  int pos2 = N2 - singlebatch_rank - 1;
  while (pos1 >= 0 && pos2 >= 0 && input1->Shape()->dim(pos1) == input2->Shape()->dim(pos2)) {
    pos1--;
    pos2--;
  }

  // use B-1 dimension to distinguish input and mask
  bool is_lhs_input = pos1 > pos2 || (pos1 == pos2 && (pos1 < 0 || input1->Shape()->dim(pos1) != 1));
  select_input_on_lhs_condition(is_lhs_input, add_node, input, mask);
  for (int i = is_lhs_input ? pos2 : pos1; i >= 0; --i) {
    if (mask->Shape()->dim(i) != 1) {
      return false;
    }
  }

  is_inner_broadcast = false;
  return true;
}

// coalesce subgraph nodes into fused node
void FuseBiasSoftmaxSubgraph(
    Graph& graph,
    Node& add_node,
    Node& softmax_node,
    NodeArg* input,
    NodeArg* mask,
    int axis,
    bool is_inner_broadcast) {
  const std::array fused_inputs{input, mask};

  std::string fused_desc =
      "fused " + add_node.Name() + " and " + softmax_node.Name() + " into softmax(input + bias)";

  std::string op_type = "BiasSoftmax";
  Node& fused_node = graph.AddNode(graph.GenerateNodeName(op_type),
                                   op_type,
                                   fused_desc,
                                   fused_inputs,
                                   {},
                                   {},
                                   kMSDomain);

  // add softmax axis and broadcast axis (to simplify runtime logic)
  // recall broadcast along axes B ... (k-1)
  fused_node.AddAttribute("axis", static_cast<int64_t>(axis));
  fused_node.AddAttribute("is_inner_broadcast", static_cast<int64_t>(is_inner_broadcast ? 1 : 0));

  // finalize node fusion (e.g. remove old nodes and shift outputs)
  fused_node.SetExecutionProviderType(add_node.GetExecutionProviderType());
  graph_utils::FinalizeNodeFusion(graph, {add_node, softmax_node}, fused_node);
}

}  // namespace

namespace onnxruntime {

Status BiasSoftmaxFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  // only support GPU execution provider
  auto& cep = GetCompatibleExecutionProviders();
  if (cep.size() > 0 && cep.find(kCudaExecutionProvider) == cep.end() && cep.find(kRocmExecutionProvider) == cep.end())
    return Status::OK();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (nullptr == node_ptr)
      continue;  // node was removed

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    Node *add_node, *softmax_node;
    if (!TryBiasSoftmaxSubgraphMatch(graph, node, add_node, softmax_node)) {
      continue;
    }

    NodeArg *input, *mask;
    int new_axis;
    bool is_inner_broadcast;
    if (!TrySelectInputAndBiasWithAlignment(*add_node, *softmax_node, input, mask, new_axis, is_inner_broadcast)) {
      continue;
    }

    FuseBiasSoftmaxSubgraph(graph, *add_node, *softmax_node, input, mask, new_axis, is_inner_broadcast);
    modified = true;

    VLOGF(logger, 1, "Fused Add + Softmax into BiasSoftmax node.\n");
  }

  return Status::OK();
}

}  // namespace onnxruntime
