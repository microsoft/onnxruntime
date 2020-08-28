// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/providers/common.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/bias_softmax_fusion.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include <deque>

#define ILOGF(format_str, ...) \
  LOGF_DEFAULT(INFO, format_str, ##__VA_ARGS__)

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

// helper to check dimensions match on concrete or symbolic value
bool operator!=(
  const ONNX_NAMESPACE::TensorShapeProto_Dimension& lhs,
  const ONNX_NAMESPACE::TensorShapeProto_Dimension& rhs) {
    return !(lhs == rhs);
} 

bool operator==(const ONNX_NAMESPACE::TensorShapeProto_Dimension& lhs, int value) {
  return utils::HasDimValue(lhs) && lhs.dim_value() == value;
}

bool operator!=(const ONNX_NAMESPACE::TensorShapeProto_Dimension& lhs, int value) {
  return !(lhs == value);
}

void select_input_on_lhs_condition(bool lhs_condition, Node& add_node, NodeArg** input, NodeArg** mask) {
  if (lhs_condition) {
    *input = add_node.MutableInputDefs()[0];
    *mask = add_node.MutableInputDefs()[1];
  }
  else {
    *input = add_node.MutableInputDefs()[1];
    *mask = add_node.MutableInputDefs()[0];
  }
}

Status BiasSoftmaxFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (nullptr == node_ptr)
      continue;  // node was removed

    auto& node = *node_ptr;

    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    // pattern match on dropout(softmax(input + mask)) subgraph
    // -----------------------------------------------------------------------

    // check node is add and has single output
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Add", {7}) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        !optimizer_utils::CheckOutputEdges(graph, node, 1)) {
      continue;
    }


    // check shape information is not available for both add inputs
    auto& add_node = node;
    NodeArg* input1 = add_node.MutableInputDefs()[0];
    NodeArg* input2 = add_node.MutableInputDefs()[1];
    const TensorShapeProto* S1 = input1->Shape();
    const TensorShapeProto* S2 = input2->Shape();
    if (S1 == nullptr || S2 == nullptr || S1->dim_size() < 1 || S2->dim_size() < 1) {
      continue;
    }

    // check add is only consumed by softmax with matching exec provider
    auto p = add_node.OutputNodesBegin();
    if (p == add_node.OutputNodesEnd()) {
      continue;
    }
    Node& softmax_node = const_cast<Node&>(*p);
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(softmax_node, "Softmax", {1, 11}) ||
        softmax_node.GetExecutionProviderType() != add_node.GetExecutionProviderType()) {
      continue;
    }

    // can't perform conversion if output is graph output
    if (!graph.GetNodeOutputsInGraphOutputs(add_node).empty()) {
      continue;
    }

    // check mask can broadcast across input batches with simple division
    // -----------------------------------------------------------------------

    /* Here we check input and mask dimensions are as expected:

       Let 
           input shape = [ a0 ... a(B-1) aB a(B+1) ... a(k-1) ak a(k+1) ... a(N-1) ] 
       with rank = N and softmax axis = k and to be determined broadcast axis = B.

       Then
           mask shape  = [ a0 ... a(B-1) 1    1    ...   1    ak a(k+1) ... a(N-1) ] 
       with rank = N and B < k, OR
           mask shape  = [                   ...    1    1    ak a(k+1) ... a(N-1) ] 
       with rank = N-k.

       The mask will be repeated every aB*a(B+1)...*a(k-1) input batches.
       (If input and mask shape match, B = k and no broadcast occurs.)

       In the BERT case scores shape = [batch_size, num_heads, seq_length, seq_length]
             and sequence mask shape = [batch_size,     1,         1,      seq_length]
    */

    // confirm all dimensions starting from softmax axis match for input and mask
    bool singlebatch_shape_matches = true;

    int axis = 1;
    auto& softmax_attr = softmax_node.GetAttributes();
    if (softmax_attr.find("axis") != softmax_attr.end()) {
      auto& axis_attr = softmax_attr.at("axis");
      axis = utils::HasInt(axis_attr)? (int)axis_attr.i() : 1;
    }

    int N1 = input1->Shape()->dim_size();
    int N2 = input2->Shape()->dim_size();
    int k = (int)HandleNegativeAxis(axis, std::max({N1, N2}));
    int singlebatch_rank = std::max({N1-k, N2-k});

    if (singlebatch_rank > N1 || singlebatch_rank > N2) {
      continue;
    }
    for (int i = 1; i <= singlebatch_rank; i++) {
      if (input1->Shape()->dim(N1-i) != input2->Shape()->dim(N2-i)) {
        singlebatch_shape_matches = false;
        break;
      }
    }

    if (!singlebatch_shape_matches) {
      continue;
    }

    // identify broadcast dimensions (i.e. B to k-1 in expression above)
    // also distinguish between input and mask in this process
    bool mask_can_simple_broadcast = true;

    int B;
    NodeArg *input, *mask;

     // case 1: mask rank == input rank
    if (N1 == N2) {

      // discover B (first axis where shapes don't match)
      B = 0;
      while (B < k && input1->Shape()->dim(B) == input2->Shape()->dim(B)) {
        B++;
      }

      // use B dimension to distinguish input and mask
      select_input_on_lhs_condition(input1->Shape()->dim(B) != 1, add_node, &input, &mask);

      // confirm mask dimensions are ones on broadcast axes B to (k-1)
      for (int i = B; i < k; i++) {
        if (mask->Shape()->dim(i) != 1) {
          mask_can_simple_broadcast = false;
          break;
        }
      }
    }

    // case 2: mask rank < input rank
    else { 

      B = 0;
      select_input_on_lhs_condition(N1 > N2, add_node, &input, &mask);

      // confirm any mask dimensions are ones before softmax axis
      int mask_rank = mask->Shape()->dim_size();
      for (int i = 0; i < mask_rank - singlebatch_rank; i++) {
        if (mask->Shape()->dim(i) != 1) {
          mask_can_simple_broadcast = false;
          break;
        }
      }
    }

    if (!mask_can_simple_broadcast) {
      continue;
    }

    // coalesce subgraph nodes into fused node
    // -----------------------------------------------------------------------
    std::vector<NodeArg*> fused_inputs{input, mask};
    ILOGF("Fusing subgraph into BiasSoftmax node.\n");

    std::string op_type = "BiasSoftmax";
    Node& fused_node = graph.AddNode(graph.GenerateNodeName(op_type),
                                    op_type,
                                    "fused softmax(input + bias)",
                                    fused_inputs,
                                    {},
                                    {},
                                    kMSDomain);

    // add softmax axis and broadcast axis (to simplify runtime logic)
    // recall broadcast along axes B ... (k-1)
    fused_node.AddAttribute("softmax_axis", (int64_t)k);
    fused_node.AddAttribute("broadcast_axis", (int64_t)B);

    // finalize node fusion (e.g. remove old nodes and shift outputs)
    fused_node.SetExecutionProviderType(add_node.GetExecutionProviderType());
    graph_utils::FinalizeNodeFusion(graph, {add_node, softmax_node}, fused_node);

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
