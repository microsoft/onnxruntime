// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "onnx/defs/shape_inference.h"
#include "onnx/defs/tensor_proto_util.h"

#pragma once

#define DEBUG_LOG(x) LOGS(logger, VERBOSE) << x

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

// This file is for helping attention fusion for GPT models.
namespace AttentionFusionHelper {

struct MatchGemmResult {
  const Node* gemm;                     // the Gemm node.
  const Node* input_node;               // one node in the subgraph that accept the input.
  const Node* output_node;              // the node that have output of the subgraph.
  std::vector<NodeIndex> node_indices;  // id of all nodes.
};

// Compare the expected parameters (starts, ends, axes and step)
bool CheckSliceParameters(const Graph& graph, const Node& slice, const std::vector<int>& input_indices, const std::vector<int64_t>& expected_values, const logging::Logger& logger) {
  ORT_ENFORCE(input_indices.size() == expected_values.size() && input_indices.size() > 0);

  // Here assumes that the last element of input_indices is the maximum one.
  if (slice.InputDefs().size() <= static_cast<size_t>(input_indices[input_indices.size() - 1])) {
    DEBUG_LOG("Slice does not have enough number of inputs");
    return false;
  }

  for (size_t i = 0; i < expected_values.size(); i++) {
    const NodeArg& input = *(slice.InputDefs()[input_indices[i]]);
    if (expected_values[i] >= static_cast<int64_t>(INT_MAX)) {
      std::vector<int64_t> ends;
      if (!(optimizer_utils::AppendTensorFromInitializer(graph, input, ends, true) && ends.size() == 1 && ends[0] >= INT_MAX)) {
        DEBUG_LOG("Slice ends is less than INT_MAX");
        return false;
      }
    } else if (!optimizer_utils::IsInitializerWithExpectedValue(graph, input, expected_values[i], true)) {
      DEBUG_LOG("Slice parameter is not expected. Input index:" << input_indices[i] << "Expected value:" << expected_values[i]);
      return false;
    }
  }

  return true;
}
/** Match GEMM subgraph:
      +-----------------------------------------------------------------------------------------+
      |                                                                                         |
      |                (*,-1,max,0)                                                             v
[Input]--> Shape --> Slice ---------> Squeeze --> Unsqueeze (axes=0) --> Concat (-1, *) --> Reshape-->Gemm (B:W*4W, C:4W, or B:W*W, C:W, or B:4W*W, C:W)
      |                                                                                                    |
      |                                                             Concat (  ,  , 4W or W)-------------Reshape ----> [Output]
      |                                                                     ^  ^
      |                                                                     |  |
      +----> Shape --> Gather (indices=0) --> Unsqueeze (axes=0) -----------+  |
      |                                                                        |
      +----> Shape --> Gather (indices=1) --> Unsqueeze (axes=0) --------------+

The 3 Shape nodes are merged into one node if use_shared_node is true.
*/
bool MatchGemmSubgraph(Graph& graph,
                       Node& node_after_gemm_reshape,
                       int dst_arg_index,
                       MatchGemmResult& result,
                       bool use_shared_node,
                       const logging::Logger& logger) {
  DEBUG_LOG("Start MatchGemmSubgraph");
  // GPT Attention fusion supports opset version 9 or later.
  std::vector<graph_utils::EdgeEndToMatch> parent_path{
      {0, dst_arg_index, "Reshape", {5, 13}, kOnnxDomain},
      {0, 0, "Gemm", {9, 11, 13}, kOnnxDomain},
      {0, 0, "Reshape", {5, 13}, kOnnxDomain},
      {0, 1, "Concat", {4, 11, 13}, kOnnxDomain},
      {0, 1, "Unsqueeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Squeeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Slice", {1, 10, 11, 13}, kOnnxDomain},
      {0, 0, "Shape", {1, 13}, kOnnxDomain}};

  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(node_after_gemm_reshape, true, parent_path, edges, logger)) {
    DEBUG_LOG("Faild to match gemm path");
    return false;
  }

  const Node& reshape_after_gemm = edges[0]->GetNode();
  const Node& gemm = edges[1]->GetNode();
  const Node& reshape_before_gemm = edges[2]->GetNode();
  const Node& concat = edges[3]->GetNode();
  const Node& unsqueeze = edges[4]->GetNode();
  const Node& squeeze = edges[5]->GetNode();
  const Node& slice = edges[6]->GetNode();
  const Node& shape_before_slice = edges[7]->GetNode();

  const auto& subgraph_input = shape_before_slice.InputDefs()[0];
  if (reshape_before_gemm.InputDefs()[0]->Name() != subgraph_input->Name()) {
    DEBUG_LOG("Input of reshape_before_gemm is not the input of subgraph");
    return false;
  }

  if (!optimizer_utils::CheckOutputEdges(graph, shape_before_slice, use_shared_node ? 3 : 1) ||
      !optimizer_utils::CheckOutputEdges(graph, slice, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, squeeze, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, unsqueeze, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, concat, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, reshape_before_gemm, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, gemm, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, reshape_after_gemm, 1)) {
    DEBUG_LOG("Output edge count not expected for nodes in gemm path");
    return false;
  }

  if (gemm.InputDefs().size() != 3) {
    DEBUG_LOG("Gemm does not have 3 inputs");
    return false;
  }

  // Get the shape of bias, to be compared with the last input value of Concat
  if (!graph_utils::IsInitializer(graph, gemm.InputDefs()[2]->Name(), true)) {
    DEBUG_LOG("Gemm bias is not constant");
    return false;
  }
  auto bias_shape = gemm.InputDefs()[2]->Shape();
  if (bias_shape == nullptr || static_cast<size_t>(bias_shape->dim_size()) != 1 || !utils::HasDimValue(bias_shape->dim(0))) {
    DEBUG_LOG("Gemm bias shape not expected");
    return false;
  }

  if (!CheckSliceParameters(graph, slice, {1, 2, 3}, {-1, INT_MAX, 0}, logger)) {
    DEBUG_LOG("CheckSliceParameters return false");
    return false;
  }

  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(concat.InputDefs()[0]), (int64_t)-1, true)) {
    DEBUG_LOG("concat first input value is not -1");
    return false;
  }

  // Find the concat node for Gather paths.
  std::vector<graph_utils::EdgeEndToMatch> edge_to_match{{0, 1, "Concat", {4, 11, 13}, kOnnxDomain}};
  if (!graph_utils::FindPath(reshape_after_gemm, true, edge_to_match, edges, logger)) {
    DEBUG_LOG("Faild to match concat node for Gather paths");
    return false;
  }

  const Node& concat_after_gather = edges[0]->GetNode();
  if (concat_after_gather.InputDefs().size() != 3 ||
      !optimizer_utils::CheckOutputEdges(graph, concat_after_gather, 1)) {
    DEBUG_LOG("concat_after_gather does not have expected number of inputs or output edges");
    return false;
  }

  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(concat_after_gather.InputDefs()[2]), bias_shape->dim(0).dim_value(), true)) {
    DEBUG_LOG("concat_after_gather input 2 does not have expected value");
    return false;
  }

  result.node_indices.reserve(15);

  // Match: [Input] ----> Shape --> Gather (indices=0 or 1) --> Unsqueeze (axes=0) ----> Concat ( , , )
  for (int i = 0; i < 2; i++) {
    std::vector<graph_utils::EdgeEndToMatch> gather_path1{
        {0, i, "Unsqueeze", {1, 11, 13}, kOnnxDomain},
        {0, 0, "Gather", {1, 11, 13}, kOnnxDomain},
        {0, 0, "Shape", {1, 13}, kOnnxDomain}};

    if (!graph_utils::FindPath(concat_after_gather, true, gather_path1, edges, logger)) {
      DEBUG_LOG("Faild to match gemm gather path");
      return false;
    }

    const Node& unsqueeze_after_gather = edges[0]->GetNode();
    const Node& gather = edges[1]->GetNode();
    const Node& shape = edges[2]->GetNode();

    if (!optimizer_utils::CheckOutputEdges(graph, unsqueeze_after_gather, 1) ||
        !optimizer_utils::CheckOutputEdges(graph, gather, 1) ||
        !optimizer_utils::CheckOutputEdges(graph, shape, 1) && !use_shared_node) {
      DEBUG_LOG("Output edge count not expected for nodes in gemm gather path");
      return false;
    }

    result.node_indices.push_back(unsqueeze_after_gather.Index());
    result.node_indices.push_back(gather.Index());

    if (use_shared_node) {
      if (shape.Index() != shape_before_slice.Index()) {
        return false;
      }
    } else {
      result.node_indices.push_back(shape.Index());
    }

    if (shape.InputDefs()[0]->Name() != subgraph_input->Name()) {
      return false;
    }

    std::vector<int64_t> axes;
    if (!(graph_utils::GetRepeatedNodeAttributeValues(unsqueeze_after_gather, "axes", axes) && axes.size() == 1 && axes[0] == 0)) {
      DEBUG_LOG("unsqueeze_after_gather axes value not expected");
      return false;
    }

    if (!optimizer_utils::IsAttributeWithExpectedValue(gather, "axis", (int64_t)0)) {
      DEBUG_LOG("gather axis value not expected");
      return false;
    }

    if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(gather.InputDefs()[1]), static_cast<int64_t>(i), true)) {
      DEBUG_LOG("gather input 1 value is not expected");
      return false;
    }
  }

  result.gemm = &gemm;
  result.input_node = &shape_before_slice;
  result.output_node = &reshape_after_gemm;
  result.node_indices.insert(result.node_indices.end(),
                             {reshape_after_gemm.Index(),
                              gemm.Index(),
                              reshape_before_gemm.Index(),
                              concat.Index(),
                              unsqueeze.Index(),
                              squeeze.Index(),
                              slice.Index(),
                              shape_before_slice.Index(),
                              concat_after_gather.Index()});

  DEBUG_LOG("Pass MatchGemmSubgraph");
  return true;
}

bool ValidateGemmInitializer(const Graph& graph, const Node& gemm, int64_t hidden_size, bool is_before_split, const logging::Logger& logger) {
  DEBUG_LOG("Start ValidateGemmInitializer");
  const NodeArg& bias = *(gemm.InputDefs()[2]);
  if (!graph_utils::IsInitializer(graph, bias.Name(), true)) {
    DEBUG_LOG("Gemm bias is not constant initializer");
    return false;
  }

  int64_t bias_length = (is_before_split ? 3 : 1) * hidden_size;
  if (!optimizer_utils::ValidateShape(bias, {bias_length})) {
    DEBUG_LOG("Gemm bias shape is not expected");
    return false;
  }

  const NodeArg& weights = *(gemm.InputDefs()[1]);
  if (!graph_utils::IsInitializer(graph, weights.Name(), true)) {
    DEBUG_LOG("Gemm weight is not constant initializer");
    return false;
  }

  if (!optimizer_utils::ValidateShape(weights, {hidden_size, bias_length})) {
    DEBUG_LOG("Gemm weight shape is not expected");
    return false;
  }

  DEBUG_LOG("Pass ValidateGemmInitializer");
  return true;
}

struct MatchUnidirMaskResult {
  const Node* div_node;                 // the root node (Div) of the subgraph
  bool is_unidirectional;               // whether the mask is unidirectional.
  std::vector<NodeIndex> node_indices;  // id of all nodes in the subgraph for removing later.
};

// Return true when mask is unidirectionl (lower trigular) or all elements are 1.
template <class T>
bool ValidateUnidirMask(std::vector<T> mask_data, int64_t w, bool& is_undirectional) {
  // The mask data has shape 1x1xWxW
  if (mask_data.size() == static_cast<size_t>(w * w)) {
    bool is_one = true;
    is_undirectional = true;

    const T* p = mask_data.data();
    for (int i = 0; i < w; i++) {
      for (int j = 0; j < w; j++) {
        if (*p != static_cast<T>(1)) {
          is_one = false;
        }

        if (*p != ((i >= j) ? static_cast<T>(1) : static_cast<T>(0))) {
          is_undirectional = false;
        }

        p++;
      }
    }

    if (is_undirectional || is_one)
      return true;
  }

  return false;
}

bool ValidateUnidirMask(const Graph& graph, const NodeArg& mask, bool& is_unidirectional, const logging::Logger& logger) {
  if (!graph_utils::IsInitializer(graph, mask.Name(), true)) {
    DEBUG_LOG("unidir mask is not constant");
    return false;
  }

  // Check that the mask shape is 1x1xWxW
  auto shape = mask.Shape();
  if (shape == nullptr || static_cast<size_t>(shape->dim_size()) != 4 || !utils::HasDimValue(shape->dim(0)) || static_cast<int64_t>(1) != shape->dim(0).dim_value() || !utils::HasDimValue(shape->dim(1)) || static_cast<int64_t>(1) != shape->dim(1).dim_value() || !utils::HasDimValue(shape->dim(2)) || !utils::HasDimValue(shape->dim(3)) || shape->dim(2).dim_value() != shape->dim(3).dim_value()) {
    DEBUG_LOG("unidir mask shape not expected");
    return false;
  }

  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  if (!graph.GetInitializedTensor(mask.Name(), tensor_proto) || tensor_proto == nullptr) {
    return false;
  }

  if (tensor_proto->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
    DEBUG_LOG("This optimizer does not support external data for unidirectional mask right now");
    return false;
  }

  if (tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    std::vector<int32_t> int32_data = ONNX_NAMESPACE::ParseData<int32_t>(tensor_proto);
    if (!ValidateUnidirMask(int32_data, shape->dim(2).dim_value(), is_unidirectional)) {
      DEBUG_LOG("Mask is neither unidirectional nor all ones");
      return false;
    }
  } else if (tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    std::vector<float_t> float_data = ONNX_NAMESPACE::ParseData<float_t>(tensor_proto);
    if (!ValidateUnidirMask(float_data, shape->dim(2).dim_value(), is_unidirectional)) {
      DEBUG_LOG("Mask is neither unidirectional nor all ones");
      return false;
    }
  } else {
    DEBUG_LOG("Expect mask data type is uint8 or float");
    return false;
  }

  return true;
}

/**  Match Unidirectional Mask subgraph.
     In the below graph, ':' is followed by variable name in code. * means the input on the left side.


                                                               (axes=0)
                                        +---------------------Unsqueeze----------------------------------------+
                                        |                     :unsqueeze2                                      |
                                        |                      (axes=0)                                        |
                                        +---------------------Unsqueeze-------------------+                    |
                                        |                     :unsqueeze3                 |                    |
                       (*,-1,max,0)     | (axes=0)  A          (axes=0)           starts  |ends                |
 [Div] --> Shape --> Slice ---------> Squeeze -----> Sub -->  Unsqueeze ----------------+ |                    |ends
      |    :shape1   :slice1          :squeeze1       ^       :unsqueeze1               v v                    v
      |                                               |B                  Slice(1x1xWxW, , ,2,1) --> Slice(*,0, ,3, 1) :last_slice
      |                                               |                   :mask_slice                  |
      |                (*, -2, -1, 0)   (axes=0)      |                                              Cast(9)
      +----> Shape --> Slice ---------> Squeeze-------+                                                |
      |      :shape2   :slice2         :squeeze2                                                       v condition
      +----------------------------------------------------------------------------------------->Where( ,*,-10000)--->[Add]

 When use_shared_node is true, shape1 and shape2 is one node, and also unsqueeze2 and unsqueeze3 is same.
*/
bool MatchUnidirMaskSubgraph(const Graph& graph, const Node& add_node, MatchUnidirMaskResult& result, bool use_shared_node, const logging::Logger& logger) {
  DEBUG_LOG("Start MatchUnidirMaskSubgraph");
  std::vector<graph_utils::EdgeEndToMatch> root_path{
      {0, 0, "Where", {9}, kOnnxDomain},
      {0, 1, "Div", {7, 13}, kOnnxDomain}};

  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(add_node, true, root_path, edges, logger)) {
    DEBUG_LOG("Faild to match the path (Div-->Where-->Add) for unidirectional mask");
    return false;
  }

  const Node& where_node = edges[0]->GetNode();
  const Node& div_node = edges[1]->GetNode();

  const float expected_value = -10000.0f;
  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(where_node.InputDefs()[2]), expected_value, true)) {
    return false;
  }

  std::vector<graph_utils::EdgeEndToMatch> path1{
      {0, 0, "Cast", {9, 13}, kOnnxDomain},
      {0, 0, "Slice", {10, 11, 13}, kOnnxDomain},  // Last Slice
      {0, 0, "Slice", {10, 11, 13}, kOnnxDomain},  // Mask Slice
      {0, 1, "Unsqueeze", {9, 11, 13}, kOnnxDomain},
      {0, 0, "Sub", {7, 13}, kOnnxDomain},
      {0, 0, "Squeeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Slice", {10, 11, 13}, kOnnxDomain},  // Slice 1
      {0, 0, "Shape", {1, 13}, kOnnxDomain}};

  if (!graph_utils::FindPath(where_node, true, path1, edges, logger)) {
    DEBUG_LOG("Faild to match path 1 for unidirectional mask");
    return false;
  }

  const Node& cast = edges[0]->GetNode();
  const Node& last_slice = edges[1]->GetNode();
  const Node& mask_slice = edges[2]->GetNode();
  const Node& unsqueeze1 = edges[3]->GetNode();
  const Node& sub = edges[4]->GetNode();
  const Node& squeeze1 = edges[5]->GetNode();
  const Node& slice1 = edges[6]->GetNode();
  const Node& shape1 = edges[7]->GetNode();

  if (!optimizer_utils::CheckOutputEdges(graph, where_node, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, cast, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, last_slice, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, mask_slice, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, unsqueeze1, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, sub, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, squeeze1, use_shared_node ? 2 : 3) ||
      !optimizer_utils::CheckOutputEdges(graph, slice1, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, shape1, use_shared_node ? 2 : 1)) {
    DEBUG_LOG("Output edge count not expected for nodes in path 1 of unidirectional mask");
    return false;
  }

  if (div_node.OutputDefs()[0]->Name() != shape1.InputDefs()[0]->Name()) {
    DEBUG_LOG("Div and Shape1 does not have edge");
    return false;
  }

  if (!CheckSliceParameters(graph, last_slice, {1, 3, 4}, {0, 3, 1}, logger)) {
    DEBUG_LOG("CheckSliceParameters returns false for last_slice");
    return false;
  }

  if (!CheckSliceParameters(graph, mask_slice, {3, 4}, {2, 1}, logger)) {
    DEBUG_LOG("CheckSliceParameters returns false for mask_slice");
    return false;
  }

  if (!ValidateUnidirMask(graph, *(mask_slice.InputDefs()[0]), result.is_unidirectional, logger)) {
    DEBUG_LOG("ValidateUnidirMask returns false for mask_slice");
    return false;
  }

  if (!CheckSliceParameters(graph, slice1, {1, 2, 3}, {-1, INT_MAX, 0}, logger)) {
    DEBUG_LOG("CheckSliceParameters returns false for slice1");
    return false;
  }

  std::vector<graph_utils::EdgeEndToMatch> slice_ends_path{
      {0, 2, "Unsqueeze", {9, 11, 13}, kOnnxDomain},
      {0, 0, "Squeeze", {1, 11, 13}, kOnnxDomain}};

  if (!graph_utils::FindPath(last_slice, true, slice_ends_path, edges, logger) ||
      edges[1]->GetNode().Index() != squeeze1.Index()) {
    DEBUG_LOG("Faild to match path 2 for unidirectional mask");
    return false;
  }

  const Node& unsqueeze2 = edges[0]->GetNode();
  if (!optimizer_utils::CheckOutputEdges(graph, unsqueeze2, use_shared_node ? 2 : 1)) {
    DEBUG_LOG("Output edge count not expected for unsqueeze2 of unidirectional mask");
    return false;
  }

  if (!graph_utils::FindPath(mask_slice, true, slice_ends_path, edges, logger) ||
      edges[1]->GetNode().Index() != squeeze1.Index()) {
    DEBUG_LOG("Faild to match path 3 for unidirectional mask");
    return false;
  }

  const Node& unsqueeze3 = edges[0]->GetNode();
  if (!optimizer_utils::CheckOutputEdges(graph, unsqueeze3, use_shared_node ? 2 : 1)) {
    DEBUG_LOG("Output edge count not expected for unsqueeze3 of unidirectional mask");
    return false;
  }

  std::vector<graph_utils::EdgeEndToMatch> path4{
      {0, 1, "Squeeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Slice", {10, 11, 13}, kOnnxDomain},  // Slice 2
      {0, 0, "Shape", {1, 13}, kOnnxDomain}};

  if (!graph_utils::FindPath(sub, true, path4, edges, logger)) {
    DEBUG_LOG("Faild to match path 4 for unidirectional mask");
    return false;
  }

  if (div_node.OutputDefs()[0]->Name() != edges[2]->GetNode().InputDefs()[0]->Name()) {
    DEBUG_LOG("Div and Shape does not have edge");
    return false;
  }

  const Node& squeeze2 = edges[0]->GetNode();
  const Node& slice2 = edges[1]->GetNode();
  const Node& shape2 = edges[2]->GetNode();
  if (!optimizer_utils::CheckOutputEdges(graph, squeeze2, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, slice2, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, shape2, use_shared_node ? 2 : 1)) {
    DEBUG_LOG("Output edge count not expected for squeeze_2/slices2/shape2 of unidirectional mask");
    return false;
  }

  if (!CheckSliceParameters(graph, slice2, {1, 2, 3}, {-2, -1, 0}, logger)) {
    DEBUG_LOG("CheckSliceParameters return false for slice2");
    return false;
  }

  if (use_shared_node && (shape1.Index() != shape2.Index() || unsqueeze2.Index() != unsqueeze3.Index())) {
    return false;
  }

  result.div_node = &div_node;
  result.node_indices = {
      where_node.Index(),
      cast.Index(),
      last_slice.Index(),
      mask_slice.Index(),
      unsqueeze1.Index(),
      sub.Index(),
      squeeze1.Index(),
      slice1.Index(),
      shape1.Index(),
      unsqueeze2.Index(),
      squeeze2.Index(),
      slice2.Index()};

  if (!use_shared_node) {
    result.node_indices.push_back(unsqueeze3.Index());
    result.node_indices.push_back(shape2.Index());
  }

  DEBUG_LOG("Pass MatchUnidirMaskSubgraph");
  return true;
}

struct AttentionMaskNodes {
  const Node* softmax;
  bool has_input_mask;  // When it is false, the following nodes will be NULL.

  const Node* add;
  const Node* mul;
  const Node* sub;
  const Node* cast;  // optional, could be NULL.
  const Node* unsqueeze_2;
  const Node* unsqueeze_1;
};

struct AttentionMaskNodesDistilBert {
  const Node* softmax;
  const Node* where;
  const Node* expand;
  const Node* reshape;
  const Node* equal;
  const Node* shape;
};

void SetMaskNodesToRemove(const Graph& graph, AttentionMaskNodes& mask_nodes, std::vector<NodeIndex>& nodes_to_remove) {
  nodes_to_remove.push_back(mask_nodes.softmax->Index());
  if (!mask_nodes.has_input_mask) {
    return;
  }

  nodes_to_remove.push_back(mask_nodes.add->Index());

  // When the last Attention node is fused. Original mask processing nodes can be removed safely.
  if (optimizer_utils::CheckOutputEdges(graph, *(mask_nodes.mul), 1)) {
    nodes_to_remove.push_back(mask_nodes.mul->Index());
    nodes_to_remove.push_back(mask_nodes.sub->Index());
    if (mask_nodes.cast != nullptr) {
      nodes_to_remove.push_back(mask_nodes.cast->Index());
    }
    nodes_to_remove.push_back(mask_nodes.unsqueeze_2->Index());
    nodes_to_remove.push_back(mask_nodes.unsqueeze_1->Index());
  }
}

void SetMaskNodesToRemove(const Graph&, AttentionMaskNodesDistilBert& mask_nodes, std::vector<NodeIndex>& nodes_to_remove) {
  nodes_to_remove.push_back(mask_nodes.softmax->Index());
  nodes_to_remove.push_back(mask_nodes.where->Index());
  nodes_to_remove.push_back(mask_nodes.expand->Index());
  nodes_to_remove.push_back(mask_nodes.reshape->Index());
  nodes_to_remove.push_back(mask_nodes.equal->Index());
  nodes_to_remove.push_back(mask_nodes.shape->Index());
}

/**  Match Input Mask subgraph:
                                                                                                       {UnidirMask Subgraph}
                                                                                                                   |
                                                                  (optional)                                       v
[Attention_mask] --> Unsqueeze (axes=1) --> Unsqueeze (axes=2) --> Cast ---->Sub(1,*) --> Mul(*, -10000.0) --> Add( ,*)--->SoftMax -->[MatMul]

When is_input_mask_optional is true, this function also matches the following subgraph:
    {UnidirMask Subgraph [Where]} --> Softmax --> [MatMul]
In this case, we only match two nodes: "Softmax" and "Where". Note that "Where" is the last node in unidirectional subgraph.
*/
bool MatchInputMaskSubgraph(const Graph& graph, const Node& qkv_matmul, AttentionMaskNodes& result, const logging::Logger& logger, bool is_input_mask_optional) {
  DEBUG_LOG("Start MatchInputMaskSubgraph");

  std::vector<graph_utils::EdgeEndToMatch> softmax_path{
      {0, 0, "Softmax", {1, 11, 13}, kOnnxDomain}};

  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(qkv_matmul, true, softmax_path, edges, logger)) {
    DEBUG_LOG("Failed to find Softmax node");
    return false;
  }

  const Node& softmax = edges[0]->GetNode();
  if (!optimizer_utils::CheckOutputEdges(graph, softmax, 1)) {
    DEBUG_LOG("Output edge count not expected for Softmax");
    return false;
  }

  result.softmax = &softmax;
  result.has_input_mask = false;

  // GPT-2 might not have input mask. In that case the subgraph is like:
  // {UnidirMask Subgraph} --> Softmax --> [MatMul]
  if (is_input_mask_optional) {
    const Node* parent = graph_utils::GetInputNode(softmax, 0);
    if (parent != nullptr && parent->OpType() == "Where") {  // UnidirMask Subgraph ends withs Where node
      return true;
    }
  }

  std::vector<graph_utils::EdgeEndToMatch> mask_path{
      {0, 0, "Add", {7, 13}, kOnnxDomain},
      {0, 1, "Mul", {7, 13}, kOnnxDomain},
      {0, 0, "Sub", {7, 13}, kOnnxDomain}};

  if (!graph_utils::FindPath(softmax, true, mask_path, edges, logger)) {
    DEBUG_LOG("Failed to find path for mask");
    return false;
  }

  const Node& mask_add = edges[0]->GetNode();
  const Node& mask_mul = edges[1]->GetNode();
  const Node& mask_sub = edges[2]->GetNode();

  // Match optional mask cast node
  Node* p_mask_cast = nullptr;
  Node* p_mask_unsqueeze_2 = nullptr;
  Node* p_mask_unsqueeze_1 = nullptr;
  std::vector<graph_utils::EdgeEndToMatch> mask_path_format_1{
      {0, 1, "Cast", {9}, kOnnxDomain},
      {0, 0, "Unsqueeze", {1, 11}, kOnnxDomain},
      {0, 0, "Unsqueeze", {1, 11}, kOnnxDomain}};

  std::vector<graph_utils::EdgeEndToMatch> mask_path_format_2{
      {0, 1, "Unsqueeze", {1, 11}, kOnnxDomain},
      {0, 0, "Unsqueeze", {1, 11}, kOnnxDomain}};

  if (graph_utils::FindPath(mask_sub, true, mask_path_format_1, edges, logger)) {
    p_mask_cast = const_cast<Node*>(&edges[0]->GetNode());
    p_mask_unsqueeze_2 = const_cast<Node*>(&edges[1]->GetNode());
    p_mask_unsqueeze_1 = const_cast<Node*>(&edges[2]->GetNode());
  } else if (graph_utils::FindPath(mask_sub, true, mask_path_format_2, edges, logger)) {
    p_mask_unsqueeze_2 = const_cast<Node*>(&edges[0]->GetNode());
    p_mask_unsqueeze_1 = const_cast<Node*>(&edges[1]->GetNode());
  } else {
    DEBUG_LOG("Failed to find path for mask");
    return false;
  }

  const Node& mask_unsqueeze_2 = *p_mask_unsqueeze_2;
  const Node& mask_unsqueeze_1 = *p_mask_unsqueeze_1;

  if (!optimizer_utils::CheckOutputEdges(graph, softmax, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, mask_add, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, mask_sub, 1) ||
      (p_mask_cast != nullptr && !optimizer_utils::CheckOutputEdges(graph, *p_mask_cast, 1)) ||
      !optimizer_utils::CheckOutputEdges(graph, mask_unsqueeze_2, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, mask_unsqueeze_1, 1)) {
    DEBUG_LOG("Output edge count not expected for mask nodes");
    return false;
  }

  if (!optimizer_utils::IsAttributeWithExpectedValue(softmax, "axis", static_cast<int64_t>(3))) {
    DEBUG_LOG("Softmax attribute axis is expected to be 3");
    return false;
  }

  std::vector<int64_t> axes;
  if (!(graph_utils::GetRepeatedNodeAttributeValues(mask_unsqueeze_1, "axes", axes) && axes.size() == 1 && axes[0] == 1)) {
    DEBUG_LOG("mask_unsqueeze_1 axes not matched. Expect: 1");
    return false;
  }

  if (!(graph_utils::GetRepeatedNodeAttributeValues(mask_unsqueeze_2, "axes", axes) && axes.size() == 1 && axes[0] == 2)) {
    DEBUG_LOG("mask_unsqueeze_2 axes not matched. Expect: 2");
    return false;
  }

  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(mask_sub.InputDefs()[0]), float(1), false)) {
    DEBUG_LOG("mask_sub const input not matched");
    return false;
  }

  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(mask_mul.InputDefs()[1]), float(-10000), false)) {
    DEBUG_LOG("mask_mul const input not matched");
    return false;
  }

  result.has_input_mask = true;
  result.add = &mask_add;
  result.mul = &mask_mul;
  result.sub = &mask_sub;
  result.cast = p_mask_cast;
  result.unsqueeze_2 = p_mask_unsqueeze_2;
  result.unsqueeze_1 = p_mask_unsqueeze_1;
  DEBUG_LOG("Pass MatchInputMaskSubgraph");
  return true;
}

bool MatchInputMaskSubgraph(const Graph& graph, const Node& layer_norm, const Node& qkv_matmul,
                            AttentionMaskNodesDistilBert& result, NodeIndex& record_node_idx, const logging::Logger& logger) {
  DEBUG_LOG("Start MatchInputMaskSubgraphDistilBert");

  std::vector<graph_utils::EdgeEndToMatch> mask_path{
      {0, 0, "Softmax", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Where", {9}, kOnnxDomain},
      {0, 0, "Expand", {8, 13}, kOnnxDomain},
      {0, 0, "Reshape", {1, 5, 13}, kOnnxDomain},
      {0, 0, "Equal", {1, 7, 11, 13}, kOnnxDomain}};

  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(qkv_matmul, true, mask_path, edges, logger)) {
    DEBUG_LOG("Failed to find mask path");
    return false;
  }

  const Node& softmax = edges[0]->GetNode();
  const Node& where = edges[1]->GetNode();
  const Node& expand = edges[2]->GetNode();
  const Node& reshape = edges[3]->GetNode();
  const Node& equal = edges[4]->GetNode();

  if (!optimizer_utils::CheckOutputEdges(graph, softmax, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, where, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, expand, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, reshape, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, equal, 1)) {
    DEBUG_LOG("Output edge count not expected for mask nodes");
    return false;
  }

  if (!optimizer_utils::IsAttributeWithExpectedValue(softmax, "axis", static_cast<int64_t>(3))) {
    DEBUG_LOG("Softmax attribute axis is expected to be 3");
    return false;
  }

  //check where has X=-Infinity
  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(where.InputDefs()[1]), -INFINITY, true)) {
    DEBUG_LOG("where const not matched.");
    return false;
  }

  //expand has another input Shape <-- qk_MatMul
  std::vector<graph_utils::EdgeEndToMatch> shape_path{
      {0, 1, "Shape", {1, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain}};
  if (!graph_utils::FindPath(expand, true, shape_path, edges, logger)) {
    DEBUG_LOG("Failed to find shape path");
    return false;
  }
  const Node& shape = edges[0]->GetNode();
  const Node& qk_matmul = edges[1]->GetNode();
  const Node* p_qk_matmul = graph_utils::GetInputNode(where, 2);
  if (p_qk_matmul == nullptr) {
    return false;
  }
  if (p_qk_matmul->Index() != qk_matmul.Index()) {
    return false;
  }

  //equal has input B=0
  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(equal.InputDefs()[1]), 0.0f, true)) {
    DEBUG_LOG("equal const not matched.");
    return false;
  }

  //reshape node's shape input
  std::vector<graph_utils::EdgeEndToMatch> reshape_shape_path_1{
      {0, 1, "Concat", {4, 11, 13}, kOnnxDomain},
      {0, 0, "Unsqueeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Gather", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Shape", {1, 13}, kOnnxDomain}};
  if (!graph_utils::FindPath(reshape, true, reshape_shape_path_1, edges, logger)) {
    DEBUG_LOG("Failed to find reshape shape path 1");
    return false;
  }
  const Node& concat = edges[0]->GetNode();
  const Node& unsqueeze_1 = edges[1]->GetNode();
  const Node& gather_1 = edges[2]->GetNode();
  const Node& shape_1 = edges[3]->GetNode();

  // check that the recorded unsqueeze in CheckNodesInPathV() is the same node as unsqueeze_1
  if (unsqueeze_1.Index() != record_node_idx) {
    return false;
  }

  std::vector<graph_utils::EdgeEndToMatch> reshape_shape_path_2{
      {0, 3, "Unsqueeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Gather", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Shape", {1, 13}, kOnnxDomain}};
  if (!graph_utils::FindPath(concat, true, reshape_shape_path_2, edges, logger)) {
    DEBUG_LOG("Failed to find reshape shape path 2");
    return false;
  }
  const Node& gather_2 = edges[1]->GetNode();
  const Node& shape_2 = edges[2]->GetNode();
  //check gather has the right indices
  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(gather_1.InputDefs()[1]), static_cast<int64_t>(0), true) ||
      !optimizer_utils::IsInitializerWithExpectedValue(graph, *(gather_2.InputDefs()[1]), static_cast<int64_t>(1), true)) {
    DEBUG_LOG("gather indices not matched.");
    return false;
  }

  // check same root
  if (shape_1.InputDefs().size() != 1 || shape_2.InputDefs().size() != 1) {
    return false;
  }
  const NodeArg& shape_1_arg_0 = *(shape_1.InputDefs()[0]);
  const NodeArg& shape_2_arg_0 = *(shape_2.InputDefs()[0]);
  if (shape_1_arg_0.Name() != shape_2_arg_0.Name()) {
    return false;
  }
  // check the the NodeArg is same as the root input of Attention
  if (layer_norm.OutputDefs().size() < 1) {
    return false;
  }
  const NodeArg& root_input = *(layer_norm.OutputDefs()[0]);
  if (shape_1_arg_0.Name() != root_input.Name()) {
    return false;
  }

  //check concat input shape information
  if (concat.InputDefs().size() != 4) {
    return false;
  }
  std::vector<int64_t> shape_value;
  if (!optimizer_utils::AppendTensorFromInitializer(graph, *(concat.InputDefs()[1]), shape_value, true) ||
      shape_value.size() != 1 ||
      shape_value[0] != 1) {
    return false;
  }
  shape_value.clear();
  if (!optimizer_utils::AppendTensorFromInitializer(graph, *(concat.InputDefs()[2]), shape_value, true) ||
      shape_value.size() != 1 ||
      shape_value[0] != 1) {
    return false;
  }

  result.softmax = &softmax;
  result.where = &where;
  result.expand = &expand;
  result.reshape = &reshape;
  result.equal = &equal;
  result.shape = &shape;

  DEBUG_LOG("Pass MatchInputMaskSubgraphDistilBert");
  return true;
}

struct MatchPastResult {
  NodeArg* past;
  NodeArg* present;
  std::vector<NodeIndex> node_indices;
};

/** Match Past Subgraph
              --> Gather (indices=1) --> v_Concat(*, ) --> Unsqueeze(axes=0)--------------------------------------------------------------------+
             /                                                                                                                                  v
       [Past] --> Gather (indices=0) --> Transpose (perm=0,1,3,2) --> k_Concat(*, )--> Transpose(perm=0,1,3,2) --> Unsqueeze(axes=0)-->Concat(*, ) --> [Present]
*/
bool MatchPastSubgraph(Graph& graph, const Node& k_concat, const Node& v_concat, MatchPastResult& result, const logging::Logger& logger) {
  DEBUG_LOG("Start MatchPastSubgraph");
  std::vector<graph_utils::EdgeEndToMatch> past_k_path{
      {0, 0, "Transpose", {1, 13}, kOnnxDomain},
      {0, 0, "Gather", {1, 11, 13}, kOnnxDomain}};

  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(k_concat, true, past_k_path, edges, logger)) {
    DEBUG_LOG("Failed to find path for past_k");
    return false;
  }
  const Node& past_k_transpose = edges[0]->GetNode();
  const Node& past_k_gather = edges[1]->GetNode();

  std::vector<graph_utils::EdgeEndToMatch> present_k_path{
      {0, 0, "Transpose", {1, 13}, kOnnxDomain},
      {0, 0, "Unsqueeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Concat", {4, 11, 13}, kOnnxDomain}};
  if (!graph_utils::FindPath(k_concat, false, present_k_path, edges, logger)) {
    DEBUG_LOG("Failed to find path for present_k");
    return false;
  }
  const Node& present_k_transpose = edges[0]->GetNode();
  const Node& present_k_unsqueeze = edges[1]->GetNode();
  const Node& present_concat = edges[2]->GetNode();

  std::vector<graph_utils::EdgeEndToMatch> present_past_v_path{
      {0, 1, "Unsqueeze", {1, 11, 13}, kOnnxDomain},
      {0, 0, "Concat", {4, 11, 13}, kOnnxDomain},
      {0, 0, "Gather", {1, 11, 13}, kOnnxDomain}};
  if (!graph_utils::FindPath(present_concat, true, present_past_v_path, edges, logger)) {
    DEBUG_LOG("Failed to find path for present_v and past_v");
    return false;
  }
  const Node& present_v_unsqueeze = edges[0]->GetNode();
  const Node& past_v_concat = edges[1]->GetNode();
  const Node& past_v_gather = edges[2]->GetNode();
  if (past_v_concat.Index() != v_concat.Index()) {
    DEBUG_LOG("Failed to match v_concat");
    return false;
  }

  std::vector<int64_t> perm;
  if (!(graph_utils::GetRepeatedNodeAttributeValues(past_k_transpose, "perm", perm) && perm.size() == 4 && perm[0] == 0 && perm[1] == 1 && perm[2] == 3 && perm[3] == 2)) {
    DEBUG_LOG("past_k_transpose perm attribute not matched");
    return false;
  }

  if (!(graph_utils::GetRepeatedNodeAttributeValues(present_k_transpose, "perm", perm) && perm.size() == 4 && perm[0] == 0 && perm[1] == 1 && perm[2] == 3 && perm[3] == 2)) {
    DEBUG_LOG("present_k_transpose perm attribute not matched");
    return false;
  }

  std::vector<int64_t> axes;
  if (!(graph_utils::GetRepeatedNodeAttributeValues(present_k_unsqueeze, "axes", axes) && axes.size() == 1 && axes[0] == 0)) {
    DEBUG_LOG("present_k_unsqueeze axes value not expected");
    return false;
  }

  if (!(graph_utils::GetRepeatedNodeAttributeValues(present_v_unsqueeze, "axes", axes) && axes.size() == 1 && axes[0] == 0)) {
    DEBUG_LOG("present_v_unsqueeze axes value not expected");
    return false;
  }

  // Check Gather for past_v has indices == 1
  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(past_v_gather.InputDefs()[1]), int64_t(1), true)) {
    DEBUG_LOG("past_v_gather indices != 1");
    return false;
  }

  // Check Gather for past_v has indices == 0
  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(past_k_gather.InputDefs()[1]), int64_t(0), true)) {
    DEBUG_LOG("past_k_gather indices != 0");
    return false;
  }

  if (past_v_gather.InputDefs()[0]->Name() != past_k_gather.InputDefs()[0]->Name()) {
    DEBUG_LOG("past_v_gather and past_k_gather does not have same past input");
    return false;
  }

  if (!optimizer_utils::CheckOutputEdges(graph, k_concat, 2) ||
      !optimizer_utils::CheckOutputEdges(graph, past_k_transpose, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, past_k_gather, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, present_k_transpose, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, present_k_unsqueeze, 1) ||
      present_concat.GetOutputEdgesCount() != 0 ||  // present_concat only has a graph output, but no output edges to other nodes.
      !optimizer_utils::CheckOutputEdges(graph, present_v_unsqueeze, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, past_v_concat, 2) ||
      !optimizer_utils::CheckOutputEdges(graph, past_v_gather, 1)) {
    DEBUG_LOG("Output edge count not expected for nodes in past subgraph");
    return false;
  }
  result.node_indices = {
      k_concat.Index(),
      past_k_transpose.Index(),
      past_k_gather.Index(),
      present_k_transpose.Index(),
      present_k_unsqueeze.Index(),
      present_concat.Index(),
      present_v_unsqueeze.Index(),
      past_v_concat.Index(),
      past_v_gather.Index()};

  result.past = graph.GetNode(past_v_gather.Index())->MutableInputDefs()[0];
  result.present = graph.GetNode(present_concat.Index())->MutableOutputDefs()[0];

  DEBUG_LOG("Pass MatchPastSubgraph");
  return true;
}

bool CheckDistilBertReshapeShape(const Graph& graph, const Node& reshape, int64_t hidden_size, NodeIndex& record_node_idx, const logging::Logger& logger) {
  const Node* p_concat = graph_utils::GetInputNode(reshape, 1);
  if (p_concat == nullptr || (*p_concat).OpType() != "Concat") {
    return false;
  }
  if ((*p_concat).InputDefs().size() != 3) {
    return false;
  }

  // lazy check: record unqueeze first and then check in the mask path
  std::vector<graph_utils::EdgeEndToMatch> shape_path{
      {0, 1, "Concat", {4, 11, 13}, kOnnxDomain},
      {0, 0, "Unsqueeze", {1, 11, 13}, kOnnxDomain}};
  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(reshape, true, shape_path, edges, logger)) {
    DEBUG_LOG("Failed to find shape path");
    return false;
  }
  record_node_idx = edges[1]->GetNode().Index();

  std::vector<int64_t> shape;
  const NodeArg& concat_input_arg_1 = *((*p_concat).InputDefs()[1]);
  if (!optimizer_utils::AppendTensorFromInitializer(graph, concat_input_arg_1, shape) ||
      shape.size() != 1 ||
      shape[0] != -1) {
    return false;
  }

  shape.clear();
  const NodeArg& concat_input_arg_2 = *((*p_concat).InputDefs()[2]);
  if (!optimizer_utils::AppendTensorFromInitializer(graph, concat_input_arg_2, shape) ||
      shape.size() != 1 ||
      shape[0] != hidden_size) {
    return false;
  }

  return true;
}

/** Check the following nodes (optional Concat is excluded) for path v:
                                     v_Reshape  (shape=0,0,H,-1)
                                      |
                                    v_Transpose (perm=0,2,1,3)
                                      |
                                  [p_Concat?]
                         \       /
                       qkv_MatMul
                              |
                           Transpose (perm=0,2,1,3)
                              |
                           Reshape---[shape=0,0,-1]
*/
bool CheckNodesInPathV(const Graph& graph, const Node& reshape, const Node& transpose, const Node& qkv_matmul, const Node& v_transpose, const Node& v_reshape,
                       int64_t& num_heads, int64_t& head_size, int64_t hidden_size, NodeIndex& record_node_idx, const logging::Logger& logger) {
  DEBUG_LOG("Start CheckNodesInPathV");
  // Internal nodes of attention subgraph only allow edges within the subgraph, and no graph output is allowed.
  // No constraints for reshape node since it is the last node of Attention.
  if (!optimizer_utils::CheckOutputEdges(graph, transpose, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, qkv_matmul, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, v_transpose, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, v_reshape, 1)) {
    DEBUG_LOG("Output edge count not expected for nodes in path v");
    return false;
  }
  std::vector<int64_t> perm;
  if (!(graph_utils::GetRepeatedNodeAttributeValues(transpose, "perm", perm) && perm.size() == 4 && perm[0] == 0 && perm[1] == 2 && perm[2] == 1 && perm[3] == 3)) {
    DEBUG_LOG("Failed in match Transpose attribute perm. Expected: 0, 2, 1, 3");
    return false;
  }

  if (!(graph_utils::GetRepeatedNodeAttributeValues(v_transpose, "perm", perm) && perm.size() == 4 && perm[0] == 0 && perm[1] == 2 && perm[2] == 1 && perm[3] == 3)) {
    DEBUG_LOG("Failed in match v_transpose attribute perm. Expected: 0, 2, 1, 3");
    return false;
  }

  if (num_heads > 0 && head_size > 0 && head_size != num_heads * head_size) {
    DEBUG_LOG("hidden_size != num_heads * head_size");
    return false;
  }

  // Check reshape for q, k or v has shape input (0, 0, N, -1) or (0, 0, N, H)
  std::vector<int64_t> v_reshape_shape;
  if (!optimizer_utils::AppendTensorFromInitializer(graph, *(v_reshape.InputDefs()[1]), v_reshape_shape) ||
      v_reshape_shape.size() != 4 ||
      v_reshape_shape[0] != 0 ||
      (v_reshape_shape[1] != 0 && v_reshape_shape[1] != -1) ||  //v_reshape_shape[1] != -1 added for supporting distilbert
      v_reshape_shape[2] <= 0 ||
      v_reshape_shape[2] > hidden_size ||
      (head_size < 0 && v_reshape_shape[3] != -1) ||
      (head_size == 0 && v_reshape_shape[2] * v_reshape_shape[3] != hidden_size)) {
    DEBUG_LOG("v_reshape initializer value is not expected");
    return false;
  }

  num_heads = v_reshape_shape[2];
  head_size = v_reshape_shape[3];

  // Check reshape for attention output has shape input (0, 0, -1) or (0, 0, N*H)
  // In DistilBert, the reshape after qkv paths can not be fused during reshape fusion, so we do not have the correspondig
  // initializer. We need to get the shape information from the input of concat.
  std::vector<int64_t> reshape_shape;
  if (!optimizer_utils::AppendTensorFromInitializer(graph, *(reshape.InputDefs()[1]), reshape_shape)) {
    if (CheckDistilBertReshapeShape(graph, reshape, hidden_size, record_node_idx, logger)) {
      DEBUG_LOG("Pass CheckNodesInPathV");
      return true;
    }
    return false;
  }

  if (reshape_shape.size() != 3 ||
      reshape_shape[0] != 0 ||
      (reshape_shape[1] != 0) ||
      (reshape_shape[2] != num_heads * head_size && reshape_shape[2] != -1)) {
    DEBUG_LOG("reshape initializer value is not expected");
    return false;
  }

  DEBUG_LOG("Pass CheckNodesInPathV");
  return true;
}

bool CheckNodesInPathV(const Graph& graph, const Node& reshape, const Node& transpose, const Node& qkv_matmul, const Node& v_transpose, const Node& v_reshape,
                       int64_t& num_heads, int64_t& head_size, int64_t hidden_size, const logging::Logger& logger) {
  NodeIndex dummy_idx(0);
  if (!CheckNodesInPathV(graph, reshape, transpose, qkv_matmul, v_transpose, v_reshape, num_heads, head_size, hidden_size, dummy_idx, logger)) {
    return false;
  }
  return true;
}

bool CheckNodesInPathQ(const Graph& graph, const Node& qk_div, const Node& q_reshape, const Node& q_transpose, int64_t num_heads, int64_t head_size, const logging::Logger& logger) {
  DEBUG_LOG("Start CheckNodesInPathQ");
  std::vector<int64_t> q_reshape_shape;
  if (!optimizer_utils::AppendTensorFromInitializer(graph, *(q_reshape.InputDefs()[1]), q_reshape_shape) ||
      q_reshape_shape.size() != 4 ||
      q_reshape_shape[0] != 0 ||
      (q_reshape_shape[1] != 0 && q_reshape_shape[1] != -1) ||  //q_reshape_shape[1] != -1 added for supporting distilbert
      q_reshape_shape[2] != num_heads ||
      q_reshape_shape[3] != head_size) {
    DEBUG_LOG("q_reshape const not matched");
    return false;
  }

  float expected_value = std::sqrt(static_cast<float>(head_size));
  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(qk_div.InputDefs()[1]), expected_value, false)) {
    DEBUG_LOG("qk_div const not matched.");
    return false;
  }

  std::vector<int64_t> perm;
  if (!(graph_utils::GetRepeatedNodeAttributeValues(q_transpose, "perm", perm) && perm.size() == 4 && perm[0] == 0 && perm[1] == 2 && perm[2] == 1 && perm[3] == 3)) {
    DEBUG_LOG("q_transpose perm attribute not matched");
    return false;
  }
  DEBUG_LOG("Pass CheckNodesInPathQ");
  return true;
}

bool CheckNodesInPathK(const Graph& graph, const Node& k_reshape, const Node& k_transpose, int64_t num_heads, int64_t head_size, const logging::Logger& logger) {
  DEBUG_LOG("Start CheckNodesInPathK");
  std::vector<int64_t> perm;
  if (!(graph_utils::GetRepeatedNodeAttributeValues(k_transpose, "perm", perm) && perm.size() == 4 && perm[0] == 0 && perm[1] == 2 && perm[2] == 3 && perm[3] == 1)) {
    DEBUG_LOG("k_transpose perm attribute not matched");
    return false;
  }

  std::vector<int64_t> k_reshape_shape;
  if (!optimizer_utils::AppendTensorFromInitializer(graph, *(k_reshape.InputDefs()[1]), k_reshape_shape) ||
      k_reshape_shape.size() != 4 ||
      k_reshape_shape[0] != 0 ||
      (k_reshape_shape[1] != 0 && k_reshape_shape[1] != -1) ||  //k_reshape_shape[1] != -1 added for supporting distilbert
      k_reshape_shape[2] != num_heads ||
      k_reshape_shape[3] != head_size) {
    DEBUG_LOG("k_reshape const not matched");
    return false;
  }
  DEBUG_LOG("Pass CheckNodesInPathK");
  return true;
}

// Add a Cast to convert Mask from int64 to int32.
NodeArg& CastMaskToInt32(Graph& graph, NodeArg* mask_input, ProviderType provider_type) {
  // Derive int32 shape info from mask_input
  TypeProto mask_int32;
  mask_int32.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT32);
  auto dim0 = mask_int32.mutable_tensor_type()->mutable_shape()->add_dim();
  auto dim1 = mask_int32.mutable_tensor_type()->mutable_shape()->add_dim();
  const TensorShapeProto* mask_shape = mask_input->Shape();
  if (mask_shape != nullptr && static_cast<size_t>(mask_shape->dim_size()) == 2) {
    *dim0 = mask_shape->dim(0);
    *dim1 = mask_shape->dim(1);
  }

  NodeArg& cast32 = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("Mask_Int32"), &mask_int32);
  const std::vector<NodeArg*> input_defs{mask_input};
  const std::vector<NodeArg*> output_defs{&cast32};
  Node& node = graph.AddNode(graph.GenerateNodeName("MaskCast"),
                             "Cast",
                             "Cast mask from int64 to int32",
                             input_defs,
                             output_defs,
                             nullptr,
                             kOnnxDomain);

  // Add attribute: "to" = 6
  ONNX_NAMESPACE::AttributeProto to;
  to.set_name("to");
  to.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
  to.set_i(static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_INT32));
  node.AddAttribute("to", to);

  node.SetExecutionProviderType(provider_type);
  return cast32;
}

NodeArg* GetOrCreateMaskInt32(
    Graph& graph,
    NodeArg* mask_input,
    std::map<std::string, NodeArg*>& mask_int32_map,
    ProviderType provider_type) {
  // Lookup in cache map
  auto search = mask_int32_map.find(mask_input->Name());
  if (search != mask_int32_map.end()) {
    return search->second;
  }

  NodeArg& cast32 = CastMaskToInt32(graph, mask_input, provider_type);

  // Add it to cache map.
  mask_int32_map.insert(std::pair<std::string, NodeArg*>(mask_input->Name(), &cast32));
  return &cast32;
}

/** Fuse Attention SubGraph.
@remark add_after_layer_norm is the Add node in the bottom of sub-graph.
 Abbreviatios: B is batch_size, S is sequence_length, W is hidden_size, P is past sequence length,
               N is number of attention heads, H is head size, and W=N*H, h=Sqrt(H)
               B and S could be symbolic. ? means it is optional.
    Graph before Fusion (q_, k_, v_, qk_, qkv_ and mask_ prefix is added before Operator type):
                  Add
               /       \ [Input](BxSxW)
              /         \
             /   LayerNormalization
            /            |
           /       {Gemm_Subgraph} <---[weights](Wx3W); [Bias](3W)
          |               |
          |             Split
          |          /     |     \
          |         /      |      \
          | q_Reshape   k_Reshape   v_Reshape  (shape=0,0,H,-1)
          |         |        |        |
          |q_Transpose  k_Transpose v_Transpose
          |  (0,2,1,3)  (0,2,3,1)    (perm=0,2,1,3)
          |   \          /            |                       [Past]?
               \        /             |                          |
          |     \    k_Concat? <------|---------------------{Past_Subgraphj}?
          |      \    /               |                          |
          |      qk_MatMul            |                          |
          |           |    [B=h]      |                          |
          |           |   /           |                         /
          |        qk_Div         v_Concat? <------------------
          |            |              |
          | {Unidir_Mask_Subgraph}    |                             [Mask]?
          |            |              /                               |
          |       mask_Add? <--------/---------------------{Attention_Mask_Subgraph}?
          |            |            /
          |          Softmax       /
          |             \         /
          |              \       /
          |            qkv_MatMul
          |                   |
          |                Transpose (perm=0,2,1,3)
          |                   |
          |                Reshape---[shape=0,0,-1]
          |                   |
          |                 {Gemm_Subgraph} <---[weights](WxW); [Bias](W)
          |                  /
          +--------------> Add

After Fusion:

      Add
      |   \
      |  LayerNormalization [Weights] [Bias]   [Mask]?  [Past]?
      |                 \   |        /         /         /
       \                 \  |       /         /         /
        \                 Attention <------------------
         \                |       |
          \      {Gemm_Subgraph}  v
           \              |       [Present]?
            \             |
             \            /
              --------> Add
TODO: replace Gemm_Subgraph by MatMul + Add
*/
bool FuseGptAttention(Node& layer_norm, Graph& graph, int64_t hidden_size, std::map<std::string, NodeArg*>& mask_int32_map, bool use_shared_node, const logging::Logger& logger) {
  DEBUG_LOG("Start FuseGptAttention");
  const Node* parent_node = graph_utils::GetInputNode(layer_norm, 0);
  if (nullptr == parent_node || !graph_utils::IsSupportedOptypeVersionAndDomain(*parent_node, "Add", {7, 13, 14}, kOnnxDomain)) {
    return false;
  }

  const Node* add_after_gemm = graph_utils::FirstChildByType(*graph.GetNode(parent_node->Index()), "Add");
  if (nullptr == add_after_gemm) {
    return false;
  }

  MatchGemmResult gemm1_result;
  if (!MatchGemmSubgraph(graph, *graph.GetNode(add_after_gemm->Index()), 1, gemm1_result, use_shared_node, logger) ||
      !ValidateGemmInitializer(graph, *gemm1_result.gemm, hidden_size, false, logger)) {
    return false;
  }

  std::vector<graph_utils::EdgeEndToMatch> path1{
      {0, 0, "Reshape", {5, 13}, kOnnxDomain},
      {0, 0, "Transpose", {1, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9}, kOnnxDomain}};

  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(*gemm1_result.input_node, true, path1, edges, logger)) {
    DEBUG_LOG("Faild to find path to qkv_matmul");
    return false;
  }

  const Node& reshape = edges[0]->GetNode();
  const Node& transpose = edges[1]->GetNode();
  const Node& qkv_matmul = edges[2]->GetNode();

  const Node* v_concat = graph_utils::GetInputNode(qkv_matmul, 1);
  if (v_concat == nullptr) {
    return false;
  }

  bool has_past = graph_utils::IsSupportedOptypeVersionAndDomain(*v_concat, "Concat", {4, 11, 13}, kOnnxDomain);

  std::vector<graph_utils::EdgeEndToMatch> path2{
      {0, 1, "Transpose", {1, 13}, kOnnxDomain},
      {0, 0, "Reshape", {5, 13}, kOnnxDomain},
      {2, 0, "Split", {2, 11, 13}, kOnnxDomain}};

  if (!graph_utils::FindPath(has_past ? *v_concat : qkv_matmul, true, path2, edges, logger)) {
    DEBUG_LOG("Faild to find path v to Split");
    return false;
  }

  const Node& v_transpose = edges[0]->GetNode();
  const Node& v_reshape = edges[1]->GetNode();
  const Node& v_split = edges[2]->GetNode();

  MatchGemmResult gemm0_result;
  if (!MatchGemmSubgraph(graph, *graph.GetNode(v_split.Index()), 0, gemm0_result, use_shared_node, logger) ||
      !ValidateGemmInitializer(graph, *gemm0_result.gemm, hidden_size, true, logger)) {
    return false;
  }

  const Node* gemm0_parent = graph_utils::GetInputNode(*gemm0_result.input_node, 0);
  if (gemm0_parent == nullptr || gemm0_parent->Index() != layer_norm.Index()) {
    return false;
  }

  int64_t num_heads = 0;  // will be updated in CheckNodesInPathV
  int64_t head_size = -1;
  if (!CheckNodesInPathV(graph, reshape, transpose, qkv_matmul, v_transpose, v_reshape, num_heads, head_size, hidden_size, logger)) {
    DEBUG_LOG("CheckNodesInPathV return false");
    return false;
  }

  if (!optimizer_utils::CheckOutputEdges(graph, v_split, 3)) {
    DEBUG_LOG("Output edge count not expected for nodes in path v");
    return false;
  }

  // Find input mask. Unsqueeze -> Unsqueeze -> (Cast) -> Sub -> Mul -> Add -> Softmax
  AttentionMaskNodes mask_nodes;
  if (!MatchInputMaskSubgraph(graph, qkv_matmul, mask_nodes, logger, true)) {
    DEBUG_LOG("MatchInputMaskSubgraph returns false");
    return false;
  }

  MatchUnidirMaskResult unidir_mask_result;
  if (!MatchUnidirMaskSubgraph(graph, *(mask_nodes.has_input_mask ? mask_nodes.add : mask_nodes.softmax), unidir_mask_result, use_shared_node, logger)) {
    DEBUG_LOG("MatchUnidirMaskSubgraph returns NULL");
    return false;
  }

  // path to q
  std::vector<graph_utils::EdgeEndToMatch> q_path{
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
      {0, 0, "Transpose", {1, 13}, kOnnxDomain},
      {0, 0, "Reshape", {5, 13}, kOnnxDomain},
      {0, 0, "Split", {2, 11, 13}, kOnnxDomain}};

  const Node* qk_div = unidir_mask_result.div_node;
  if (!graph_utils::FindPath(*qk_div, true, q_path, edges, logger)) {
    DEBUG_LOG("Failed to find path for q");
    return false;
  }

  const Node& qk_matmul = edges[0]->GetNode();
  const Node& q_transpose = edges[1]->GetNode();
  const Node& q_reshape = edges[2]->GetNode();
  const Node& q_split = edges[3]->GetNode();
  if (q_split.Index() != v_split.Index()) {
    DEBUG_LOG("q and v are not from same Split node");
    return false;
  }

  if (!CheckNodesInPathQ(graph, *qk_div, q_reshape, q_transpose, num_heads, head_size, logger)) {
    DEBUG_LOG("CheckNodesInPathQ returns false");
    return false;
  }

  const Node* k_concat = nullptr;
  if (has_past) {
    k_concat = graph_utils::GetInputNode(qk_matmul, 1);
    if (k_concat == nullptr || !graph_utils::IsSupportedOptypeVersionAndDomain(*k_concat, "Concat", {4, 11, 13}, kOnnxDomain)) {
      return false;
    }
  }

  // path to k
  std::vector<graph_utils::EdgeEndToMatch> k_path{
      {0, 1, "Transpose", {1, 13}, kOnnxDomain},
      {0, 0, "Reshape", {5, 13}, kOnnxDomain},
      {1, 0, "Split", {2, 11, 13}, kOnnxDomain}};

  if (!graph_utils::FindPath(has_past ? *k_concat : qk_matmul, true, k_path, edges, logger)) {
    DEBUG_LOG("Failed to find path for k");
    return false;
  }

  const Node& k_transpose = edges[0]->GetNode();
  const Node& k_reshape = edges[1]->GetNode();
  const Node& k_split = edges[2]->GetNode();
  if (k_split.Index() != v_split.Index()) {
    DEBUG_LOG("k and v are not from same Split node");
    return false;
  }

  if (!CheckNodesInPathK(graph, k_reshape, k_transpose, num_heads, head_size, logger)) {
    DEBUG_LOG("CheckNodesInPathK returns false");
    return false;
  }

  MatchPastResult past_result;
  if (has_past && !MatchPastSubgraph(graph, *k_concat, *v_concat, past_result, logger)) {
    DEBUG_LOG("MatchPastSubgraph returns false");
    return false;
  }

  // Now everything is ready, we will start fusing subgraph.
  NodeArg* qkv_weights = graph.GetNode(gemm0_result.gemm->Index())->MutableInputDefs()[1];
  NodeArg* qkv_bias = graph.GetNode(gemm0_result.gemm->Index())->MutableInputDefs()[2];

  // Create Attention Node.
  std::vector<NodeArg*> input_defs{layer_norm.MutableOutputDefs()[0], qkv_weights, qkv_bias};
  std::vector<NodeArg*> output_defs{graph.GetNode(reshape.Index())->MutableOutputDefs()[0]};

  if (mask_nodes.has_input_mask) {
    NodeArg* mask_input = graph.GetNode(mask_nodes.unsqueeze_1->Index())->MutableInputDefs()[0];
    NodeArg* mask_int32 = GetOrCreateMaskInt32(graph, mask_input, mask_int32_map, layer_norm.GetExecutionProviderType());
    input_defs.push_back(mask_int32);
  } else {
    // Add a missing optional input for mask.
    std::string empty_name;
    input_defs.push_back(&graph.GetOrCreateNodeArg(empty_name, nullptr));
  }

  if (has_past) {
    input_defs.push_back(past_result.past);
    output_defs.push_back(past_result.present);
  }

  Node& attention_node = graph.AddNode(
      graph.GenerateNodeName("Attention"),
      "Attention",
      "Fused Attention subgraphs ",
      input_defs,
      output_defs,
      nullptr,
      kMSDomain);
  attention_node.AddAttribute("num_heads", num_heads);
  attention_node.AddAttribute("unidirectional", static_cast<int64_t>(unidir_mask_result.is_unidirectional));

  // Assign provider to this new node.
  attention_node.SetExecutionProviderType(layer_norm.GetExecutionProviderType());

  // Remove nodes that are not used anymore.
  std::vector<NodeIndex> nodes_to_remove{
      reshape.Index(),
      transpose.Index(),
      qkv_matmul.Index(),
      v_transpose.Index(),
      v_reshape.Index(),
      v_split.Index(),
      qk_div->Index(),
      qk_matmul.Index(),
      q_transpose.Index(),
      q_reshape.Index(),
      k_transpose.Index(),
      k_reshape.Index()};

  nodes_to_remove.insert(nodes_to_remove.end(), unidir_mask_result.node_indices.begin(), unidir_mask_result.node_indices.end());
  nodes_to_remove.insert(nodes_to_remove.end(), gemm0_result.node_indices.begin(), gemm0_result.node_indices.end());
  if (has_past) {
    nodes_to_remove.insert(nodes_to_remove.end(), past_result.node_indices.begin(), past_result.node_indices.end());
  }
  SetMaskNodesToRemove(graph, mask_nodes, nodes_to_remove);

  for (const auto& node_index : nodes_to_remove) {
    Node* node = graph.GetNode(node_index);
    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(node->Index());
  }

  DEBUG_LOG("Fused an attention node for GPT.");
  return true;
}

};  // namespace AttentionFusionHelper

}  // namespace onnxruntime
