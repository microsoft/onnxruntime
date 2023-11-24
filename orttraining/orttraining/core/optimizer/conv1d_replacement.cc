// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include "core/optimizer/initializer.h"
#include "orttraining/core/optimizer/conv1d_replacement.h"
#include "core/graph/graph_utils.h"

/*
  In LoRA code, it will use conv1d to do projection for qkv,
  while the conv1d calculation is mathematically equivalent to MatMul, and MatMul is much faster than conv1d in GPU.
  The graph transformation is doing the following graph substitution:
  1. The input graph is:
  conv_input  conv_weight
        \       /
         \     /
          conv1d

  2. The output graph is as follows,
     the number of MatMul is equal to attribute "group" of conv1d
        conv_input   conv1d.group    conv_weight  conv1d.group
          \          /                   \         /
           \        /                   Squeeze   /
            \      /                       \     /
              Split                         Split
          /   /  ... \                   /   /   ... \
         /   /    ... \                 /   /     ... \
        /   /      ... \               /   /       ... \
    input0  input1 ... inputN     weight0 weight1  ... weightN
        \      \          \           /    /           /
          \       \          \       /    /          /
            \       \          \   /     /         /
              \       \          X      /        /
                \       \       /  \   /        /
                  \       \   /      X        /
                    \       X       / \     /
                      \   /   \   /     \  /
                     MatMul   MatMul ... MatMul
                        \       |     ... /
                          \     |       /
                            \   |     /
*/
namespace onnxruntime {
bool NodeCanBeReplacedByMatmul(const Node& node) {
  // If node type is Conv, and attr "dilations" is 1, "kernel_shape" is 1, "stride" is 1, group is 1 or 2,
  // then it can be replaced by MatMul
  // Kernel_shape is 1 means it is conv1d
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Conv", {1, 11})) {
    return false;
  }
  const auto* dilations = graph_utils::GetNodeAttribute(node, "dilations");
  const auto* kernel_shape = graph_utils::GetNodeAttribute(node, "kernel_shape");
  const auto* stride = graph_utils::GetNodeAttribute(node, "strides");
  const auto* group = graph_utils::GetNodeAttribute(node, "group");
  if (dilations == nullptr || kernel_shape == nullptr || stride == nullptr || group == nullptr) {
    return false;
  }
  if ((dilations->ints_size() && dilations->ints(0) != 1) ||
      (kernel_shape->ints_size() && kernel_shape->ints(0) != 1) ||
      (stride->ints_size() && stride->ints(0) != 1) ||
      group->i() >= 3) {
    return false;
  }

  return true;
}

void Conv1dToMatmul(Graph& graph, Node& conv) {
  // Shape of conv1d input: [batch_size, in_channels, in_length]
  // Shape of conv1d weight:[output_channels, input_channels/group, kernel_shape], kernel_shape is 1
  // We need to split the input into "group", and squeeze&split the weight, and then do MatMul
  const std::string node_description("Conv1dReplacement");
  auto execution_provider_type = conv.GetExecutionProviderType();
  // 1. Split conv input
  auto group_attr = graph_utils::GetNodeAttribute(conv, "group");
  int64_t group_num = 1;  // default group is 1 from ONNX schema
  if (group_attr != nullptr) {
    group_num = group_attr->i();
  }
  auto conv1d_input = conv.MutableInputDefs()[0];
  std::vector<onnxruntime::NodeArg*> conv1d_input_splitted_outputs;
  for (int i = 0; i < group_num; i++) {
    conv1d_input_splitted_outputs.push_back(&graph.GetOrCreateNodeArg(
        graph.GenerateNodeArgName("input_split_output"), nullptr));
  }
  auto& input_split = graph.AddNode(graph.GenerateNodeName("Split"), "Split", node_description, {conv1d_input},
                                    {conv1d_input_splitted_outputs});
  input_split.SetExecutionProviderType(execution_provider_type);
  input_split.AddAttribute("axis", int64_t(1));
  auto onnx_opset_version = graph.DomainToVersionMap().at(kOnnxDomain);
  if (onnx_opset_version >= 18) {
    input_split.AddAttribute("num_outputs", group_num);
  }
  // 2. Squeeze conv weight
  auto conv1d_weight = conv.MutableInputDefs()[1];
  auto weight_squeeze_output = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("weight_squeeze_output"), nullptr);
  auto& weight_squeeze = graph.AddNode(graph.GenerateNodeName("WeightSqueeze"), "Squeeze",
                                       node_description, {conv1d_weight}, {weight_squeeze_output});
  if (onnx_opset_version > 12) {
    // After onnx version 12, squeeze node has axes as input instead of attribute
    ONNX_NAMESPACE::TensorProto initializer_proto;
    initializer_proto.set_name(graph.GenerateNodeName("ConstAsInitializer"));
    initializer_proto.add_dims(static_cast<int64_t>(1));
    initializer_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    InlinedVector<int64_t> initializer_proto_value{2};
    initializer_proto.set_raw_data(initializer_proto_value.data(), initializer_proto_value.size() * sizeof(int64_t));
    auto& axes_input = graph_utils::AddInitializer(graph, initializer_proto);
    // Squeeze node doesn't have opschema here, so we need to set input args count manually
    weight_squeeze.MutableInputArgsCount().resize(2);
    graph_utils::AddNodeInput(weight_squeeze, 1, axes_input);
  } else {
    weight_squeeze.AddAttribute("axes", std::vector<int64_t>{2});
  }
  weight_squeeze.SetExecutionProviderType(execution_provider_type);
  // 3. Split conv weight
  std::vector<onnxruntime::NodeArg*> conv1d_weight_splitted_outputs;
  for (int i = 0; i < group_num; i++) {
    conv1d_weight_splitted_outputs.push_back(&graph.GetOrCreateNodeArg(
        graph.GenerateNodeArgName("weight_split_output"), nullptr));
  }
  auto& weight_split = graph.AddNode(graph.GenerateNodeName("Split"), "Split", node_description,
                                     {weight_squeeze_output}, {conv1d_weight_splitted_outputs});
  weight_split.AddAttribute("axis", int64_t(0));
  weight_split.SetExecutionProviderType(execution_provider_type);
  if (onnx_opset_version >= 18) {
    weight_split.AddAttribute("num_outputs", group_num);
  }
  // 4. Do MatMul
  std::vector<onnxruntime::NodeArg*> matmul_outputs;
  for (int i = 0; i < group_num; i++) {
    auto matmul_output = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("matmul_output"), nullptr);
    matmul_outputs.push_back(matmul_output);
    auto& matmul = graph.AddNode(graph.GenerateNodeName("Matmul"), "MatMul", node_description,
                                 {conv1d_weight_splitted_outputs[i], conv1d_input_splitted_outputs[i]},
                                 {matmul_output});
    matmul.SetExecutionProviderType(execution_provider_type);
  }
  // 5. Concat matmul outputs
  auto& concat_node = graph.AddNode(graph.GenerateNodeName("Concat"), "Concat", node_description,
                                    matmul_outputs, {});
  concat_node.SetExecutionProviderType(execution_provider_type);
  concat_node.AddAttribute("axis", int64_t(1));
  // 6. Clean up - delted original "conv" node, its output is replaced by concat_node
  graph_utils::FinalizeNodeFusion(graph, concat_node, conv);
}

Status Conv1dReplacement::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (!node_ptr)
      continue;  // node was removed
    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
    if (NodeCanBeReplacedByMatmul(node)) {
      LOGS(logger, VERBOSE) << "lora conv1d replacement, node name: " + node.Name();
      Conv1dToMatmul(graph, node);
      modified = true;
    }
  }
  return Status::OK();
}
}  // namespace onnxruntime
