// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/conv_bn_fusion.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

Status ConvBNFusion::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  auto& conv_node = node;
  Node& bn_node = *graph.GetNode(conv_node.OutputNodesBegin()->Index());

  // Get value of attribute epsilon
  const onnxruntime::NodeAttributes& attributes = bn_node.GetAttributes();
  const ONNX_NAMESPACE::AttributeProto* attr = &(attributes.find("epsilon")->second);
  if (attr == nullptr || attr->type() != AttributeProto_AttributeType_FLOAT) {
    return Status::OK();
  }
  float epsilon = static_cast<float>(attr->f());

  // Get initializers of BatchNormalization
  const auto& bn_inputs = bn_node.InputDefs();
  const auto* bn_scale_tensor_proto = graph_utils::GetConstantInitializer(graph, bn_inputs[1]->Name());
  ORT_ENFORCE(bn_scale_tensor_proto);

  const auto* bn_B_tensor_proto = graph_utils::GetConstantInitializer(graph, bn_inputs[2]->Name());
  ORT_ENFORCE(bn_B_tensor_proto);

  const auto* bn_mean_tensor_proto = graph_utils::GetConstantInitializer(graph, bn_inputs[3]->Name());
  ORT_ENFORCE(bn_mean_tensor_proto);

  const auto* bn_var_tensor_proto = graph_utils::GetConstantInitializer(graph, bn_inputs[4]->Name());
  ORT_ENFORCE(bn_var_tensor_proto);

  const auto& conv_inputs = conv_node.InputDefs();
  const auto* conv_W_tensor_proto = graph_utils::GetConstantInitializer(graph, conv_inputs[1]->Name());
  ORT_ENFORCE(conv_W_tensor_proto);

  // Conv only supports floating point data types, so can only fuse with an initializer containing those types
  if (!optimizer_utils::IsFloatingPointDataType(*bn_scale_tensor_proto) ||
      !optimizer_utils::IsFloatingPointDataType(*bn_B_tensor_proto) ||
      !optimizer_utils::IsFloatingPointDataType(*bn_mean_tensor_proto) ||
      !optimizer_utils::IsFloatingPointDataType(*bn_var_tensor_proto) ||
      !optimizer_utils::IsFloatingPointDataType(*conv_W_tensor_proto) ||
      bn_scale_tensor_proto->dims_size() != 1 ||
      bn_B_tensor_proto->dims_size() != 1 ||
      bn_mean_tensor_proto->dims_size() != 1 ||
      bn_var_tensor_proto->dims_size() != 1 ||
      bn_scale_tensor_proto->dims(0) != bn_B_tensor_proto->dims(0) ||
      bn_B_tensor_proto->dims(0) != bn_mean_tensor_proto->dims(0) ||
      bn_mean_tensor_proto->dims(0) != bn_var_tensor_proto->dims(0) ||
      bn_scale_tensor_proto->data_type() != bn_B_tensor_proto->data_type() ||
      bn_B_tensor_proto->data_type() != bn_mean_tensor_proto->data_type() ||
      bn_mean_tensor_proto->data_type() != bn_var_tensor_proto->data_type() ||
      conv_W_tensor_proto->data_type() != bn_scale_tensor_proto->data_type() ||
      !(conv_W_tensor_proto->dims_size() > 2 && conv_W_tensor_proto->dims(0) == bn_scale_tensor_proto->dims(0))) {
    return Status::OK();
  }

  Initializer bn_scale{*bn_scale_tensor_proto, graph.ModelPath()};
  Initializer bn_B{*bn_B_tensor_proto, graph.ModelPath()};
  Initializer bn_mean{*bn_mean_tensor_proto, graph.ModelPath()};
  Initializer bn_var{*bn_var_tensor_proto, graph.ModelPath()};
  Initializer conv_W{*conv_W_tensor_proto, graph.ModelPath()};

  std::unique_ptr<Initializer> conv_B = nullptr;
  const ONNX_NAMESPACE::TensorProto* conv_B_tensor_proto = nullptr;
  if (conv_inputs.size() == 3) {
    conv_B_tensor_proto = graph_utils::GetConstantInitializer(graph, conv_inputs[2]->Name());
    ORT_ENFORCE(conv_B_tensor_proto);

    if (!optimizer_utils::IsFloatingPointDataType(*conv_B_tensor_proto) ||
        conv_B_tensor_proto->dims_size() != 1 ||
        conv_B_tensor_proto->dims(0) != bn_B_tensor_proto->dims(0) ||
        conv_B_tensor_proto->data_type() != bn_B_tensor_proto->data_type()) {
      return Status::OK();
    }
    conv_B = std::make_unique<Initializer>(*conv_B_tensor_proto, graph.ModelPath());
  }

  // Calculate new value of initializers of conv node
  bn_var.add(epsilon);
  bn_var.sqrt();
  bn_scale.div(bn_var);
  conv_W.scale_by_axis(bn_scale, 1);

  if (conv_inputs.size() == 3) {
    conv_B->sub(bn_mean);
    conv_B->mul(bn_scale);
    conv_B->add(bn_B);
  } else {
    bn_mean.mul(bn_scale);
    bn_B.sub(bn_mean);
  }

  // Create new initializers of conv
  ONNX_NAMESPACE::TensorProto new_conv_W_tensor_proto(*conv_W_tensor_proto);
  conv_W.ToProto(new_conv_W_tensor_proto);

  ONNX_NAMESPACE::TensorProto new_conv_B_tensor_proto;
  NodeArg* bn_B_node_arg = nullptr;
  if (conv_inputs.size() == 3) {
    conv_B->ToProto(new_conv_B_tensor_proto);
  } else {
    bn_B.ToProto(new_conv_B_tensor_proto);
    bn_B_node_arg = graph.GetNodeArg(bn_B_tensor_proto->name());
    if (bn_B_node_arg == nullptr) {
      return Status::OK();
    }
  }

  // Replace initializers of conv node
  auto new_W_name = graph.GenerateNodeArgName("ConvBnFusion_W_" + conv_W_tensor_proto->name());
  auto new_B_name = graph.GenerateNodeArgName("ConvBnFusion_BN_B_" + bn_B_tensor_proto->name());

  new_conv_W_tensor_proto.set_name(new_W_name);
  new_conv_B_tensor_proto.set_name(new_B_name);

  NodeArg& new_conv_W_node_arg = graph_utils::AddInitializer(graph, new_conv_W_tensor_proto);
  graph_utils::ReplaceNodeInput(node, 1, new_conv_W_node_arg);

  auto& new_conv_B_node_arg = graph_utils::AddInitializer(graph, new_conv_B_tensor_proto);

  if (conv_inputs.size() == 3) {
    graph_utils::ReplaceNodeInput(node, 2, new_conv_B_node_arg);
  } else {
    graph_utils::AddNodeInput(node, 2, new_conv_B_node_arg);
  }

  // trim off any output defs that are optional in the bn_node before we finalize fusion, as we copy the '
  // defs across to the Conv node so the output name is maintained. we checked in SatisfyCondition that
  // none of these optional outputs exist, so it's safe to do this.
  bn_node.MutableOutputDefs().resize(1);

  // Move the output definition and edges from the BN node to the Conv node and delete the BN node.
  graph_utils::FinalizeNodeFusion(graph, conv_node, bn_node);

  rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;

  return Status::OK();
}

bool ConvBNFusion::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger&) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Conv", {1, 11}) ||
      node.GetOutputEdgesCount() != 1) {
    return false;
  }

  const auto& next_node = *node.OutputNodesBegin();
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "BatchNormalization", {7, 9, 14}) ||
      next_node.GetInputEdgesCount() != 1 ||
      // Make sure the two nodes do not span execution providers.
      next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
    return false;
  }

  // Check that the appropriate inputs to the Conv and BN nodes are constants.
  if (!graph_utils::NodeArgIsConstant(graph, *node.InputDefs()[1]) ||
      (node.InputDefs().size() == 3 && !graph_utils::NodeArgIsConstant(graph, *node.InputDefs()[2])) ||
      !graph_utils::NodeArgIsConstant(graph, *next_node.InputDefs()[1]) ||
      !graph_utils::NodeArgIsConstant(graph, *next_node.InputDefs()[2]) ||
      !graph_utils::NodeArgIsConstant(graph, *next_node.InputDefs()[3]) ||
      !graph_utils::NodeArgIsConstant(graph, *next_node.InputDefs()[4])) {
    return false;
  }

  // First output from BN is required. Others are optional. If any optional outputs exist we can't fuse.
  const auto& output_defs = next_node.OutputDefs();
  if (output_defs.size() > 1) {
    for (size_t i = 1, end = output_defs.size(); i < end; ++i) {
      if (output_defs[i] != nullptr && output_defs[i]->Exists())
        return false;
    }
  }

  if (!graph.GetNodeOutputsInGraphOutputs(node).empty()) {
    return false;
  }

  return true;
}

}  // namespace onnxruntime
