// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/conv_bn_fusion.h"
#include "core/optimizer/utils.h"

#include <limits>
#include <type_traits>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

namespace {

constexpr float kDefaultBatchNormalizationEpsilon = 1e-5f;

int64_t GetGroup(const Node& node) {
  const auto* group_attr = graph_utils::GetNodeAttribute(node, "group");
  return group_attr != nullptr && utils::HasInt(*group_attr) ? group_attr->i() : 1;
}

bool IsValidWeightShapeForFusion(const TensorProto& weight,
                                 int64_t bn_channels,
                                 bool is_conv_transpose,
                                 int64_t group) {
  if (weight.dims_size() <= 2) {
    return false;
  }

  if (!is_conv_transpose) {
    return weight.dims(0) == bn_channels;
  }

  if (group <= 0 ||
      weight.dims(0) <= 0 ||
      weight.dims(1) <= 0 ||
      weight.dims(0) % group != 0 ||
      weight.dims(1) > std::numeric_limits<int64_t>::max() / group) {
    return false;
  }

  return weight.dims(1) * group == bn_channels;
}

template <typename T>
void ScaleConvTransposeWeightData(Initializer& weight, const Initializer& scalers, int64_t group) {
  const auto dims = weight.dims();
  ORT_ENFORCE(dims.size() > 2, "ConvTranspose weight should have at least 3 dimensions.");
  ORT_ENFORCE(group > 0, "ConvTranspose group should be positive.");

  const int64_t input_channels = dims[0];
  const int64_t output_channels_per_group = dims[1];
  ORT_ENFORCE(input_channels > 0 && output_channels_per_group > 0, "Invalid ConvTranspose weight shape.");
  ORT_ENFORCE(input_channels % group == 0, "Invalid ConvTranspose group for weight shape.");
  ORT_ENFORCE(output_channels_per_group <= std::numeric_limits<int64_t>::max() / group,
              "Invalid ConvTranspose output channel count.");

  const int64_t input_channels_per_group = input_channels / group;
  ORT_ENFORCE(static_cast<size_t>(output_channels_per_group * group) == scalers.size(),
              "Invalid ConvTranspose channel scaler size.");

  size_t kernel_size = 1;
  for (size_t i = 2; i < dims.size(); ++i) {
    ORT_ENFORCE(dims[i] > 0, "Invalid ConvTranspose kernel shape.");
    kernel_size *= static_cast<size_t>(dims[i]);
  }

  using Numeric = std::conditional_t<std::is_same_v<T, double>, double, float>;
  T* weight_data = weight.data<T>();
  const T* scaler_data = scalers.data<T>();

  for (int64_t input_channel = 0; input_channel < input_channels; ++input_channel) {
    const int64_t group_index = input_channel / input_channels_per_group;
    for (int64_t output_channel = 0; output_channel < output_channels_per_group; ++output_channel) {
      const auto scaler_index = static_cast<size_t>(group_index * output_channels_per_group + output_channel);
      const Numeric scale = static_cast<Numeric>(scaler_data[scaler_index]);
      const size_t weight_offset =
          (static_cast<size_t>(input_channel) * static_cast<size_t>(output_channels_per_group) +
           static_cast<size_t>(output_channel)) *
          kernel_size;

      for (size_t kernel_index = 0; kernel_index < kernel_size; ++kernel_index) {
        T& value = weight_data[weight_offset + kernel_index];
        value = T(static_cast<Numeric>(value) * scale);
      }
    }
  }
}

void ScaleConvTransposeWeightByOutputChannel(Initializer& weight, const Initializer& scalers, int64_t group) {
  switch (weight.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      ScaleConvTransposeWeightData<float>(weight, scalers, group);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      ScaleConvTransposeWeightData<MLFloat16>(weight, scalers, group);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      ScaleConvTransposeWeightData<double>(weight, scalers, group);
      break;
    default:
      ORT_ENFORCE(false, "Unsupported ConvTranspose weight data type for BN fusion.");
  }
}

}  // namespace

Status ConvBNFusion::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  auto& conv_node = node;
  Node& bn_node = *graph.GetNode(conv_node.OutputNodesBegin()->Index());
  const bool is_conv_transpose = conv_node.OpType() == "ConvTranspose";
  const int64_t group = is_conv_transpose ? GetGroup(conv_node) : 1;

  float epsilon = kDefaultBatchNormalizationEpsilon;
  const auto* attr = graph_utils::GetNodeAttribute(bn_node, "epsilon");
  if (attr != nullptr) {
    if (!utils::HasFloat(*attr)) {
      return Status::OK();
    }
    epsilon = static_cast<float>(attr->f());
  }

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

  // Conv and ConvTranspose only support floating point data types, so can only fuse with initializers containing those types
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
      conv_W_tensor_proto->data_type() != bn_scale_tensor_proto->data_type()) {
    return Status::OK();
  }

  if (!IsValidWeightShapeForFusion(*conv_W_tensor_proto, bn_scale_tensor_proto->dims(0), is_conv_transpose, group)) {
    return Status::OK();
  }

  Initializer bn_scale{graph, *bn_scale_tensor_proto, graph.ModelPath()};
  Initializer bn_B{graph, *bn_B_tensor_proto, graph.ModelPath()};
  Initializer bn_mean{graph, *bn_mean_tensor_proto, graph.ModelPath()};
  Initializer bn_var{graph, *bn_var_tensor_proto, graph.ModelPath()};
  Initializer conv_W{graph, *conv_W_tensor_proto, graph.ModelPath()};

  std::optional<Initializer> conv_B;
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
    conv_B.emplace(graph, *conv_B_tensor_proto, graph.ModelPath());
  }

  // Calculate new value of initializers of conv node
  bn_var.add(epsilon);
  bn_var.sqrt();
  bn_scale.div(bn_var);
  if (is_conv_transpose) {
    ScaleConvTransposeWeightByOutputChannel(conv_W, bn_scale, group);
  } else {
    conv_W.scale_by_axis(bn_scale, 1);
  }

  if (conv_inputs.size() == 3) {
    conv_B->sub(bn_mean);
    conv_B->mul(bn_scale);
    conv_B->add(bn_B);
  } else {
    bn_mean.mul(bn_scale);
    bn_B.sub(bn_mean);
  }

  // Create new initializers of conv
  ONNX_NAMESPACE::TensorProto new_conv_W_tensor_proto;
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

  NodeArg& new_conv_W_node_arg = graph_utils::AddInitializerWithOrtValue(graph, new_conv_W_tensor_proto);
  graph_utils::ReplaceNodeInput(node, 1, new_conv_W_node_arg);

  auto& new_conv_B_node_arg = graph_utils::AddInitializerWithOrtValue(graph, new_conv_B_tensor_proto);

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
  const bool is_supported_conv = graph_utils::IsSupportedOptypeVersionAndDomain(node, "Conv", {1, 11, 22});
  const bool is_supported_conv_transpose =
      graph_utils::IsSupportedOptypeVersionAndDomain(node, "ConvTranspose", {1, 11, 22});

  if ((!is_supported_conv && !is_supported_conv_transpose) ||
      node.GetOutputEdgesCount() != 1) {
    return false;
  }

  const auto& next_node = *node.OutputNodesBegin();
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "BatchNormalization", {7, 9, 14, 15}) ||
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

  if (graph.NodeProducesGraphOutput(node)) {
    return false;
  }

  return true;
}

}  // namespace onnxruntime
