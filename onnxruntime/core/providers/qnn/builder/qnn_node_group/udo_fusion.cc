// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/udo_fusion.h"

#include <gsl/gsl>
#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"
#include <QnnOpDef.h>

namespace onnxruntime {
namespace qnn {

namespace {

Status GetInputNodeUnits(const GraphViewer& graph_viewer,
                         const NodeUnit& node_unit,
                         const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                         const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map,
                         /*out*/ std::map<size_t, const NodeUnit*>& input_node_units) {
  const Node& node = node_unit.GetNode();

  // input must be of a valid type.
  for (auto input_edge_iter = node.InputEdgesBegin(); input_edge_iter != node.InputEdgesEnd(); ++input_edge_iter) {
    auto& input_node = (*input_edge_iter).GetNode();
    if (graph_viewer.GetNode(input_node.Index()) == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input node not exists in graph.");
    }

    const auto input_node_unit_it = node_unit_map.find(&input_node);
    if (input_node_unit_it == node_unit_map.end()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input node has no NodeUnit mapping.");
    }
    const NodeUnit* input_node_unit = input_node_unit_it->second;

    // Check if input quant node has already been handled. Should not be the case if the calling
    // fusion function has been called in topological order, but check to be safe.
    if (input_node_unit->OpType() == DEQUANTIZE_LINEAR && qnn_node_group_map.count(input_node_unit) != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input quant node has been added");
    }

    if (input_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input node is not in single format.");
    }
    input_node_units[(*input_edge_iter).GetDstArgIndex()] = input_node_unit;
  }
  return Status::OK();
}

Status GetOutputNodeUnits(const GraphViewer& graph_viewer,
                          const NodeUnit& node_unit,
                          const std::unordered_map<const Node*, const NodeUnit*>& node_unit_map,
                          /*out*/ std::map<size_t, const NodeUnit*>& output_node_units) {
  const Node& node = node_unit.GetNode();

  // Child must be of a valid type.
  for (auto output_edge_iter = node.OutputEdgesBegin(); output_edge_iter != node.OutputEdgesEnd(); ++output_edge_iter) {
    auto& output_node = (*output_edge_iter).GetNode();
    if (graph_viewer.GetNode(output_node.Index()) == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Output node not exists in graph.");
    }

    const auto output_node_unit_it = node_unit_map.find(&output_node);
    if (output_node_unit_it == node_unit_map.end()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Output node has no NodeUnit mapping.");
    }
    const NodeUnit* output_node_unit = output_node_unit_it->second;

    if (output_node_unit->UnitType() != NodeUnit::Type::SingleNode) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Output node is not in single format.");
    }
    output_node_units[(*output_edge_iter).GetSrcArgIndex()] = output_node_unit;
  }
  return Status::OK();
}

static Status CreateOrValidateOnQnn(
    const std::string& op_type,
    const std::string& op_package,
    QnnModelWrapper& qnn_model_wrapper,
    const std::map<size_t, const NodeUnit*>& input_node_units,
    const NodeUnit& node_unit,
    const std::map<size_t, const NodeUnit*>& output_node_units,
    bool do_op_validation,
    const logging::Logger& logger) {
  ORT_UNUSED_PARAMETER(do_op_validation);

  const std::string node_name = utils::GetUniqueName(node_unit);

  // get qnn inputs
  const auto& inputs = node_unit.Inputs();
  //   std::vector<QnnTensorWrapper> input_tensor_wrappers;
  std::vector<std::string> input_names;
  for (size_t i = 0; i < inputs.size(); i++) {
    const NodeUnit* input_edge_src_node_unit = (input_node_units.find(i) != input_node_units.end()) ? input_node_units.at(i) : nullptr;
    if (!inputs[i].node_arg.Exists()) {
      continue;
    }

    // since input could come from initialize or graph input, which are not NodeUnit,
    // we have to compare the name to get the correct order
    std::string input_name = inputs[i].node_arg.Name();
    const NodeUnitIODef* input_def = &inputs[i];
    if (input_edge_src_node_unit && input_edge_src_node_unit->OpType() == DEQUANTIZE_LINEAR) {
      input_name = input_edge_src_node_unit->Inputs()[0].node_arg.Name();
      input_def = &(input_edge_src_node_unit->Inputs()[0]);
    }

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
      TensorInfo tensor_info;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(*input_def, tensor_info));

      QnnTensorWrapper tensor_wrapper;
      //   input_tensor_wrappers.emplace_back(QnnTensorWrapper());
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(tensor_info, input_name, tensor_wrapper));
      //   input_tensor_wrappers.emplace_back(tensor_wrapper);
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tensor_wrapper)),
                        "Failed to add tensor: " + input_name);
    } else {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_name;
    }

    input_names.emplace_back(input_name);
  }

  // get qnn outputs
  const auto& outputs = node_unit.Outputs();
  //   std::vector<QnnTensorWrapper> output_tensor_wrappers;
  std::vector<std::string> output_names;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const NodeUnit* output_edge_dst_node_unit = (output_node_units.find(i) != output_node_units.end()) ? output_node_units.at(i) : nullptr;
    if (!outputs[i].node_arg.Exists()) {
      continue;
    }
    std::string output_name = outputs[i].node_arg.Name();
    const NodeUnitIODef* output_def = &outputs[i];
    if (output_edge_dst_node_unit && output_edge_dst_node_unit->OpType() == QUANTIZE_LINEAR) {
      output_name = output_edge_dst_node_unit->Outputs()[0].node_arg.Name();
      output_def = &(output_edge_dst_node_unit->Outputs()[0]);
    }

    TensorInfo output_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(*output_def, output_info));
    bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

    Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper output_tensor_wrapper(output_name,
                                           tensor_type,
                                           output_info.qnn_data_type,
                                           std::move(output_info.quant_param),
                                           std::move(output_info.shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_wrapper)), "Failed to add tensor.");
    output_names.emplace_back(output_name);
  }

  // get qnn params
  NodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_names;
  for (auto& attr : node_unit.GetNode().GetAttributes()) {
    std::string attr_name = attr.first;
    auto& attr_value = attr.second;
    LOGS(logger, VERBOSE) << "Parse attribute name: " << attr_name << " for op " << node_name;
    switch (attr_value.type()) {
      case ONNX_NAMESPACE::AttributeProto::FLOAT: {
        auto optional_float = node_helper.GetFloat(attr_name);
        ORT_RETURN_IF_NOT(optional_float.has_value(),
                          "Failed to get values from attr ", attr_name, " in op ", node_name, " to qnn_model_wrapper.");
        ORT_RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_name, optional_float.value(), attr_name, param_names));
        break;
      }
      case ONNX_NAMESPACE::AttributeProto::FLOATS: {
        auto optional_floats = node_helper.GetFloats(attr_name);
        ORT_RETURN_IF_NOT(optional_floats.has_value(),
                          "Failed to get values from attr ", attr_name, " in op ", node_name, " to qnn_model_wrapper.");
        std::vector<float> floats_data(optional_floats.value().begin(), optional_floats.value().end());
        auto param_wrapper = createQnnParamWrapper<float>(node_unit.Index(), node_name, attr_name,
                                                          {static_cast<uint32_t>(floats_data.size())}, std::move(floats_data));
        param_names.push_back(param_wrapper.GetParamTensorName());
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(param_wrapper)),
                          "Failed to add tensor attr ", attr_name, " in op ", node_name, " to qnn_model_wrapper.");
        break;
      }
      case ONNX_NAMESPACE::AttributeProto::INT: {
        auto optional_int64 = node_helper.GetInt64(attr_name);
        ORT_RETURN_IF_NOT(optional_int64.has_value(),
                          "Failed to get values from attr ", attr_name, " in op ", node_name, " to qnn_model_wrapper.");
        ORT_RETURN_IF_ERROR(AddQnnScalar<int32_t>(qnn_model_wrapper, node_unit.Index(), node_name, SafeInt<int32_t>(optional_int64.value()), attr_name, param_names));
        break;
      }
      case ONNX_NAMESPACE::AttributeProto::INTS: {
        auto optional_int64s = node_helper.GetInt64s(attr_name);
        ORT_RETURN_IF_NOT(optional_int64s.has_value(),
                          "Failed to get values from attr ", attr_name, " in op ", node_name, " to qnn_model_wrapper.");
        std::vector<int32_t> int32s_data(optional_int64s.value().size(), 0);
        for (size_t i = 0; i < optional_int64s.value().size(); i++) {
          int32s_data[i] = SafeInt<int32_t>(optional_int64s.value()[i]);
        }
        auto param_wrapper = createQnnParamWrapper<int32_t>(node_unit.Index(), node_name, attr_name,
                                                            {static_cast<uint32_t>(int32s_data.size())}, std::move(int32s_data));
        param_names.push_back(param_wrapper.GetParamTensorName());
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(param_wrapper)),
                          "Failed to add tensor attr ", attr_name, " in op ", node_name, " to qnn_model_wrapper.");
        break;
      }
      case ONNX_NAMESPACE::AttributeProto::STRING: {
        auto optional_string = node_helper.GetString(attr_name);
        ORT_RETURN_IF_NOT(optional_string.has_value(),
                          "Failed to get values from attr ", attr_name, " in op ", node_name, " to qnn_model_wrapper.");
        ORT_RETURN_IF_ERROR(AddQnnScalar(qnn_model_wrapper, node_unit.Index(), node_name, optional_string.value(), attr_name, param_names));
        break;
      }
      default: {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to add scalar attr ", attr_name, " data_type ", attr_value.type(), " in op ", node_name, " to qnn_model_wrapper.");
      }
    }
  }
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                    op_package,
                                                    op_type,
                                                    std::move(input_names),
                                                    std::move(output_names),
                                                    std::move(param_names),
                                                    do_op_validation),
                    "Failed to add node.");

  return Status::OK();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> UDOQDQFusion::TryFusion(
    const std::string& op_type,
    const std::string& op_package,
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& udo_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();

  // find all input DequantizeLinear nodes
  std::map<size_t, const NodeUnit*> input_node_units;
  Status status = GetInputNodeUnits(graph_viewer, udo_node_unit, node_to_node_unit, node_unit_to_qnn_node_group, input_node_units);
  if (!status.IsOK()) {
    return nullptr;
  }

  // find all output QuantizeLinear nodes
  std::map<size_t, const NodeUnit*> output_node_units;
  status = GetOutputNodeUnits(graph_viewer, udo_node_unit, node_to_node_unit, output_node_units);
  if (!status.IsOK()) {
    return nullptr;
  }

  // Convert UDO node
  status = CreateOrValidateOnQnn(op_type, op_package, qnn_model_wrapper, input_node_units, udo_node_unit, output_node_units, true, logger);
  if (!status.IsOK()) {
    return nullptr;
  }

  return std::make_unique<UDOQDQFusion>(op_type, op_package, input_node_units, udo_node_unit, output_node_units);
}

UDOQDQFusion::UDOQDQFusion(
    const std::string& op_type,
    const std::string& op_package, const std::map<size_t, const NodeUnit*>& input_node_units, const NodeUnit& node_unit, const std::map<size_t, const NodeUnit*>& output_node_units)
    : op_type_(op_type),
      op_package_(op_package),
      input_node_units_(input_node_units),
      node_unit_(&node_unit),
      output_node_units_(output_node_units) {
  // only return input dq nodes/ node unit / output q nodes since they are the same group
  for (auto& input_node_unit : input_node_units_) {
    if (input_node_unit.second->OpType() == DEQUANTIZE_LINEAR) {
      all_nodes_.push_back(input_node_unit.second);
    }
  }
  all_nodes_.push_back(node_unit_);
  for (auto& output_node_unit : output_node_units_) {
    if (output_node_unit.second->OpType() == QUANTIZE_LINEAR) {
      all_nodes_.push_back(output_node_unit.second);
    }
  }
}
Status UDOQDQFusion::IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  return CreateOrValidateOnQnn(op_type_, op_package_, qmw, input_node_units_, *node_unit_, output_node_units_, true, logger);
}

Status UDOQDQFusion::AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  return CreateOrValidateOnQnn(op_type_, op_package_, qmw, input_node_units_, *node_unit_, output_node_units_, false, logger);
}

gsl::span<const NodeUnit* const> UDOQDQFusion::GetNodeUnits() const {
  auto res = gsl::make_span<const NodeUnit* const>(all_nodes_.data(), all_nodes_.size());
  return res;
}

const NodeUnit* UDOQDQFusion::GetTargetNodeUnit() const {
  return node_unit_;
}

}  // namespace qnn
}  // namespace onnxruntime
