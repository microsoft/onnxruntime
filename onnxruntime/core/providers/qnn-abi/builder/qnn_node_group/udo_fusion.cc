// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/qnn_node_group/udo_fusion.h"

#include <algorithm>
#include <cassert>
#include <gsl/gsl>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <QnnOpDef.h>

#include "core/providers/qnn-abi/builder/op_builder_factory.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_node_group/utils.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"
#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {

namespace {

Ort::Status GetInputNodeUnits(const QnnModelWrapper& qnn_model_wrapper,
                              const OrtNodeUnit& node_unit,
                              const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                              const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& qnn_node_group_map,
                              /*out*/ std::map<size_t, const OrtNodeUnit*>& input_node_units) {
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();
  const OrtNode& node = node_unit.GetNode();

  // input must be of a valid type.
  // Get node inputs
  size_t num_inputs = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetNumInputs(&node, &num_inputs));
  std::vector<const OrtValueInfo*> inputs(num_inputs);
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetInputs(&node, inputs.data(), inputs.size()));

  for (size_t i = 0; i < num_inputs; ++i) {
    const OrtValueInfo* input_value_info = inputs[i];

    // Get the producer of this input
    const OrtNode* input_node = nullptr;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.ValueInfo_GetValueProducer(input_value_info, &input_node, nullptr));
    if (input_node != nullptr) {
      const auto input_node_unit_it = node_unit_map.find(input_node);
      RETURN_IF(input_node_unit_it == node_unit_map.end(), "Input node has no OrtNodeUnit mapping.");
      const OrtNodeUnit* input_node_unit = input_node_unit_it->second;

      // Check if input quant node has already been handled. Should not be the case if the calling
      // fusion function has been called in topological order, but check to be safe.
      RETURN_IF(input_node_unit->OpType() == DEQUANTIZE_LINEAR && qnn_node_group_map.count(input_node_unit) != 0,
                "Input quant node has been added");

      RETURN_IF(input_node_unit->UnitType() != OrtNodeUnit::Type::SingleNode, "Input node is not in single format.");
      input_node_units[i] = input_node_unit;
    }
  }
  return Ort::Status();
}

Ort::Status GetOutputNodeUnits(const QnnModelWrapper& qnn_model_wrapper,
                               const OrtNodeUnit& node_unit,
                               const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_unit_map,
                               /*out*/ std::map<size_t, const OrtNodeUnit*>& output_node_units) {
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();
  const OrtNode& node = node_unit.GetNode();

  // Child must be of a valid type.
  // Get node outputs
  size_t num_outputs = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetNumOutputs(&node, &num_outputs));
  std::vector<const OrtValueInfo*> outputs(num_outputs);
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetOutputs(&node, outputs.data(), outputs.size()));

  for (size_t i = 0; i < num_outputs; ++i) {
    const OrtValueInfo* output_value_info = outputs[i];

    // Get the consumer of this output
    size_t num_consumers;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.ValueInfo_GetValueNumConsumers(output_value_info, &num_consumers));
    std::vector<const OrtNode*> output_nodes(num_consumers);
    std::vector<int64_t> output_consumer_indices(num_consumers);
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.ValueInfo_GetValueConsumers(output_value_info,
                                                                   output_nodes.data(),
                                                                   output_consumer_indices.data(),
                                                                   num_consumers));
    if (!output_nodes.empty() && output_nodes[0] != nullptr) {
      const OrtNode* output_node = output_nodes[0];
      const auto output_node_unit_it = node_unit_map.find(output_node);
      RETURN_IF(output_node_unit_it == node_unit_map.end(), "Output node has no OrtNodeUnit mapping.");
      const OrtNodeUnit* output_node_unit = output_node_unit_it->second;

      RETURN_IF(output_node_unit->UnitType() != OrtNodeUnit::Type::SingleNode, "Output node is not in single format.");
      output_node_units[i] = output_node_unit;
    }
  }
  return Ort::Status();
}

static Ort::Status CreateOrValidateOnQnn(
    const std::string& op_type,
    const std::string& op_package,
    QnnModelWrapper& qnn_model_wrapper,
    const std::map<size_t, const OrtNodeUnit*>& input_node_units,
    const OrtNodeUnit& node_unit,
    const std::map<size_t, const OrtNodeUnit*>& output_node_units,
    bool do_op_validation,
    const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(do_op_validation);

  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();
  const std::string node_name = utils::GetUniqueName(node_unit);

  // get qnn inputs
  const auto& inputs = node_unit.Inputs();
  std::vector<std::string> input_names;
  for (size_t i = 0; i < inputs.size(); i++) {
    const OrtNodeUnit* input_edge_src_node_unit = (input_node_units.find(i) != input_node_units.end()) ? input_node_units.at(i) : nullptr;
    if (!inputs[i].Exists()) {
      continue;
    }

    // since input could come from initialize or graph input, which are not OrtNodeUnit,
    // we have to compare the name to get the correct order
    std::string input_name = inputs[i].name;
    const OrtNodeUnitIODef* input_def = &inputs[i];
    if (input_edge_src_node_unit && input_edge_src_node_unit->OpType() == DEQUANTIZE_LINEAR) {
      input_name = input_edge_src_node_unit->Inputs()[0].name;
      input_def = &(input_edge_src_node_unit->Inputs()[0]);
    }

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
      TensorInfo tensor_info;
      RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(*input_def, tensor_info));

      QnnTensorWrapper tensor_wrapper;
      RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(tensor_info, input_name, tensor_wrapper));
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(tensor_wrapper)),
                    ("Failed to add tensor: " + input_name).c_str());
    } else {
      ORT_CXX_LOG(logger, ORT_LOGGING_LEVEL_VERBOSE, ("Tensor already added, skip it: " + input_name).c_str());
    }

    input_names.emplace_back(input_name);
  }

  // get qnn outputs
  const auto& outputs = node_unit.Outputs();
  std::vector<std::string> output_names;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const OrtNodeUnit* output_edge_dst_node_unit = (output_node_units.find(i) != output_node_units.end()) ? output_node_units.at(i) : nullptr;
    if (!outputs[i].Exists()) {
      continue;
    }
    std::string output_name = outputs[i].name;
    const OrtNodeUnitIODef* output_def = &outputs[i];
    if (output_edge_dst_node_unit && output_edge_dst_node_unit->OpType() == QUANTIZE_LINEAR) {
      output_name = output_edge_dst_node_unit->Outputs()[0].name;
      output_def = &(output_edge_dst_node_unit->Outputs()[0]);
    }

    TensorInfo output_info = {};
    RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(*output_def, output_info));
    bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

    Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    QnnTensorWrapper output_tensor_wrapper(output_name,
                                           tensor_type,
                                           output_info.qnn_data_type,
                                           std::move(output_info.quant_param),
                                           std::move(output_info.shape));
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor_wrapper)), "Failed to add tensor.");
    output_names.emplace_back(output_name);
  }

  // get qnn params
  size_t num_attributes = 0;
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetNumAttributes(&(node_unit.GetNode()), &num_attributes));
  std::vector<const OrtOpAttr*> attributes(num_attributes);
  ORT_CXX_RETURN_ON_API_FAIL(ort_api.Node_GetAttributes(&(node_unit.GetNode()), attributes.data(), attributes.size()));

  OrtNodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_names;
  for (const OrtOpAttr* attr : attributes) {
    OrtOpAttrType attr_type = ORT_OP_ATTR_UNDEFINED;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.OpAttr_GetType(attr, &attr_type));
    const char* attribute_name;
    ORT_CXX_RETURN_ON_API_FAIL(ort_api.OpAttr_GetName(attr, &attribute_name));
    std::string attr_name = std::string(attribute_name);

    ORT_CXX_LOG(logger,
                ORT_LOGGING_LEVEL_VERBOSE,
                ("Parse attribute name: " + attr_name + " for op " + node_name).c_str());
    switch (attr_type) {
      case ORT_OP_ATTR_FLOAT: {
        auto optional_float = node_helper.GetFloat(attr_name);
        RETURN_IF_NOT(optional_float.has_value(),
                      ("Failed to get values from attr " + attr_name +
                       " in op " + node_name + " to qnn_model_wrapper.")
                          .c_str());
        RETURN_IF_ERROR(AddQnnScalar<float>(qnn_model_wrapper, node_unit.Index(), node_name, optional_float.value(),
                                            attr_name, param_names));
        break;
      }
      case ORT_OP_ATTR_FLOATS: {
        auto optional_floats = node_helper.GetFloats(attr_name);
        RETURN_IF_NOT(optional_floats.has_value(),
                      ("Failed to get values from attr " + attr_name +
                       " in op " + node_name + " to qnn_model_wrapper.")
                          .c_str());
        std::vector<float> floats_data(optional_floats.value().begin(), optional_floats.value().end());
        auto param_wrapper = createQnnParamWrapper<float>(node_unit.Index(), node_name, attr_name,
                                                          {static_cast<uint32_t>(floats_data.size())}, std::move(floats_data));
        param_names.push_back(param_wrapper.GetParamTensorName());
        RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(param_wrapper)),
                      ("Failed to add tensor attr " + attr_name +
                       " in op " + node_name + " to qnn_model_wrapper.")
                          .c_str());
        break;
      }
      case ORT_OP_ATTR_INT: {
        auto optional_int64 = node_helper.GetInt64(attr_name);
        RETURN_IF_NOT(optional_int64.has_value(),
                      ("Failed to get values from attr " + attr_name +
                       " in op " + node_name + " to qnn_model_wrapper.")
                          .c_str());
        RETURN_IF_ERROR(AddQnnScalar<int32_t>(qnn_model_wrapper,
                                              node_unit.Index(),
                                              node_name,
                                              SafeInt<int32_t>(optional_int64.value()),
                                              attr_name,
                                              param_names));
        break;
      }
      case ORT_OP_ATTR_INTS: {
        auto optional_int64s = node_helper.GetInt64s(attr_name);
        RETURN_IF_NOT(optional_int64s.has_value(),
                      ("Failed to get values from attr " + attr_name +
                       " in op " + node_name + " to qnn_model_wrapper.")
                          .c_str());
        std::vector<int32_t> int32s_data(optional_int64s.value().size(), 0);
        for (size_t i = 0; i < optional_int64s.value().size(); i++) {
          int32s_data[i] = SafeInt<int32_t>(optional_int64s.value()[i]);
        }
        auto param_wrapper = createQnnParamWrapper<int32_t>(node_unit.Index(), node_name, attr_name,
                                                            {static_cast<uint32_t>(int32s_data.size())}, std::move(int32s_data));
        param_names.push_back(param_wrapper.GetParamTensorName());
        RETURN_IF_NOT(qnn_model_wrapper.AddParamWrapper(std::move(param_wrapper)),
                      ("Failed to add tensor attr " + attr_name +
                       " in op " + node_name + " to qnn_model_wrapper.")
                          .c_str());
        break;
      }
      case ORT_OP_ATTR_STRING: {
        auto optional_string = node_helper.GetString(attr_name);
        RETURN_IF_NOT(optional_string.has_value(),
                      ("Failed to get values from attr " + attr_name +
                       " in op " + node_name + " to qnn_model_wrapper.")
                          .c_str());
        RETURN_IF_ERROR(AddQnnScalar(qnn_model_wrapper, node_unit.Index(), node_name, optional_string.value(),
                                     attr_name, param_names));
        break;
      }
      default: {
        return MAKE_EP_FAIL(("Failed to add scalar attr " + attr_name + " data_type " + std::to_string(attr_type) +
                             " in op " + node_name + " to qnn_model_wrapper.")
                                .c_str());
      }
    }
  }
  RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                op_package,
                                                op_type,
                                                std::move(input_names),
                                                std::move(output_names),
                                                std::move(param_names),
                                                do_op_validation),
                "Failed to add node.");

  return Ort::Status();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> UDOQDQFusion::TryFusion(
    const std::string& op_type,
    const std::string& op_package,
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& udo_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  // find all input DequantizeLinear nodes
  std::map<size_t, const OrtNodeUnit*> input_node_units;
  Ort::Status status = GetInputNodeUnits(qnn_model_wrapper, udo_node_unit, node_to_node_unit, node_unit_to_qnn_node_group, input_node_units);
  if (!status.IsOK()) {
    return nullptr;
  }

  // find all output QuantizeLinear nodes
  std::map<size_t, const OrtNodeUnit*> output_node_units;
  status = GetOutputNodeUnits(qnn_model_wrapper, udo_node_unit, node_to_node_unit, output_node_units);
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
    const std::string& op_package,
    const std::map<size_t, const OrtNodeUnit*>& input_node_units,
    const OrtNodeUnit& node_unit,
    const std::map<size_t, const OrtNodeUnit*>& output_node_units)
    : op_type_(op_type),
      op_package_(op_package),
      input_node_units_(input_node_units),
      node_unit_(&node_unit),
      output_node_units_(output_node_units) {
  // only return input dq nodes / node unit / output q nodes since they are the same group
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
Ort::Status UDOQDQFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(op_type_, op_package_, qmw, input_node_units_, *node_unit_, output_node_units_, true, logger);
}

Ort::Status UDOQDQFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  return CreateOrValidateOnQnn(op_type_, op_package_, qmw, input_node_units_, *node_unit_, output_node_units_, false, logger);
}

gsl::span<const OrtNodeUnit* const> UDOQDQFusion::GetNodeUnits() const {
  auto res = gsl::make_span<const OrtNodeUnit* const>(all_nodes_.data(), all_nodes_.size());
  return res;
}

const OrtNodeUnit* UDOQDQFusion::GetTargetNodeUnit() const {
  return node_unit_;
}

}  // namespace qnn
}  // namespace onnxruntime
