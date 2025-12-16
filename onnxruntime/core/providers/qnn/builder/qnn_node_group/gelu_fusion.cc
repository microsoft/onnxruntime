// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_node_group/gelu_fusion.h"

#include <gsl/gsl>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include "core/providers/qnn/ort_api.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/qnn_node_group/utils.h"

namespace onnxruntime {
namespace qnn {

// Helper function to extract value from raw data based on QNN data type
static Status GetValueOnQnnDataType(const Qnn_DataType_t qnn_data_type,
                                    const uint8_t* raw_ptr,
                                    double& value) {
  switch (qnn_data_type) {
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_SFIXED_POINT_8: {
      value = static_cast<double>(*reinterpret_cast<const int8_t*>(raw_ptr));
      break;
    }
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_SFIXED_POINT_16: {
      value = static_cast<double>(*reinterpret_cast<const int16_t*>(raw_ptr));
      break;
    }
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_SFIXED_POINT_32: {
      value = static_cast<double>(*reinterpret_cast<const int32_t*>(raw_ptr));
      break;
    }
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8: {
      value = static_cast<double>(*reinterpret_cast<const uint8_t*>(raw_ptr));
      break;
    }
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16: {
      value = static_cast<double>(*reinterpret_cast<const uint16_t*>(raw_ptr));
      break;
    }
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32: {
      value = static_cast<double>(*reinterpret_cast<const uint32_t*>(raw_ptr));
      break;
    }
    case QNN_DATATYPE_FLOAT_32: {
      value = static_cast<double>(*reinterpret_cast<const float*>(raw_ptr));
      break;
    }
    case QNN_DATATYPE_FLOAT_16: {
      value = static_cast<double>(reinterpret_cast<const MLFloat16*>(raw_ptr)->ToFloat());
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Qnn Data Type: ", qnn_data_type, " not supported.");
  }
  return Status::OK();
}

// Helper function to extract a scalar float value from a constant initializer
// Handles both float and quantized (INT type) constant inputs
static std::optional<float> GetConstantInitializerFloatScalar(QnnModelWrapper& qnn_model_wrapper,
                                                              const NodeUnitIODef& io_def) {
  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
  const auto& name = io_def.node_arg.Name();

  if (!graph_viewer.IsConstantInitializer(name, true)) {
    return std::nullopt;
  }

  // Get tensor info to check if it's quantized
  TensorInfo tensor_info = {};
  if (!qnn_model_wrapper.GetTensorInfo(io_def, tensor_info).IsOK()) {
    return std::nullopt;
  }

  // Must be an initializer
  if (!tensor_info.is_initializer || !tensor_info.initializer_tensor) {
    return std::nullopt;
  }

  // Unpack the initializer data
  std::vector<uint8_t> unpacked_tensor;
  if (!qnn_model_wrapper.UnpackInitializerData(*tensor_info.initializer_tensor, unpacked_tensor).IsOK()) {
    return std::nullopt;
  }

  if (unpacked_tensor.empty()) {
    return std::nullopt;
  }

  // Extract the value using GetValueOnQnnDataType
  double extracted_value = 0.0;
  if (!GetValueOnQnnDataType(tensor_info.qnn_data_type, unpacked_tensor.data(), extracted_value).IsOK()) {
    return std::nullopt;
  }

  // Check if quantized and dequantize if needed
  const bool is_quantized = tensor_info.quant_param.IsQuantized();
  if (is_quantized) {
    // For quantized tensors, dequantize the value
    if (!tensor_info.quant_param.IsPerTensor()) {
      return std::nullopt;  // Only support per-tensor quantization
    }

    const Qnn_QuantizeParams_t& quant_param = tensor_info.quant_param.Get();
    double dequantized_value = utils::Dequantize(quant_param.scaleOffsetEncoding.offset,
                                                 quant_param.scaleOffsetEncoding.scale,
                                                 extracted_value);
    return static_cast<float>(dequantized_value);
  }

  // For non-quantized tensors, return the extracted value directly
  return static_cast<float>(extracted_value);
}

// Helper function to check if a constant initializer has the expected float value
static bool IsInitializerWithExpectedValue(QnnModelWrapper& qnn_model_wrapper,
                                           const NodeUnitIODef& io_def,
                                           float expected_value,
                                           float tolerance = 1e-5f) {
  std::optional<float> actual_value = GetConstantInitializerFloatScalar(qnn_model_wrapper, io_def);
  if (!actual_value.has_value()) {
    return false;
  }

  // Compare with expected value within tolerance
  return std::fabs(actual_value.value() - expected_value) <= tolerance;
}

// Forward declaration.
static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                    gsl::span<const NodeUnit* const> node_units,
                                    const NodeUnitIODef& root_input,
                                    const NodeUnitIODef& final_output,
                                    bool validate);

// Helper function to validate on QNN
static Status ValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                            gsl::span<const NodeUnit* const> node_units,
                            const NodeUnitIODef& root_input,
                            const NodeUnitIODef& final_output) {
  return CreateOrValidateOnQnn(qnn_model_wrapper, node_units, root_input, final_output, true);
}

// Helper function to create on QNN
static Status CreateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                          gsl::span<const NodeUnit* const> node_units,
                          const NodeUnitIODef& root_input,
                          const NodeUnitIODef& final_output) {
  return CreateOrValidateOnQnn(qnn_model_wrapper, node_units, root_input, final_output, false);
}

// Gets the parent and child of the Erf node. Can handle the following sequences
//  - Parent -> Erf -> Child.
//  - Parent -> DQ -> Erf -> Q -> Child.
//
// Also returns the outputs of the Erf. For the sequence `DQ -> Erf -> Q`, returns the outputs of the Q.
static bool GetErfParentAndChild(QnnModelWrapper& qnn_model_wrapper, const NodeUnit& erf_node_unit,
                                 const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
                                 const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
                                 /*out*/ const NodeUnit*& parent_node_unit,
                                 /*out*/ const NodeUnit*& child_node_unit,
                                 /*out*/ const NodeUnit*& dq_node_unit,
                                 /*out*/ const NodeUnit*& q_node_unit,
                                 /*out*/ gsl::span<const NodeUnitIODef>& erf_outputs) {
  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();

  auto get_first_parent = [&](const NodeUnit& node_unit) -> const NodeUnit* {
    const auto& inputs = node_unit.Inputs();
    if (inputs.empty()) {
      return nullptr;
    }
    return GetParentOfInput(graph_viewer, node_unit, inputs[0],
                            node_to_node_unit, node_unit_to_qnn_node_group);
  };

  auto get_first_child = [&](const NodeUnit& node_unit) -> const NodeUnit* {
    const auto& outputs = node_unit.Outputs();
    if (outputs.empty()) {
      return nullptr;
    }

    return GetOnlyChildOfOutput(graph_viewer, node_unit, outputs[0],
                                node_to_node_unit, node_unit_to_qnn_node_group);
  };

  const NodeUnit* erf_parent_node_unit = get_first_parent(erf_node_unit);
  if (erf_parent_node_unit == nullptr) {
    return false;
  }

  const NodeUnit* erf_child_node_unit = get_first_child(erf_node_unit);
  if (erf_child_node_unit == nullptr) {
    return false;
  }

  if (erf_node_unit.UnitType() == NodeUnit::Type::SingleNode &&
      erf_parent_node_unit->OpType() == "DequantizeLinear" &&
      erf_child_node_unit->OpType() == "QuantizeLinear") {
    // This is the explicit sequence DQ -> Erf -> Q.
    // Look past the DQ and Q nodes to get the actual parent and child.
    // We do this because ORT utils do not automatically group DQ -> Erf -> Q into a NodeUnit.
    dq_node_unit = erf_parent_node_unit;
    q_node_unit = erf_child_node_unit;
    erf_parent_node_unit = get_first_parent(*erf_parent_node_unit);
    erf_child_node_unit = get_first_child(*erf_child_node_unit);

    erf_outputs = q_node_unit->Outputs();
  } else {
    erf_outputs = erf_node_unit.Outputs();
  }

  parent_node_unit = erf_parent_node_unit;
  child_node_unit = erf_child_node_unit;
  return parent_node_unit != nullptr && child_node_unit != nullptr;
}

std::unique_ptr<IQnnNodeGroup> GeluFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& erf_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const logging::Logger& /*logger*/) {
  if (erf_node_unit.OpType() != "Erf") {
    return nullptr;
  }

  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();
  const NodeUnit* div_node_unit = nullptr;
  const NodeUnit* add_node_unit = nullptr;
  const NodeUnit* dq_node_unit = nullptr;
  const NodeUnit* q_node_unit = nullptr;
  gsl::span<const NodeUnitIODef> erf_outputs;

  if (!GetErfParentAndChild(qnn_model_wrapper, erf_node_unit, node_to_node_unit, node_unit_to_qnn_node_group,
                            div_node_unit, add_node_unit, dq_node_unit, q_node_unit, erf_outputs)) {
    return nullptr;
  }

  // Erf must have a Div parent.
  if (div_node_unit == nullptr || div_node_unit->OpType() != "Div") {
    return nullptr;
  }

  // Div must have 2 inputs
  const auto& div_inputs = div_node_unit->Inputs();
  if (div_inputs.size() < 2) {
    return nullptr;
  }

  // Check second input of Div is sqrt(2) ≈ 1.4142
  if (!IsInitializerWithExpectedValue(qnn_model_wrapper, div_inputs[1], static_cast<float>(M_SQRT2))) {
    return nullptr;
  }

  // Erf must have an Add child consuming its output
  if (add_node_unit == nullptr || add_node_unit->OpType() != "Add") {
    return nullptr;
  }

  // Add must have 2 inputs
  const auto& add_inputs = add_node_unit->Inputs();
  if (add_inputs.size() < 2) {
    return nullptr;
  }

  // Check the other input node (e.g. not the Erf) is 1.0f
  bool is_erf_first_input = (add_inputs[0].node_arg.Name() == erf_outputs[0].node_arg.Name());
  const auto& add_const_input = add_inputs[is_erf_first_input ? 1 : 0];
  if (!IsInitializerWithExpectedValue(qnn_model_wrapper, add_const_input, 1.0f)) {
    return nullptr;
  }

  // Add must have a Mul child consuming its output
  const auto& add_outputs = add_node_unit->Outputs();
  if (add_outputs.empty()) {
    return nullptr;
  }

  const NodeUnit* mul_node_unit = GetOnlyChildOfOutput(graph_viewer, *add_node_unit, add_outputs[0],
                                                       node_to_node_unit, node_unit_to_qnn_node_group);
  if (mul_node_unit == nullptr || mul_node_unit->OpType() != "Mul") {
    return nullptr;
  }

  // Now check which pattern we have
  const auto& root_input_name = div_inputs[0].node_arg.Name();
  const auto& mul_inputs = mul_node_unit->Inputs();

  if (mul_inputs.size() < 2) {
    return nullptr;
  }

  // Try to match Pattern 1: root -> Mul(0.5) -> ... -> Mul
  // In this case, one input to the final Mul should be from a Mul node
  const NodeUnit* mul2_node_unit = nullptr;

  // Check if either input to mul_node_unit comes from a Mul node
  for (size_t i = 0; i < 2; ++i) {
    const auto& mul_input = mul_inputs[i];

    const NodeUnit* producer_unit = GetParentOfInput(graph_viewer, *mul_node_unit, mul_input,
                                                     node_to_node_unit, node_unit_to_qnn_node_group);
    if (producer_unit && producer_unit->OpType() == "Mul") {
      const auto& mul2_inputs = producer_unit->Inputs();
      if (mul2_inputs.size() >= 2) {
        bool has_root_input = (mul2_inputs[0].node_arg.Name() == root_input_name ||
                               mul2_inputs[1].node_arg.Name() == root_input_name);
        if (has_root_input) {
          int root_index = (mul2_inputs[0].node_arg.Name() == root_input_name) ? 0 : 1;
          const auto& mul_const_input = mul2_inputs[1 - root_index];

          if (IsInitializerWithExpectedValue(qnn_model_wrapper, mul_const_input, 0.5f)) {
            mul2_node_unit = producer_unit;
            break;
          }
        }
      }
    }
    if (mul2_node_unit != nullptr) break;
  }

  std::vector<const NodeUnit*> node_units;
  const NodeUnit* final_mul_node_unit = nullptr;

  if (mul2_node_unit != nullptr) {
    // Pattern 1: root -> Mul(0.5) -> ... -> Mul
    if (dq_node_unit != nullptr) {
      assert(q_node_unit != nullptr);
      node_units = {div_node_unit, dq_node_unit, &erf_node_unit, q_node_unit, add_node_unit, mul2_node_unit,
                    mul_node_unit};
    } else {
      node_units = {div_node_unit, &erf_node_unit, add_node_unit, mul2_node_unit, mul_node_unit};
    }
    final_mul_node_unit = mul_node_unit;
  } else {
    // Try Pattern 2: root -> ... -> Mul -> Mul(0.5)
    // Check if one input to mul_node_unit is root
    bool has_root_input = (mul_inputs[0].node_arg.Name() == root_input_name ||
                           mul_inputs[1].node_arg.Name() == root_input_name);

    if (!has_root_input) {
      return nullptr;
    }

    // mul_node_unit must have a Mul child consuming its output
    const auto& mul_outputs = mul_node_unit->Outputs();
    if (mul_outputs.empty()) {
      return nullptr;
    }

    const NodeUnit* mul2_node_unit_pattern2 = GetOnlyChildOfOutput(graph_viewer, *mul_node_unit, mul_outputs[0],
                                                                   node_to_node_unit, node_unit_to_qnn_node_group);
    if (mul2_node_unit_pattern2 == nullptr || mul2_node_unit_pattern2->OpType() != "Mul") {
      return nullptr;
    }

    // Verify this final Mul has 2 inputs
    const auto& mul2_inputs = mul2_node_unit_pattern2->Inputs();
    if (mul2_inputs.size() < 2) {
      return nullptr;
    }

    // Check the constant input is 0.5f
    int mul_const_input_index = 0;
    if (mul2_inputs[0].node_arg.Name() == mul_outputs[0].node_arg.Name()) {
      mul_const_input_index = 1;
    }
    const auto& mul_const_input = mul2_inputs[mul_const_input_index];
    if (!IsInitializerWithExpectedValue(qnn_model_wrapper, mul_const_input, 0.5f)) {
      return nullptr;
    }

    // Pattern 2
    if (dq_node_unit != nullptr) {
      assert(q_node_unit != nullptr);
      node_units = {div_node_unit, dq_node_unit, &erf_node_unit, q_node_unit, add_node_unit,
                    mul_node_unit, mul2_node_unit_pattern2};
    } else {
      node_units = {div_node_unit, &erf_node_unit, add_node_unit, mul_node_unit, mul2_node_unit_pattern2};
    }

    final_mul_node_unit = mul2_node_unit_pattern2;
  }

  // Validate on QNN
  const NodeUnitIODef& root_input = div_inputs[0];
  const NodeUnitIODef& final_output = final_mul_node_unit->Outputs()[0];

  if (Status status = ValidateOnQnn(qnn_model_wrapper, node_units, root_input, final_output);
      !status.IsOK()) {
    return nullptr;
  }

  return std::make_unique<GeluFusion>(std::move(node_units), &erf_node_unit);
}

GeluFusion::GeluFusion(std::vector<const NodeUnit*>&& node_units, const NodeUnit* target_node_unit)
    : node_units_(std::move(node_units)), target_node_unit_(target_node_unit) {
}

Status GeluFusion::IsSupported(QnnModelWrapper& qmw, const logging::Logger& /*logger*/) const {
  ORT_RETURN_IF_NOT(!node_units_.empty(), "GeluFusion node_units_ is empty");
  const NodeUnitIODef& root_input = node_units_[0]->Inputs()[0];
  const NodeUnitIODef& final_output = node_units_.back()->Outputs()[0];
  return ValidateOnQnn(qmw, node_units_, root_input, final_output);
}

Status GeluFusion::AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& /*logger*/) const {
  ORT_RETURN_IF_NOT(!node_units_.empty(), "GeluFusion node_units_ is empty");
  const NodeUnitIODef& root_input = node_units_[0]->Inputs()[0];
  const NodeUnitIODef& final_output = node_units_.back()->Outputs()[0];
  return CreateOnQnn(qmw, node_units_, root_input, final_output);
}

gsl::span<const NodeUnit* const> GeluFusion::GetNodeUnits() const {
  return gsl::span<const NodeUnit* const>(node_units_.data(), node_units_.size());
}

const NodeUnit* GeluFusion::GetTargetNodeUnit() const {
  return target_node_unit_;
}

static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                    gsl::span<const NodeUnit* const> node_units,
                                    const NodeUnitIODef& root_input,
                                    const NodeUnitIODef& final_output,
                                    bool validate) {
  assert(node_units.size() >= 4);
  const auto& node_name = utils::GetUniqueName(*node_units[0]);

  QnnTensorWrapper input_tensor;
  QnnTensorWrapper output_tensor;

  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(root_input, input_tensor));
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(final_output, output_tensor));

  if (validate) {
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(node_name,
                                                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                          QNN_OP_GELU,
                                                          {input_tensor.GetQnnTensor()},
                                                          {output_tensor.GetQnnTensor()},
                                                          {}));
  } else {
    // Only add tensor wrappers if they don't already exist
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(root_input.node_arg.Name())) {
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
    }
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(final_output.node_arg.Name())) {
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");
    }
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_GELU,
                                                      {root_input.node_arg.Name()},
                                                      {final_output.node_arg.Name()},
                                                      {},
                                                      validate),
                      "Failed to add fused Gelu node.");
  }

  return Status::OK();
}

}  // namespace qnn
}  // namespace onnxruntime
