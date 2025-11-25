// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/qnn_node_group/gelu_fusion.h"

#include <gsl/gsl>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include "core/providers/qnn-abi/ort_api.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"
#include "core/providers/qnn-abi/builder/op_builder_factory.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_node_group/utils.h"

namespace onnxruntime {
namespace qnn {

// Forward declarations.
#define ValidateOnQnn(qnn_model_wrapper, node_units, root_input, final_output) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (node_units), (root_input), (final_output), true)
#define CreateOnQnn(qnn_model_wrapper, node_units, root_input, final_output) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (node_units), (root_input), (final_output), false)

static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                         gsl::span<const OrtNodeUnit* const> node_units,
                                         const OrtNodeUnitIODef& root_input,
                                         const OrtNodeUnitIODef& final_output,
                                         bool validate);

std::unique_ptr<IQnnNodeGroup> GeluFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const OrtNodeUnit& erf_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);
  // Looking for an Erf node (can be SingleNode or QDQGroup).
  if (erf_node_unit.OpType() != "Erf") {
    return nullptr;
  }

  // Erf must have a Div parent on its input
  const auto& erf_inputs = erf_node_unit.Inputs();
  if (erf_inputs.empty()) {
    return nullptr;
  }

  const OrtNodeUnit* div_node_unit = GetParentOfInput(qnn_model_wrapper, erf_node_unit, erf_inputs[0],
                                                      node_to_node_unit, node_unit_to_qnn_node_group);
  if (div_node_unit == nullptr || div_node_unit->OpType() != "Div") {
    return nullptr;
  }

  // Div must have 2 inputs
  const auto& div_inputs = div_node_unit->Inputs();
  if (div_inputs.size() < 2) {
    return nullptr;
  }

  // Erf must have an Add child consuming its output
  const auto& erf_outputs = erf_node_unit.Outputs();
  if (erf_outputs.empty()) {
    return nullptr;
  }

  const OrtNodeUnit* add_node_unit = GetChildOfOutput(qnn_model_wrapper, erf_node_unit, erf_outputs[0],
                                                      node_to_node_unit, node_unit_to_qnn_node_group);
  if (add_node_unit == nullptr || add_node_unit->OpType() != "Add") {
    return nullptr;
  }

  // Add must have 2 inputs
  const auto& add_inputs = add_node_unit->Inputs();
  if (add_inputs.size() < 2) {
    return nullptr;
  }

  // Add must have a Mul child consuming its output
  const auto& add_outputs = add_node_unit->Outputs();
  if (add_outputs.empty()) {
    return nullptr;
  }

  const OrtNodeUnit* mul_node_unit = GetChildOfOutput(qnn_model_wrapper, *add_node_unit, add_outputs[0],
                                                      node_to_node_unit, node_unit_to_qnn_node_group);
  if (mul_node_unit == nullptr || mul_node_unit->OpType() != "Mul") {
    return nullptr;
  }

  // Now check which pattern we have
  const auto& root_input_name = div_inputs[0].name;
  const auto& mul_inputs = mul_node_unit->Inputs();

  if (mul_inputs.size() < 2) {
    return nullptr;
  }

  // Try to match Pattern 1: root -> Mul -> ... -> Mul
  // In this case, one input to the final Mul should be from a Mul node
  const OrtNodeUnit* mul2_node_unit = nullptr;

  // Check if either input to mul_node_unit comes from a Mul node
  for (size_t i = 0; i < 2; ++i) {
    const auto& mul_input_name = mul_inputs[i].name;

    // Find the node that produces this input by iterating through all nodes in QNN-ABI
    const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();

    for (const auto& [node, node_unit] : node_to_node_unit) {
      if (node == nullptr) continue;

      // Get outputs of this node and check if any matches our input
      size_t num_outputs = 0;
      RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(node, &num_outputs), ort_api, nullptr);

      std::vector<const OrtValueInfo*> outputs(num_outputs);
      RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(node, outputs.data(), outputs.size()), ort_api, nullptr);

      // Check if this node's output matches our input
      for (const auto* output_info : outputs) {
        const char* output_name = nullptr;
        RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoName(output_info, &output_name), ort_api, nullptr);

        if (output_name && output_name == mul_input_name) {
          // Found the producer node, check if it's a Mul
          const OrtNodeUnit* producer_unit = node_unit;
          if (producer_unit->OpType() == "Mul" &&
              node_unit_to_qnn_node_group.find(producer_unit) == node_unit_to_qnn_node_group.end()) {
            // Check if this Mul has root as one input (no longer checking for constant 0.5)
            const auto& mul2_inputs = producer_unit->Inputs();
            if (mul2_inputs.size() >= 2) {
              bool has_root_input = (mul2_inputs[0].name == root_input_name ||
                                     mul2_inputs[1].name == root_input_name);

              if (has_root_input) {
                mul2_node_unit = producer_unit;
                break;
              }
            }
          }
        }
      }
      if (mul2_node_unit != nullptr) break;
    }
    if (mul2_node_unit != nullptr) break;
  }

  std::vector<const OrtNodeUnit*> node_units;
  const OrtNodeUnit* final_mul_node_unit = nullptr;

  if (mul2_node_unit != nullptr) {
    // Pattern 1: root -> Mul -> ... -> Mul
    node_units = {div_node_unit, &erf_node_unit, add_node_unit, mul2_node_unit, mul_node_unit};
    final_mul_node_unit = mul_node_unit;
  } else {
    // Try Pattern 2: root -> ... -> Mul -> Mul
    // Check if one input to mul_node_unit is root
    bool has_root_input = (mul_inputs[0].name == root_input_name ||
                           mul_inputs[1].name == root_input_name);

    if (!has_root_input) {
      return nullptr;
    }

    // mul_node_unit must have a Mul child consuming its output
    const auto& mul_outputs = mul_node_unit->Outputs();
    if (mul_outputs.empty()) {
      return nullptr;
    }

    const OrtNodeUnit* mul2_node_unit_pattern2 = GetChildOfOutput(qnn_model_wrapper, *mul_node_unit, mul_outputs[0],
                                                                  node_to_node_unit, node_unit_to_qnn_node_group);
    if (mul2_node_unit_pattern2 == nullptr || mul2_node_unit_pattern2->OpType() != "Mul") {
      return nullptr;
    }

    // Verify this final Mul has 2 inputs
    const auto& mul2_inputs = mul2_node_unit_pattern2->Inputs();
    if (mul2_inputs.size() < 2) {
      return nullptr;
    }

    // Pattern 2
    node_units = {div_node_unit, &erf_node_unit, add_node_unit, mul_node_unit, mul2_node_unit_pattern2};
    final_mul_node_unit = mul2_node_unit_pattern2;
  }

  // Validate on QNN
  const OrtNodeUnitIODef& root_input = div_inputs[0];
  const OrtNodeUnitIODef& final_output = final_mul_node_unit->Outputs()[0];

  if (auto status = ValidateOnQnn(qnn_model_wrapper, node_units, root_input, final_output);
      !status.IsOK()) {
    return nullptr;
  }

  return std::make_unique<GeluFusion>(std::move(node_units), &erf_node_unit);
}

GeluFusion::GeluFusion(std::vector<const OrtNodeUnit*>&& node_units, const OrtNodeUnit* target_node_unit)
    : node_units_(std::move(node_units)), target_node_unit_(target_node_unit) {
}

Ort::Status GeluFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  const OrtNodeUnitIODef& root_input = node_units_[0]->Inputs()[0];
  const OrtNodeUnitIODef& final_output = node_units_.back()->Outputs()[0];
  return ValidateOnQnn(qmw, node_units_, root_input, final_output);
}

Ort::Status GeluFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  const OrtNodeUnitIODef& root_input = node_units_[0]->Inputs()[0];
  const OrtNodeUnitIODef& final_output = node_units_.back()->Outputs()[0];
  return CreateOnQnn(qmw, node_units_, root_input, final_output);
}

gsl::span<const OrtNodeUnit* const> GeluFusion::GetNodeUnits() const {
  return gsl::span<const OrtNodeUnit* const>(node_units_.data(), node_units_.size());
}

const OrtNodeUnit* GeluFusion::GetTargetNodeUnit() const {
  return target_node_unit_;
}

static Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                         gsl::span<const OrtNodeUnit* const> node_units,
                                         const OrtNodeUnitIODef& root_input,
                                         const OrtNodeUnitIODef& final_output,
                                         bool validate) {
  assert(node_units.size() >= 4);
  const auto& node_name = utils::GetUniqueName(*node_units[0]);

  QnnTensorWrapper input_tensor;
  QnnTensorWrapper output_tensor;

  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(root_input, input_tensor));
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(final_output, output_tensor));

  if (validate) {
    RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(node_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_GELU,
                                                      {input_tensor.GetQnnTensor()},
                                                      {output_tensor.GetQnnTensor()},
                                                      {}));
  } else {
    // Only add tensor wrappers if they don't already exist
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(root_input.name)) {
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
    }
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(final_output.name)) {
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");
    }
    RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(node_name,
                                                  QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                  QNN_OP_GELU,
                                                  {root_input.name},
                                                  {final_output.name},
                                                  {},
                                                  validate),
                  "Failed to add fused Gelu node.");
  }

  return Ort::Status();
}

}  // namespace qnn
}  // namespace onnxruntime
