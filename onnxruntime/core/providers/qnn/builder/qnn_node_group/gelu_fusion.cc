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

// Forward declarations.
#define ValidateOnQnn(qnn_model_wrapper, node_units, root_input, final_output) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (node_units), (root_input), (final_output), true)
#define CreateOnQnn(qnn_model_wrapper, node_units, root_input, final_output) \
  CreateOrValidateOnQnn((qnn_model_wrapper), (node_units), (root_input), (final_output), false)

static Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper,
                                    gsl::span<const NodeUnit* const> node_units,
                                    const NodeUnitIODef& root_input,
                                    const NodeUnitIODef& final_output,
                                    bool validate);

std::unique_ptr<IQnnNodeGroup> GeluFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper,
    const NodeUnit& erf_node_unit,
    const std::unordered_map<const Node*, const NodeUnit*>& node_to_node_unit,
    const std::unordered_map<const NodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const logging::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);
  // Looking for an Erf node (can be SingleNode or QDQGroup).
  if (erf_node_unit.OpType() != "Erf") {
    return nullptr;
  }

  const GraphViewer& graph_viewer = qnn_model_wrapper.GetGraphViewer();

  // Erf must have a Div parent on its input
  const auto& erf_inputs = erf_node_unit.Inputs();
  if (erf_inputs.empty()) {
    return nullptr;
  }

  const NodeUnit* div_node_unit = GetParentOfInput(graph_viewer, erf_node_unit, erf_inputs[0],
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

  const NodeUnit* add_node_unit = GetChildOfOutput(graph_viewer, erf_node_unit, erf_outputs[0],
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

  const NodeUnit* mul_node_unit = GetChildOfOutput(graph_viewer, *add_node_unit, add_outputs[0],
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

  // Try to match Pattern 1: root -> Mul -> ... -> Mul
  // In this case, one input to the final Mul should be from a Mul node
  const NodeUnit* mul2_node_unit = nullptr;

  // Check if either input to mul_node_unit comes from a Mul node
  for (size_t i = 0; i < 2; ++i) {
    const auto& mul_input_name = mul_inputs[i].node_arg.Name();

    // Find the node that produces this input
    for (const auto& node_index : graph_viewer.GetNodesInTopologicalOrder()) {
      const Node* node = graph_viewer.GetNode(node_index);
      if (node == nullptr) continue;

      // Check if this node's output matches our input
      for (const auto* output_def : node->OutputDefs()) {
        if (output_def && output_def->Name() == mul_input_name) {
          // Found the producer node, check if it's a Mul
          auto it = node_to_node_unit.find(node);
          if (it != node_to_node_unit.end()) {
            const NodeUnit* producer_unit = it->second;
            if (producer_unit->OpType() == "Mul" &&
                node_unit_to_qnn_node_group.find(producer_unit) == node_unit_to_qnn_node_group.end()) {
              // Check if this Mul has root as one input (no longer checking for constant 0.5)
              const auto& mul2_inputs = producer_unit->Inputs();
              if (mul2_inputs.size() >= 2) {
                bool has_root_input = (mul2_inputs[0].node_arg.Name() == root_input_name ||
                                       mul2_inputs[1].node_arg.Name() == root_input_name);

                if (has_root_input) {
                  mul2_node_unit = producer_unit;
                  break;
                }
              }
            }
          }
        }
      }
      if (mul2_node_unit != nullptr) break;
    }
    if (mul2_node_unit != nullptr) break;
  }

  std::vector<const NodeUnit*> node_units;
  const NodeUnit* final_mul_node_unit = nullptr;

  if (mul2_node_unit != nullptr) {
    // Pattern 1: root -> Mul -> ... -> Mul
    node_units = {div_node_unit, &erf_node_unit, add_node_unit, mul2_node_unit, mul_node_unit};
    final_mul_node_unit = mul_node_unit;
  } else {
    // Try Pattern 2: root -> ... -> Mul -> Mul
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

    const NodeUnit* mul2_node_unit_pattern2 = GetChildOfOutput(graph_viewer, *mul_node_unit, mul_outputs[0],
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

Status GeluFusion::IsSupported(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  const NodeUnitIODef& root_input = node_units_[0]->Inputs()[0];
  const NodeUnitIODef& final_output = node_units_.back()->Outputs()[0];
  return ValidateOnQnn(qmw, node_units_, root_input, final_output);
}

Status GeluFusion::AddToModelBuilder(QnnModelWrapper& qmw, const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
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
