// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn-abi/builder/qnn_node_group/reshape_gemm_fusion.h"

#include <algorithm>
#include <cassert>
#include <gsl/gsl>
#include <limits>
#include <optional>
#include <string>

#include "core/providers/qnn-abi/builder/op_builder_factory.h"
#include "core/providers/qnn-abi/builder/qnn_model_wrapper.h"
#include "core/providers/qnn-abi/builder/qnn_node_group/utils.h"
#include "core/providers/qnn-abi/builder/qnn_utils.h"
#include "core/providers/qnn-abi/ort_api.h"

namespace onnxruntime {
namespace qnn {

namespace {

const OrtNodeUnit* GetReshapeNodeUnit(
    const QnnModelWrapper& qnn_model_wrapper,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const OrtNode& gemm_node) {
  if (Ort::ConstNode(&gemm_node).GetOperatorType() != "Gemm") {
    return nullptr;
  }

  // Get gemm node inputs
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();
  size_t num_inputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(&gemm_node, &num_inputs), ort_api, nullptr);
  if (num_inputs < 1) {
    return nullptr;
  }
  std::vector<const OrtValueInfo*> inputs(num_inputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetInputs(&gemm_node, inputs.data(), inputs.size()), ort_api, nullptr);

  // Get the first input (index 0)
  const OrtValueInfo* input_value_info = inputs[0];

  // Get the producer of this input
  const OrtNode* reshape_node = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_GetValueProducer(input_value_info, &reshape_node, nullptr),
                             ort_api,
                             nullptr);

  // Check if it's a Reshape node
  if (reshape_node != nullptr && Ort::ConstNode(reshape_node).GetOperatorType() == "Reshape") {
    // Check if reshape node produces graph output
    size_t num_outputs = 0;
    if (num_outputs != 1) {
      return nullptr;
    }
    RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(reshape_node, &num_outputs), ort_api, nullptr);
    std::vector<const OrtValueInfo*> reshape_outputs(num_outputs);
    RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(reshape_node, reshape_outputs.data(), reshape_outputs.size()),
                               ort_api,
                               nullptr);

    const OrtValueInfo* output_info = reshape_outputs[0];

    bool is_graph_output = false;
    RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_IsGraphOutput(output_info, &is_graph_output), ort_api, nullptr);

    if (!is_graph_output) {
      size_t num_consumers;
      RETURN_DEFAULT_IF_API_FAIL(ort_api.ValueInfo_GetValueNumConsumers(output_info, &num_consumers),
                                 ort_api,
                                 nullptr);
      if (num_consumers == 1) {
        const auto it = node_to_node_unit.find(reshape_node);
        if (it != node_to_node_unit.end()) {
          const OrtNodeUnit* reshape_node_unit = it->second;
          if (reshape_node_unit && node_unit_to_qnn_node_group.count(reshape_node_unit) == 0 &&
              reshape_node_unit->UnitType() == OrtNodeUnit::Type::SingleNode) {
            return reshape_node_unit;
          }
        }
      }
    }
  }
  return nullptr;
}

bool CheckShape(const QnnModelWrapper& qnn_model_wrapper, const OrtNode& reshape_node) {
  const OrtApi& ort_api = qnn_model_wrapper.GetOrtApi();

  // Get reshape node inputs and outputs

  size_t num_inputs = 0;
  size_t num_outputs = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumInputs(&reshape_node, &num_inputs), ort_api, false);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetNumOutputs(&reshape_node, &num_outputs), ort_api, false);

  std::vector<const OrtValueInfo*> inputs(num_inputs);
  std::vector<const OrtValueInfo*> outputs(num_outputs);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetInputs(&reshape_node, inputs.data(), inputs.size()), ort_api, false);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.Node_GetOutputs(&reshape_node, outputs.data(), outputs.size()), ort_api, false);

  const OrtValueInfo* input_info = inputs[0];
  const OrtValueInfo* output_info = outputs[0];

  // Get type info for input and output
  const OrtTypeInfo* input_type_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoTypeInfo(input_info, &input_type_info), ort_api, false);
  const OrtTypeInfo* output_type_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetValueInfoTypeInfo(output_info, &output_type_info), ort_api, false);

  if (!input_type_info || !output_type_info) {
    return false;
  }

  // Cast to tensor info
  const OrtTensorTypeAndShapeInfo* input_tensor_info = nullptr;
  const OrtTensorTypeAndShapeInfo* output_tensor_info = nullptr;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.CastTypeInfoToTensorInfo(input_type_info, &input_tensor_info), ort_api, false);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.CastTypeInfoToTensorInfo(output_type_info, &output_tensor_info), ort_api, false);

  if (!input_tensor_info || !output_tensor_info) {
    return false;
  }

  // Get dimensions
  size_t input_dims_count = 0;
  size_t output_dims_count = 0;
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensionsCount(input_tensor_info, &input_dims_count), ort_api, false);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensionsCount(output_tensor_info, &output_dims_count), ort_api, false);

  if (output_dims_count != 2) {
    return false;
  }

  std::vector<int64_t> input_dims(input_dims_count);
  std::vector<int64_t> output_dims(output_dims_count);

  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensions(input_tensor_info, input_dims.data(), input_dims_count),
                             ort_api,
                             false);
  RETURN_DEFAULT_IF_API_FAIL(ort_api.GetDimensions(output_tensor_info, output_dims.data(), output_dims_count),
                             ort_api,
                             false);

  // Check if the reshape is from [x0, x1, ..., xn, k] to [x0 * x1 * ... * xn, k]
  int64_t input_product = 1;
  for (size_t i = 0; i < input_dims_count - 1; ++i) {
    if (input_dims[i] <= 0) {
      return false;
    }
    input_product *= input_dims[i];
  }

  bool result = (input_product == output_dims[0] && input_dims[input_dims_count - 1] == output_dims[1]);

  return result;
}

Ort::Status CreateOrValidateOnQnn(QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& reshape_node_unit,
                                  const OrtNodeUnit& gemm_node_unit, bool validate) {
  assert(reshape_node_unit.OpType() == "Reshape" && gemm_node_unit.OpType() == "Gemm");
  const auto& node_name = utils::GetUniqueName(gemm_node_unit);
  const OrtNodeUnitIODef& input_def = reshape_node_unit.Inputs()[0];
  const OrtNodeUnitIODef& weight_def = gemm_node_unit.Inputs()[1];
  const OrtNodeUnitIODef* bias_def_ptr = nullptr;
  bool has_bias = gemm_node_unit.Inputs().size() == 3;
  if (has_bias) {
    bias_def_ptr = &gemm_node_unit.Inputs()[2];
  }
  const OrtNodeUnitIODef& output_def = gemm_node_unit.Outputs()[0];

  QnnTensorWrapper input_tensor;
  QnnTensorWrapper bias_tensor;
  QnnTensorWrapper output_tensor;

  // Create input tensor wrapper
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(input_def, input_tensor));

  // Process weight tensor
  std::vector<uint32_t> weight_shape;
  std::vector<uint8_t> unpacked_tensor;
  std::string weight_tensor_name = weight_def.name;

  // Get weight shape and validate
  RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(weight_def.shape, weight_shape), "Failed to get weight shape");

  // Get tensor type for weight
  Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(weight_tensor_name);
  Qnn_DataType_t data_type = QNN_DATATYPE_FLOAT_32;
  RETURN_IF_ERROR(utils::GetQnnDataType(false, weight_def.type, data_type));

  // Get weight tensor proto and perform 2D transpose
  const auto* weight_tensor_proto = qnn_model_wrapper.GetConstantTensor(weight_tensor_name);
  // Transpose the weight tensor (2D matrix transpose)
  RETURN_IF_ERROR(utils::TwoDimensionTranspose(qnn_model_wrapper, weight_shape, weight_tensor_proto, unpacked_tensor));
  QnnTensorWrapper weight_tensor(weight_tensor_name, tensor_type, data_type, QnnQuantParamsWrapper(),
                                 std::move(weight_shape), std::move(unpacked_tensor));
  if (has_bias) {
    RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(*bias_def_ptr, bias_tensor));
  }

  // Create output tensor wrapper
  RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(output_def, output_tensor));

  if (validate) {
    std::vector<Qnn_Tensor_t> input_tensors = {input_tensor.GetQnnTensor(), weight_tensor.GetQnnTensor()};
    if (has_bias) {
      input_tensors.emplace_back(bias_tensor.GetQnnTensor());
    }

    // Validate the QNN node
    RETURN_IF_ERROR(qnn_model_wrapper.ValidateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_FULLY_CONNECTED, std::move(input_tensors),
                                                      {output_tensor.GetQnnTensor()}, {}));
  } else {
    // For creation, add all tensor wrappers to the model
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor)), "Failed to add input");
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(weight_tensor)), "Failed to add weight");

    // Add bias tensor if it exists
    if (has_bias) {
      RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bias_tensor)), "Failed to add bias");
    }

    // Add output tensor
    RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensor)), "Failed to add output");

    // Create input names vector
    std::vector<std::string> input_names = {input_def.name, weight_tensor_name};
    if (has_bias) {
      input_names.emplace_back(bias_def_ptr->name);
    }

    // Create the QNN node for fully connected operation
    RETURN_IF_NOT(
        qnn_model_wrapper.CreateQnnNode(node_name, QNN_OP_PACKAGE_NAME_QTI_AISW, QNN_OP_FULLY_CONNECTED,
                                        std::move(input_names), {output_def.name}, {}, validate),
        "Failed to add fused Gemm node.");
  }

  return Ort::Status();
}

}  // namespace

std::unique_ptr<IQnnNodeGroup> ReshapeGemmFusion::TryFusion(
    QnnModelWrapper& qnn_model_wrapper, const OrtNodeUnit& gemm_node_unit,
    const std::unordered_map<const OrtNode*, const OrtNodeUnit*>& node_to_node_unit,
    const std::unordered_map<const OrtNodeUnit*, const IQnnNodeGroup*>& node_unit_to_qnn_node_group,
    const Ort::Logger& logger) {
  ORT_UNUSED_PARAMETER(logger);
  if (gemm_node_unit.OpType() != "Gemm" || gemm_node_unit.UnitType() != OrtNodeUnit::Type::SingleNode) {
    return nullptr;
  }

  const OrtNode& gemm_node = gemm_node_unit.GetNode();

  // Check transA and transB attributes
  OrtNodeAttrHelper attr_helper(gemm_node_unit);
  float transA = attr_helper.Get("transA", 0.0f);
  float transB = attr_helper.Get("transB", 0.0f);

  // The pattern is from MatMul->Add, so the transA and transB should be false
  if (transA != 0.0f || transB != 0.0f) {
    return nullptr;
  }

  // Check if weight is constant
  const OrtNodeUnitIODef& weight_input = gemm_node_unit.Inputs()[1];
  // The pattern is from MatMul->Add, so the transA and transB should be false, and weight should be initializer.
  // Currently we don't handle quantized weight.
  if (!qnn_model_wrapper.IsConstantInput(weight_input.name) || weight_input.quant_param.has_value()) {
    return nullptr;
  }

  // Find the reshape node unit
  const OrtNodeUnit* reshape_node_unit = GetReshapeNodeUnit(qnn_model_wrapper, node_to_node_unit, node_unit_to_qnn_node_group, gemm_node);
  if (!reshape_node_unit) {
    return nullptr;
  }

  if (!CheckShape(qnn_model_wrapper, reshape_node_unit->GetNode())) {
    return nullptr;
  }

  return std::make_unique<ReshapeGemmFusion>(*reshape_node_unit, gemm_node_unit);
}

ReshapeGemmFusion::ReshapeGemmFusion(const OrtNodeUnit& reshape_node_unit, const OrtNodeUnit& gemm_node_unit)
    : node_units_{&reshape_node_unit, &gemm_node_unit} {
}

Ort::Status ReshapeGemmFusion::IsSupported(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOrValidateOnQnn(qmw, *node_units_[0], *node_units_[1], true);
}

Ort::Status ReshapeGemmFusion::AddToModelBuilder(QnnModelWrapper& qmw, const Ort::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  return CreateOrValidateOnQnn(qmw, *node_units_[0], *node_units_[1], false);
}

gsl::span<const OrtNodeUnit* const> ReshapeGemmFusion::GetNodeUnits() const {
  return node_units_;
}

const OrtNodeUnit* ReshapeGemmFusion::GetTargetNodeUnit() const {
  return node_units_[1];
}

}  // namespace qnn
}  // namespace onnxruntime
