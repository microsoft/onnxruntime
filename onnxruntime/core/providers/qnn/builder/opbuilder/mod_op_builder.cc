// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_set>
#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/cpu/tensor/slice_helper.h"

namespace onnxruntime {
namespace qnn {
class ModOpBuilder : public BaseOpBuilder {
 public:
  ModOpBuilder() : BaseOpBuilder("ModOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ModOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

// Mod supported Qnn datatype is limited by floor supported datatype.
bool IsCastRequired(int target_tensor_type) {
  static const std::unordered_set<int> supported_types = {
      QNN_DATATYPE_FLOAT_16,
      QNN_DATATYPE_FLOAT_32,
      QNN_DATATYPE_UFIXED_POINT_16,
      QNN_DATATYPE_UFIXED_POINT_8,
      QNN_DATATYPE_SFIXED_POINT_8};
  return supported_types.count(target_tensor_type) == 0;
}

Status ModOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                   const NodeUnit& node_unit,
                                   const logging::Logger& logger,
                                   std::vector<std::string>& input_names,
                                   bool do_op_validation) const {
  NodeAttrHelper node_helper(node_unit);
  int64_t fmod = node_helper.Get("fmod", static_cast<int64_t>(0));  // 0=integer mod. 1=float mod.
  if (1 == fmod) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN Mod Op only support fmod == 0 for now.");
  }

  const auto& inputs = node_unit.Inputs();
  const auto input_count = GetInputCountQnnRequired(node_unit);
  for (size_t input_i = 0; input_i < input_count; ++input_i) {
    ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[input_i], logger, input_names));
  }

  return Status::OK();
}

Status ModOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                 const NodeUnit& node_unit,
                                                 std::vector<std::string>&& input_names,
                                                 const logging::Logger& logger,
                                                 bool do_op_validation) const {
  TensorInfo input_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[0], input_info));
  TensorInfo output_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Outputs()[0], output_info));

  NodeAttrHelper node_helper(node_unit);
  int64_t fmod = node_helper.Get("fmod", static_cast<int64_t>(0));
  std::string& input_a_name = input_names[0];
  std::string& input_b_name = input_names[1];

  const std::string& org_output_name = node_unit.Outputs()[0].node_arg.Name();
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(org_output_name);

  std::vector<uint32_t> output_shape = output_info.shape;
  Qnn_TensorType_t op_output_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  // ElementWiseFloor only support QNN_DATATYPE_FLOAT_16, QNN_DATATYPE_FLOAT_32, QNN_DATATYPE_UFIXED_POINT_16, QNN_DATATYPE_UFIXED_POINT_8, QNN_DATATYPE_SFIXED_POINT_8
  // If not one of those, cast to QNN_DATATYPE_FLOAT_32.
  Qnn_DataType_t target_tensor_type = input_info.qnn_data_type;
  bool is_cast_required = IsCastRequired(target_tensor_type);

  if (is_cast_required) {
    target_tensor_type = QNN_DATATYPE_FLOAT_32;
    std::string cast_a_name = onnxruntime::qnn::utils::GetNodeName(node_unit) + "_Cast_A";
    std::string cast_a_output_name = cast_a_name + "_output";
    QnnTensorWrapper cast_a_output(cast_a_output_name,
                                   QNN_TENSOR_TYPE_NATIVE,
                                   target_tensor_type,
                                   QnnQuantParamsWrapper(),
                                   std::vector<uint32_t>(output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_a_output)),
                      "Failed to add output tensor for QNN Cast node.");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(cast_a_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_CAST,
                                                      {input_a_name},
                                                      {cast_a_output_name},
                                                      {},
                                                      false),
                      "Failed to create QNN Cast node.");
    input_a_name = cast_a_output_name;

    TensorInfo input_B_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(node_unit.Inputs()[1], input_B_info));
    std::vector<uint32_t> input_B_shape = input_B_info.shape;
    std::string cast_b_name = onnxruntime::qnn::utils::GetNodeName(node_unit) + "_Cast_B";
    std::string cast_b_output_name = cast_b_name + "_output";
    QnnTensorWrapper cast_b_output(cast_b_output_name,
                                   QNN_TENSOR_TYPE_NATIVE,
                                   target_tensor_type,
                                   QnnQuantParamsWrapper(),
                                   std::vector<uint32_t>(input_B_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(cast_b_output)),
                      "Failed to add output tensor for QNN Cast node.");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(cast_b_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_CAST,
                                                      {input_b_name},
                                                      {cast_b_output_name},
                                                      {},
                                                      false),
                      "Failed to create QNN Cast node.");
    input_b_name = cast_b_output_name;
  }

  if (0 == fmod) {
    // Implement mod(a, b) = a - b * floor(a / b)
    // 1. ElementWiseDiv
    std::vector<std::string> div_input;
    div_input.push_back(input_a_name);
    div_input.push_back(input_b_name);

    std::string div_name = onnxruntime::qnn::utils::GetNodeName(node_unit) + "_Div";
    std::string div_output_name = div_name + "_output";
    QnnTensorWrapper div_output(div_output_name,
                                QNN_TENSOR_TYPE_NATIVE,
                                target_tensor_type,
                                QnnQuantParamsWrapper(),
                                std::vector<uint32_t>(output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(div_output)),
                      "Failed to add Mod - ElementWiseDiv output tensor.");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(div_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_ELEMENT_WISE_DIVIDE,
                                                      std::move(div_input),
                                                      {div_output_name},
                                                      {},
                                                      do_op_validation),
                      "Failed to add Mod - ElementWiseDiv node.");

    // 2. ElementWiseFloor
    std::string floor_name = onnxruntime::qnn::utils::GetNodeName(node_unit) + "_Floor";
    std::string floor_output_name = floor_name + "_output";
    QnnTensorWrapper floor_output(floor_output_name,
                                  QNN_TENSOR_TYPE_NATIVE,
                                  target_tensor_type,
                                  QnnQuantParamsWrapper(),
                                  std::vector<uint32_t>(output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(floor_output)),
                      "Failed to add Mod - ElementWiseFloor output tensor.");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(floor_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_ELEMENT_WISE_FLOOR,
                                                      {div_output_name},
                                                      {floor_output_name},
                                                      {},
                                                      do_op_validation),
                      "Failed to add Mod - ElementWiseFloor node.");

    // 3. ElementWiseMul
    std::vector<std::string> mul_input;
    mul_input.push_back(input_b_name);
    mul_input.push_back(floor_output_name);
    std::string mul_name = onnxruntime::qnn::utils::GetNodeName(node_unit) + "_Mul";
    std::string mul_output_name = mul_name + "_output";
    QnnTensorWrapper mul_output(mul_output_name,
                                QNN_TENSOR_TYPE_NATIVE,
                                target_tensor_type,
                                QnnQuantParamsWrapper(),
                                std::vector<uint32_t>(output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(mul_output)),
                      "Failed to add Mod - ElementWiseMul output tensor.");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(mul_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_ELEMENT_WISE_MULTIPLY,
                                                      std::move(mul_input),
                                                      {mul_output_name},
                                                      {},
                                                      do_op_validation),
                      "Failed to add Mod - ElementWiseMul node.");

    // 4. ElementWiseSub
    std::vector<std::string> sub_input;
    sub_input.push_back(input_a_name);
    sub_input.push_back(mul_output_name);
    std::string sub_name = onnxruntime::qnn::utils::GetNodeName(node_unit) + "_Sub";
    std::string sub_output_name = is_cast_required ? sub_name + "_output" : org_output_name;
    QnnTensorWrapper mod_output(sub_output_name,
                                is_cast_required ? QNN_TENSOR_TYPE_NATIVE : op_output_tensor_type,
                                is_cast_required ? target_tensor_type : output_info.qnn_data_type,
                                is_cast_required ? QnnQuantParamsWrapper() : output_info.quant_param.Copy(),
                                std::vector<uint32_t>(output_shape));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(mod_output)),
                      "Failed to add Mod output tensor.");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(sub_name,
                                                      QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                      QNN_OP_ELEMENT_WISE_SUBTRACT,
                                                      std::move(sub_input),
                                                      {sub_output_name},
                                                      {},
                                                      do_op_validation),
                      "Failed to add Mod - ElementWiseSub node.");

    if (is_cast_required) {
      std::string output_cast_name = onnxruntime::qnn::utils::GetNodeName(node_unit) + "_output_Cast";
      QnnTensorWrapper mod_output(org_output_name,
                                  op_output_tensor_type,
                                  output_info.qnn_data_type,
                                  output_info.quant_param.Copy(),
                                  std::vector<uint32_t>(output_shape));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(mod_output)),
                        "Failed to add output tensor for QNN Cast node.");
      ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(output_cast_name,
                                                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                        QNN_OP_CAST,
                                                        {sub_output_name},
                                                        {org_output_name},
                                                        {},
                                                        false),
                        "Failed to create QNN Cast node.");
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN Mod Op doesn't support fmod.");
  }

  return Status::OK();
}

void CreateModOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ModOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
