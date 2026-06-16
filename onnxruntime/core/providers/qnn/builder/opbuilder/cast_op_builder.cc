// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class CastOpBuilder : public BaseOpBuilder {
 public:
  CastOpBuilder() : BaseOpBuilder("CastOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CastOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation = false) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  // QNN HTP currently does not support casting FP16/FP32 to Bool, and thus such Cast will be replaced by NotEqual with
  // an additional static input 0.f to achieve the idential functional.
  bool IsFpToBoolCast(const NodeUnit& node_unit) const;
  Status ProcessExtraInputForNotEqual(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      std::vector<std::string>& input_names,
                                      const logging::Logger& logger) const;
};

bool CastOpBuilder::IsFpToBoolCast(const NodeUnit& node_unit) const {
  const auto* input_type_proto = node_unit.Inputs()[0].node_arg.TypeAsProto();
  const auto* output_type_proto = node_unit.Outputs()[0].node_arg.TypeAsProto();

  Qnn_DataType_t input_qnn_dtype = QNN_DATATYPE_UNDEFINED;
  Qnn_DataType_t output_qnn_dtype = QNN_DATATYPE_UNDEFINED;

  if (utils::GetQnnDataType(false, input_type_proto, input_qnn_dtype) != Status::OK() ||
      utils::GetQnnDataType(false, output_type_proto, output_qnn_dtype) != Status::OK()) {
    return false;
  }

  return ((input_qnn_dtype == QNN_DATATYPE_FLOAT_16 || input_qnn_dtype == QNN_DATATYPE_FLOAT_32) &&
          output_qnn_dtype == QNN_DATATYPE_BOOL_8);
}

Status CastOpBuilder::ProcessExtraInputForNotEqual(QnnModelWrapper& qnn_model_wrapper,
                                                   const NodeUnit& node_unit,
                                                   std::vector<std::string>& input_names,
                                                   const logging::Logger& logger) const {
  const auto& input = node_unit.Inputs()[0];
  if (input.quant_param.has_value()) {
    return Status::OK();
  }

  // Build additional static input with value 0.
  const std::string& input_name = utils::GetUniqueName(node_unit, "_notequal_zero");

  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_UNDEFINED;
  const auto* type_proto = input.node_arg.TypeAsProto();
  ORT_RETURN_IF_ERROR(utils::GetQnnDataType(false, type_proto, qnn_data_type));

  QnnTensorWrapper input_tensor_wrapper(input_name,
                                        QNN_TENSOR_TYPE_STATIC,
                                        qnn_data_type,
                                        QnnQuantParamsWrapper(),
                                        std::vector<uint32_t>{1},
                                        std::vector<uint8_t>(utils::GetElementSizeByType(qnn_data_type), 0));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensor_wrapper)),
                    "Failed to add additional input tensor for QNN Cast node that will be replaced by NotEqual.");
  input_names.push_back(input_name);

  LOGS(logger, VERBOSE) << "FP-to-Bool Cast node " << node_unit.Name() << " is replaced by NotEqual.";
  return Status::OK();
}

Status CastOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  const auto& inputs = node_unit.Inputs();
  ORT_RETURN_IF_NOT(inputs.size() == 1, "QNN Cast node must have a single input.");
  const auto& input = inputs[0];

  const auto& input_name = input.node_arg.Name();

  if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
    LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_name;
    input_names.push_back(input_name);
    return IsFpToBoolCast(node_unit)
               ? ProcessExtraInputForNotEqual(qnn_model_wrapper, node_unit, input_names, logger)
               : Status::OK();
  }

  std::vector<uint8_t> unpacked_tensor;
  bool is_constant_input = qnn_model_wrapper.IsConstantInput(input_name);
  if (is_constant_input) {
    const auto& input_tensor = qnn_model_wrapper.GetConstantTensor(input_name);
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_tensor, unpacked_tensor));
  }

  Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(input_name);
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input.node_arg, input_shape),
                    "Cannot get shape for QNN Cast node's input.");

  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_UNDEFINED;
  const auto* type_proto = input.node_arg.TypeAsProto();

  ORT_RETURN_IF_ERROR(utils::GetQnnDataType(false,  // Do not try to get the quantized type. HTP cast supports normal types.
                                            type_proto,
                                            qnn_data_type));

  QnnTensorWrapper input_tensorwrapper(input_name, tensor_type, qnn_data_type, QnnQuantParamsWrapper(),
                                       std::move(input_shape), std::move(unpacked_tensor));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)),
                    "Failed to add input tensor for QNN Cast node.");
  input_names.push_back(input_name);

  return IsFpToBoolCast(node_unit)
             ? ProcessExtraInputForNotEqual(qnn_model_wrapper, node_unit, input_names, logger)
             : Status::OK();
}

Status CastOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);

  const auto& outputs = node_unit.Outputs();
  ORT_RETURN_IF_NOT(outputs.size() == 1, "QNN Cast node must have a single output.");
  const auto& output = outputs[0];
  const auto& output_name = output.node_arg.Name();

  const auto* type_proto = output.node_arg.TypeAsProto();
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_UNDEFINED;
  ORT_RETURN_IF_ERROR(utils::GetQnnDataType(false,  // Do not try to get the quantized type. HTP cast supports normal types.
                                            type_proto,
                                            qnn_data_type));

  std::vector<uint32_t> output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(output.node_arg, output_shape),
                    "Cannot get shape for QNN Cast node's output.");
  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);

  const Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  if (qnn_data_type == QNN_DATATYPE_INT_64 && tensor_type == QNN_TENSOR_TYPE_NATIVE) {
    qnn_data_type = QNN_DATATYPE_INT_32;
  }
  QnnTensorWrapper output_tensorwrapper(output_name,
                                        tensor_type,
                                        qnn_data_type,
                                        QnnQuantParamsWrapper(),
                                        std::move(output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)),
                    "Failed to add output tensor for QNN Cast node.");

  const std::string qnn_op_type = IsFpToBoolCast(node_unit)
                                      ? QNN_OP_ELEMENT_WISE_NOT_EQUAL
                                      : GetQnnOpType(node_unit.OpType());
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(utils::GetUniqueName(node_unit),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    qnn_op_type,
                                                    std::move(input_names),
                                                    {output_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to create " + qnn_op_type + " node.");

  return Status::OK();
}

void CreateCastOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<CastOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
