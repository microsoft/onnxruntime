// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <vector>

#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

#include "base_op_builder.h"

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
};

Status CastOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);

  const auto& inputs = node_unit.Inputs();
  ORT_ENFORCE(inputs.size() == 1, "QNN Cast node must have a single input.");
  const auto& input = inputs[0];

  const auto& input_name = input.node_arg.Name();

  if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
    LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_name;
    input_names.push_back(input_name);
    return Status::OK();
  }

  std::vector<uint8_t> unpacked_tensor;
  bool is_initializer_input = qnn_model_wrapper.IsInitializerInput(input_name);
  if (is_initializer_input) {
    const auto& input_tensor = qnn_model_wrapper.GetInitializerTensors().at(input_name);
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_tensor, unpacked_tensor));
  }

  Qnn_TensorType_t tensor_type = GetInputTensorType(qnn_model_wrapper, input_name);
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input.node_arg, input_shape),
                    "Cannot get shape for QNN Cast node's input.");

  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_UNDEFINED;
  const auto* type_proto = input.node_arg.TypeAsProto();

  ORT_RETURN_IF_ERROR(utils::GetQnnDataType(false,  // Do not try to get the quantized type. HTP cast supports normal types.
                                            type_proto,
                                            qnn_data_type));

  QnnTensorWrapper input_tensorwrapper(input_name, tensor_type, qnn_data_type, QNN_QUANTIZE_PARAMS_INIT,
                                       std::move(input_shape), std::move(unpacked_tensor));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)),
                    "Failed to add input tensor for QNN Cast node.");
  input_names.push_back(input_name);

  return Status::OK();
}

Status CastOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);

  const auto& outputs = node_unit.Outputs();
  ORT_ENFORCE(outputs.size() == 1, "QNN Cast node must have a single output.");
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
  QnnTensorWrapper output_tensorwrapper(output_name,
                                        tensor_type,
                                        qnn_data_type,
                                        QNN_QUANTIZE_PARAMS_INIT,
                                        std::move(output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)),
                    "Failed to add output tensor for QNN Cast node.");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(GetNodeName(node_unit),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    GetQnnOpType(node_unit.OpType()),
                                                    std::move(input_names),
                                                    {output_name},
                                                    {},
                                                    do_op_validation),
                    "Failed to create QNN Cast node.");

  return Status::OK();
}

void CreateCastOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<CastOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
