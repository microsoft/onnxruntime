// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class STFTOpBuilder : public BaseOpBuilder {
 public:
  STFTOpBuilder() : BaseOpBuilder("STFTOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(STFTOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override;

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger) const override;
};

// Checks if the given input is a window input (float type).
static bool IsWindowInput(const NodeUnitIODef& input) {
  return input.node_arg.TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT;
}

Status STFTOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);
  // TODO: STFT seg faults on QNN CPU
  bool is_cpu_backend = IsCpuBackend(qnn_model_wrapper.GetQnnBackendType());
  ORT_RETURN_IF(is_cpu_backend, "QNN EP: STFT Op disabled in CPU backend.");
  // General Datatype checks on various QNN backend (HTP, CPU, GPU)
  ORT_RETURN_IF_ERROR(ProcessDataTypes(qnn_model_wrapper, node_unit));
  return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
}

Status STFTOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();

  // Process signal input (first input)
  const auto& signal_input = inputs[0];
  const std::string& signal_input_name = signal_input.node_arg.Name();

  // Get the shape of the signal input
  TensorInfo signal_info{};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(signal_input, signal_info));

  // Check if the signal input is rank 2 (needs expansion to rank 3)
  if (signal_info.shape.size() == 2) {
    LOGS(logger, VERBOSE) << "Signal input is rank 2, adding ExpandDims op to convert to rank 3";

    // Add the original signal tensor
    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(signal_input_name)) {
      QnnTensorWrapper signal_tensorwrapper;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(signal_input, signal_tensorwrapper));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(signal_tensorwrapper)),
                        "Failed to add signal tensor.");
    }

    // Create a name for the expanded tensor
    std::string expanded_tensor_name = signal_input_name + "_expanded";

    // Create the expanded tensor with an extra dimension
    std::vector<uint32_t> expanded_shape = signal_info.shape;
    expanded_shape.push_back(1);

    // Create a tensor info for the expanded tensor based on the signal tensor info
    TensorInfo expanded_tensor_info = signal_info;
    expanded_tensor_info.shape = expanded_shape;

    // Create the tensor wrapper using MakeTensorWrapper
    QnnTensorWrapper expanded_tensorwrapper;
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(expanded_tensor_info,
                                                            expanded_tensor_name,
                                                            expanded_tensorwrapper));

    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(expanded_tensorwrapper)),
                      "Failed to add expanded signal tensor.");

    // Create axis parameter for ExpandDims (add dimension at the end)
    Qnn_Scalar_t axis_param = QNN_SCALAR_INIT;
    axis_param.dataType = QNN_DATATYPE_UINT_32;
    axis_param.uint32Value = static_cast<uint32_t>(2);  // Add at the end

    QnnParamWrapper axis_param_wrapper(node_unit.Index(),
                                       node_unit.Name(),
                                       "axis",
                                       axis_param);

    std::vector<std::string> expand_dims_params;
    expand_dims_params.push_back(axis_param_wrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(axis_param_wrapper));

    // Create the ExpandDims node
    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                          utils::GetUniqueName(signal_input_name, "_expand_dims"),
                          QNN_OP_PACKAGE_NAME_QTI_AISW,
                          QNN_OP_EXPAND_DIMS,
                          {signal_input_name},
                          {expanded_tensor_name},
                          std::move(expand_dims_params),
                          do_op_validation),
                      "Failed to create ExpandDims node.");

    // Use the expanded tensor for STFT
    input_names.push_back(expanded_tensor_name);
  } else {
    // Process as normal for rank 3 or higher inputs
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(signal_input_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << signal_input_name;
    } else {
      QnnTensorWrapper signal_tensorwrapper;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(signal_input, signal_tensorwrapper));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(signal_tensorwrapper)),
                        "Failed to add signal tensor.");
    }
    input_names.push_back(signal_input_name);
  }

  // Process frame_step input (second input)
  if (inputs.size() > 1) {
    const auto& frame_step_input = inputs[1];
    const std::string& frame_step_input_name = frame_step_input.node_arg.Name();

    if (qnn_model_wrapper.IsQnnTensorWrapperExist(frame_step_input_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << frame_step_input_name;
    } else {
      QnnTensorWrapper frame_step_tensorwrapper;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(frame_step_input, frame_step_tensorwrapper));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(frame_step_tensorwrapper)),
                        "Failed to add frame_step tensor.");
    }
    // We don't add frame_step to input_names because it will be processed as a parameter
  }

  // Process the 'window' input if it exists and is of type float
  if (inputs.size() > 2) {
    const auto& window_input = inputs[2];
    if (IsWindowInput(window_input)) {
      const std::string& window_input_name = window_input.node_arg.Name();

      if (qnn_model_wrapper.IsQnnTensorWrapperExist(window_input_name)) {
        LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << window_input_name;
      } else {
        QnnTensorWrapper window_tensorwrapper;
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(window_input, window_tensorwrapper));
        ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(window_tensorwrapper)),
                          "Failed to add window tensor.");
      }
      input_names.push_back(window_input_name);
    }
  }

  // Process frame_length input if it exists
  if (inputs.size() > 3 || (inputs.size() > 2 && !IsWindowInput(inputs[2]))) {
    int frame_length_index = inputs.size() > 3 ? 3 : 2;
    const auto& frame_length_input = inputs[frame_length_index];
    const std::string& frame_length_input_name = frame_length_input.node_arg.Name();

    if (qnn_model_wrapper.IsQnnTensorWrapperExist(frame_length_input_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << frame_length_input_name;
    } else {
      QnnTensorWrapper frame_length_tensorwrapper;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.MakeTensorWrapper(frame_length_input, frame_length_tensorwrapper));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(frame_length_tensorwrapper)),
                        "Failed to add frame_length tensor.");
    }
    // We don't add frame_length to input_names because it will be processed as a parameter
  }

  return Status::OK();
}

Status STFTOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  NodeAttrHelper node_helper(node_unit);
  const auto& inputs = node_unit.Inputs();
  bool onesided = node_helper.Get("onesided", static_cast<bool>(1));

  std::vector<std::string> param_tensor_names;
  // Extract frame_step from the inputs if it exists
  uint32_t frame_step_info = 0;
  TensorInfo frame_step = {};
  TensorInfo frame_length = {};
  uint32_t frame_length_info = 0;

  int frame_length_index = -1;
  int frame_step_index = -1;

  // Determine indices for frame_step and frame_length
  if (inputs.size() >= 2) {
    frame_step_index = 1;  // frame_step is always the second input
  }

  if (inputs.size() == 3) {
    // Check if the third input is window or frame_length
    const auto& third_input = inputs[2];
    if (!IsWindowInput(third_input)) {
      frame_length_index = 2;  // It's frame_length
    }
  } else if (inputs.size() > 3) {
    frame_length_index = 3;  // frame_length is the fourth input
  }

  // Process frame_step
  if (frame_step_index != -1) {
    const auto& frame_step_input = inputs[frame_step_index];
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(frame_step_input, frame_step));
    std::vector<uint8_t> frame_step_data;
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*frame_step.initializer_tensor, frame_step_data));
    frame_step_info = *reinterpret_cast<uint32_t*>(frame_step_data.data());
    Qnn_Scalar_t frame_step_param = QNN_SCALAR_INIT;
    frame_step_param.dataType = QNN_DATATYPE_UINT_32;
    frame_step_param.uint32Value = frame_step_info;
    QnnParamWrapper frame_step_param_wrapper(node_unit.Index(),
                                             node_unit.Name(),
                                             QNN_OP_STFT_PARAM_FRAME_STEP,
                                             frame_step_param);
    param_tensor_names.push_back(frame_step_param_wrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(frame_step_param_wrapper));
  }

  // Process frame_length if it exists
  if (frame_length_index != -1) {
    const auto& frame_length_input = inputs[frame_length_index];
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(frame_length_input, frame_length));

    std::vector<uint8_t> frame_length_data;
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*frame_length.initializer_tensor, frame_length_data));
    frame_length_info = *reinterpret_cast<uint32_t*>(frame_length_data.data());

    // Create frame_length parameter
    Qnn_Scalar_t frame_length_param = QNN_SCALAR_INIT;
    frame_length_param.dataType = QNN_DATATYPE_UINT_32;
    frame_length_param.uint32Value = frame_length_info;
    QnnParamWrapper frame_length_param_wrapper(node_unit.Index(),
                                               node_unit.Name(),
                                               QNN_OP_STFT_PARAM_FRAME_LENGTH,
                                               frame_length_param);
    param_tensor_names.push_back(frame_length_param_wrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(frame_length_param_wrapper));
  }

  const auto& outputs = node_unit.Outputs();
  const std::string& output_name = outputs[0].node_arg.Name();

  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
  Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;

  TensorInfo output_info{};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], output_info));

  QnnTensorWrapper output_tensorwrapper(output_name, tensor_type, output_info.qnn_data_type,
                                        output_info.quant_param.Copy(), std::move(output_info.shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add output tensor.");

  Qnn_Scalar_t onesided_param = QNN_SCALAR_INIT;
  onesided_param.dataType = QNN_DATATYPE_BOOL_8;
  onesided_param.bool8Value = static_cast<bool>(onesided);

  QnnParamWrapper onesided_param_wrapper(node_unit.Index(),
                                         node_unit.Name(),
                                         QNN_OP_STFT_PARAM_ONESIDED,
                                         onesided_param);

  param_tensor_names.push_back(onesided_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(onesided_param_wrapper));

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(
                        utils::GetUniqueName(node_unit),
                        QNN_OP_PACKAGE_NAME_QTI_AISW,
                        QNN_OP_STFT,
                        std::move(input_names),
                        {output_name},
                        std::move(param_tensor_names),
                        do_op_validation),
                    "Failed to create STFT node.");

  return Status::OK();
}

void CreateSTFTOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<STFTOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime