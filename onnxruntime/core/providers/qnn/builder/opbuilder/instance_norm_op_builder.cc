// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/common/safeint.h"
#include "onnx/defs/data_type_utils.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace qnn {

class InstanceNormOpBuilder : public BaseOpBuilder {
 public:
  InstanceNormOpBuilder() : BaseOpBuilder("InstanceNormOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(InstanceNormOpBuilder);

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger) const override final ORT_MUST_USE_RESULT;

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

// Instance normalization op is sensitive to data layout.
// The nodes from 1st call of GetCapability do not get layout transformer applied, so their shapes are still NCHW.
// The nodes from 2nd call of GetCapability get their layout transformed to NHWC.
// Therefore, we need to check the node domain to determine if the layout has been transformed.
Status InstanceNormOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                            const NodeUnit& node_unit,
                                            const logging::Logger& logger) const {
  ORT_UNUSED_PARAMETER(logger);

  // Check input type is float for CPU.
  const auto& inputs = node_unit.Inputs();
  // Check input type is float for CPU. Can't use Qnn Op validation API since it's before layout transformation
  ORT_RETURN_IF_ERROR(DataTypeCheckForCpuBackend(qnn_model_wrapper, inputs[0].node_arg.Type()));

  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape of input 0");
  const size_t input_rank = input_shape.size();

  if (input_rank <= 2 || input_rank > 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN InstanceNorm only supports input ranks of size 3 or 4.");
  }

  const uint32_t num_channels = (node_unit.Domain() == kMSInternalNHWCDomain) ? input_shape.back() : input_shape[1];

  std::vector<uint32_t> scale_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[1].node_arg, scale_shape), "Cannot get shape of input 1 (scale)");
  if (scale_shape.size() != 1 || scale_shape[0] != num_channels) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN InstanceNorm input 1 (scale) must have 1D shape [channel].");
  }

  std::vector<uint32_t> bias_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[2].node_arg, bias_shape), "Cannot get shape of input 2 (bias)");
  if (bias_shape.size() != 1 || bias_shape[0] != num_channels) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN InstanceNorm input 2 (bias) must have 1D shape [channel].");
  }

  NodeAttrHelper node_helper(node_unit);
  const float epsilon = node_helper.Get("epsilon", 1e-05f);  // Default is 1e-05 according to ONNX spec.
  if (epsilon <= 0.0f) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN InstanceNorm epsilon must be greater than 0.0");
  }

  // Continue Op validation if it's NHWC transformed
  if (node_unit.Domain() == kMSInternalNHWCDomain) {
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, true);
  }

  return Status::OK();
}

Status InstanceNormOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                            const NodeUnit& node_unit,
                                            const logging::Logger& logger,
                                            std::vector<std::string>& input_names,
                                            bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();

  TensorInfo input0_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(inputs[0], input0_info));

  // HTP backend can only handle rank 3 inputs if the batch size is 1. If the batch size is not 1,
  // QNN EP must reshape the input and output to (N, 1, W, C) and process the InstanceNorm as rank 4.
  if (IsNpuBackend(qnn_model_wrapper.GetQnnBackendType()) &&
      input0_info.shape.size() == 3 && input0_info.shape[0] != 1) {
    const std::string& orig_input0_name = inputs[0].node_arg.Name();
    const std::string op_input0_name = input0_info.is_initializer ? orig_input0_name
                                                                  : orig_input0_name + "_ort_qnn_ep_reshape";
    input_names.push_back(op_input0_name);

    std::vector<uint8_t> initializer_data;
    if (input0_info.is_initializer) {
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input0_info.initializer_tensor, initializer_data));
    }

    std::vector<uint32_t> op_shape = {
        input0_info.shape[0],  // N
        1,                     // Height == 1
        input0_info.shape[1],  // Width
        input0_info.shape[2]   // Channels
    };

    if (!input0_info.is_initializer) {
      // Add Reshape node to transform 1D input to 2D (i.e., set height to 1).
      // We don't need to do this for initializers, because the element layout does not change. We can just
      // modify the shape dimensions.
      bool is_graph_input = qnn_model_wrapper.IsGraphInput(orig_input0_name);
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(orig_input0_name,
                                                           op_input0_name,
                                                           input0_info.shape,
                                                           op_shape,
                                                           input0_info.qnn_data_type,
                                                           input0_info.quant_param,
                                                           do_op_validation,
                                                           is_graph_input));
    }

    Qnn_TensorType_t tensor_type = GetInputTensorType(qnn_model_wrapper, op_input0_name);
    QnnTensorWrapper input_tensorwrapper(op_input0_name, tensor_type, input0_info.qnn_data_type, input0_info.quant_param,
                                         std::move(op_shape), std::move(initializer_data));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  } else {
    ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, input_names));  // Input 0
  }

  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[1], logger, input_names));  // Scale
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[2], logger, input_names));  // Bias

  return Status::OK();
}

Status InstanceNormOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                          const NodeUnit& node_unit,
                                                          std::vector<std::string>&& input_names,
                                                          const logging::Logger& logger,
                                                          bool do_op_validation) const {
  NodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_tensor_names;

  const float epsilon = node_helper.Get("epsilon", 1e-05f);  // Default is 1e-05 according to ONNX spec.
  Qnn_Scalar_t epsilon_param = QNN_SCALAR_INIT;
  epsilon_param.dataType = QNN_DATATYPE_FLOAT_32;
  epsilon_param.floatValue = epsilon;
  QnnParamWrapper epsilon_param_wrapper(node_unit.Index(),
                                        node_unit.Name(),
                                        QNN_OP_INSTANCE_NORM_PARAM_EPSILON,
                                        epsilon_param);
  param_tensor_names.push_back(epsilon_param_wrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(epsilon_param_wrapper));

  const auto& outputs = node_unit.Outputs();

  TensorInfo output_info = {};
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetTensorInfo(outputs[0], output_info));

  // HTP backend can only handle rank 3 inputs/outputs if the batch size is 1. If the batch size is not 1,
  // QNN EP must reshape the input and output to (N, 1, W, C) and process the InstanceNorm as rank 4.
  if (!IsNpuBackend(qnn_model_wrapper.GetQnnBackendType()) ||
      output_info.shape.size() != 3 || output_info.shape[0] == 1) {
    return ProcessOutputs(qnn_model_wrapper, node_unit,
                          std::move(input_names),
                          std::move(param_tensor_names),
                          logger, do_op_validation, GetQnnOpType(node_unit.OpType()));
  }

  //
  // The output is meant to be rank 3 with batch size != 1. Must create a QNN InstanceNorm op with a rank 4 output
  // that is then reshaped to rank 3 again.
  //

  const std::string& orig_output_name = outputs[0].node_arg.Name();
  std::string op_output_name = orig_output_name + "_ort_qnn_ep_reshape";

  std::vector<uint32_t> op_output_shape = {
      output_info.shape[0],  // N
      1,                     // H == 1
      output_info.shape[1],  // W
      output_info.shape[2],  // C
  };

  QnnTensorWrapper output_tensorwrapper(op_output_name, QNN_TENSOR_TYPE_NATIVE, output_info.qnn_data_type,
                                        output_info.quant_param, std::vector<uint32_t>(op_output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(GetNodeName(node_unit),
                                                    QNN_OP_PACKAGE_NAME_QTI_AISW,
                                                    GetQnnOpType(node_unit.OpType()),
                                                    std::move(input_names),
                                                    {op_output_name},
                                                    std::move(param_tensor_names)),
                    "Failed to add node.");

  const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(orig_output_name);

  // Add Reshape to convert QNN InstanceNorm output back to rank 3 (as expected by the rest of the ONNX graph).
  ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(op_output_name,
                                                       orig_output_name,
                                                       op_output_shape,
                                                       output_info.shape,
                                                       output_info.qnn_data_type,
                                                       output_info.quant_param,
                                                       do_op_validation,
                                                       false,
                                                       is_graph_output));
  return Status::OK();
}

void CreateInstanceNormOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<InstanceNormOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
