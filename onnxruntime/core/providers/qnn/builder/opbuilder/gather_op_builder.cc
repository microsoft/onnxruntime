// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace qnn {

// Operator which only need to hanle node inputs & outputs, no attributes or no need to handle attributes
class GatherOpBuilder : public BaseOpBuilder {
 public:
  GatherOpBuilder() : BaseOpBuilder("GatherOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GatherOpBuilder);

 protected:
  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     const std::vector<std::string>& input_names,
                                     const logging::Logger& logger,
                                     bool is_quantized_model,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

Status GatherOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const NodeUnit& node_unit,
                                                    const std::vector<std::string>& input_names,
                                                    const logging::Logger& logger,
                                                    bool is_quantized_model,
                                                    bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  std::vector<QnnParamWrapper> node_params;
  int32_t axis_value = 0;
  ORT_RETURN_IF_ERROR(ProcessAxisAttribute(qnn_model_wrapper, node_unit, node_params, axis_value));

  // if indicies is scalar shape, then need to add Reshape node
  ORT_ENFORCE(input_names.size() == 2, "Gather should has 2 inputs at least!");
  Qnn_Tensor_t input;
  qnn_model_wrapper.GetQnnTensor(input_names[0], input);
  Qnn_Tensor_t indices_input;
  qnn_model_wrapper.GetQnnTensor(input_names[1], indices_input);

  // Calcualte the output shape
  std::vector<uint32_t> qnn_output_shape;
  qnn_output_shape.reserve(input.rank - 1 + indices_input.rank);

  const auto& gather_indices = node_unit.Inputs()[1];
  std::vector<uint32_t> indices_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(gather_indices.node_arg, indices_shape),
                    "Cannot get shape");

  // replace the dimension for p.axis with the shape from the indices
  for (int32_t i = 0; i < axis_value; ++i) {
    qnn_output_shape.push_back(input.currentDimensions[i]);
  }

  for (size_t i = 0; i < indices_input.rank; ++i) {
    qnn_output_shape.push_back(indices_input.currentDimensions[i]);
  }

  for (int32_t i = axis_value + 1; i < static_cast<int32_t>(input.rank); ++i) {
    qnn_output_shape.push_back(input.currentDimensions[i]);
  }

  const auto& gather_output = node_unit.Outputs()[0];
  const auto& output_name = gather_output.node_arg.Name();

  Qnn_QuantizeParams_t quantize_param = QNN_QUANTIZE_PARAMS_INIT;
  InitializeQuantizeParam(quantize_param, is_quantized_model);

  const auto* type_proto = gather_output.node_arg.TypeAsProto();
  int32_t onnx_data_type;
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  ORT_RETURN_IF_ERROR(GetQnnDataType(is_quantized_model, type_proto, onnx_data_type, qnn_data_type));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.ProcessQuantizationParameter(gather_output.quant_param,
                                                                    quantize_param.scaleOffsetEncoding.scale,
                                                                    quantize_param.scaleOffsetEncoding.offset),
                    "Cannot get quantization parameter");
  std::vector<uint32_t> target_output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(gather_output.node_arg, target_output_shape),
                    "Cannot get shape");

  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
  bool reshape_required = (qnn_output_shape.size() != target_output_shape.size());
  std::string name = output_name + (reshape_required ? "_reshape" : "");
  Qnn_TensorType_t tensor_type = (!reshape_required && is_graph_output) ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper gather_output_wrapper(name, tensor_type, 0, qnn_data_type, quantize_param,
                                         std::move(qnn_output_shape));
  std::vector<QnnTensorWrapper> output_tensors;
  output_tensors.emplace_back(std::move(gather_output_wrapper));

  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddNode(GetNodeName(node_unit),            // Node Name
                                               qnn_def::package_name,             // Package Name
                                               GetQnnOpType(node_unit.OpType()),  //
                                               std::move(node_params),            // Qnn Node Type
                                               input_names,                       // Input Tensor Names
                                               std::move(output_tensors),         // Output Tensors
                                               do_op_validation),
                    "Failed to add node.");

  if (reshape_required) {
    std::vector<std::string> reshape_input_names{name};

    // Add Reshape Node after Gather.
    Qnn_TensorType_t reshape_tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
    Qnn_TensorDataFormat_t reshape_data_format = 0;
    QnnTensorWrapper reshape_output(output_name, reshape_tensor_type, reshape_data_format, qnn_data_type, quantize_param,
                                    std::move(target_output_shape));
    std::vector<QnnTensorWrapper> reshape_output_tensors;
    reshape_output_tensors.emplace_back(std::move(reshape_output));
    const static std::string qnn_node_type = "Reshape";
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddNode(output_name,                        // Node Name
                                                 qnn_def::package_name,              // Package Name
                                                 qnn_node_type,                      // Qnn Node Type
                                                 {},                                 // Node Params
                                                 reshape_input_names,                // Input Tensor Names
                                                 std::move(reshape_output_tensors),  // Output Tensors
                                                 do_op_validation),
                      "Failed to add node.");
  }

  return Status::OK();
}

void CreateGatherOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GatherOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
