// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/common/safeint.h"

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"

namespace onnxruntime {
namespace qnn {
class ResizeOpBuilder : public BaseOpBuilder {
 public:
  ResizeOpBuilder() : BaseOpBuilder("ResizeOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ResizeOpBuilder);

  Status IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       bool is_quantized_model) const override final ORT_MUST_USE_RESULT;

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       bool is_quantized_model,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool is_quantized_model,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;
};

// Resize ops are sensitive with data layout, no special validation so far
// The nodes from 1st call of GetCapability do not get layout transformer applied, it's still NCHW
// The nodes from 2nd call of GetCapability get layout transformer applied, it's NHWC
// Need to do op validation in 1st call of GetCapability
Status ResizeOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger,
                                      bool is_quantized_model) const {
  ORT_UNUSED_PARAMETER(logger);
  ORT_UNUSED_PARAMETER(is_quantized_model);
  NodeAttrHelper node_helper(node_unit);
  const std::string& resize_mode = node_helper.Get("mode", "nearest");
  ORT_RETURN_IF("cubic" == resize_mode, "Resize donesn't support cubic mode!");

  const std::string& coordinate_mode = node_helper.Get("coordinate_transformation_mode", "half_pixel");
  if ("pytorch_half_pixel" == coordinate_mode ||
      "tf_crop_and_resize" == coordinate_mode ||
      "tf_half_pixel_for_nn" == coordinate_mode) {
    std::string msg = coordinate_mode + " mode not supported, QNN Resize only support align_corners and half_pixel";
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, msg);
  }

  if ("nearest" == resize_mode) {
    const std::string& nearest_mode = node_helper.Get("nearest_mode", "round_prefer_floor");
    ORT_RETURN_IF_NOT("floor" == nearest_mode, "QNN Resize only support nearest_mode: floor!");
  }

  auto& input_0 = node_unit.Inputs()[0];
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_0.node_arg, input_shape), "Cannot get shape");
  if (input_shape.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN Resize only support 2D!");
  }

  ONNX_NAMESPACE::DataType input_data_type = input_0.node_arg.Type();
  if (!is_quantized_model && input_data_type != ONNX_NAMESPACE::Utils::DataTypeUtils::ToType("float")) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Data type " + *input_data_type + " is not supported in CPU backend.");
  }

  return Status::OK();
}

Status ResizeOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                      const NodeUnit& node_unit,
                                      const logging::Logger& logger,
                                      bool is_quantized_model,
                                      std::vector<std::string>& input_names,
                                      bool do_op_validation) const {
  // Only cares about the 1st input
  ORT_UNUSED_PARAMETER(do_op_validation);
  Qnn_QuantizeParams_t quantize_param;
  InitializeQuantizeParam(quantize_param, is_quantized_model);
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;

  const auto& inputs = node_unit.Inputs();
  const auto& input_name = inputs[0].node_arg.Name();

  if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
    LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_name;
    input_names.push_back(input_name);
    return Status::OK();
  }

  const auto* type_proto = inputs[0].node_arg.TypeAsProto();
  ORT_RETURN_IF_ERROR(GetQnnDataType(is_quantized_model, type_proto, qnn_data_type));

  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input_shape), "Cannot get shape");
  ORT_RETURN_IF_NOT(qnn_model_wrapper.ProcessQuantizationParameter(inputs[0].quant_param,
                                                                   quantize_param.scaleOffsetEncoding.scale,
                                                                   quantize_param.scaleOffsetEncoding.offset),
                    "Cannot get quantization parameter");

  std::vector<uint8_t> unpacked_tensor;
  bool is_initializer_input = qnn_model_wrapper.IsInitializerInput(input_name);
  if (is_initializer_input) {
    const auto& input_tensor = qnn_model_wrapper.GetInitializerTensors().at(input_name);
    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(*input_tensor, unpacked_tensor));
  }

  input_names.push_back(input_name);
  Qnn_TensorType_t tensor_type = GetInputTensorType(qnn_model_wrapper, input_name);

  QnnTensorWrapper input_tensorwrapper(input_name, tensor_type, qnn_data_type, quantize_param,
                                       std::move(input_shape), std::move(unpacked_tensor));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");

  return Status::OK();
}

Status ResizeOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                    const NodeUnit& node_unit,
                                                    std::vector<std::string>&& input_names,
                                                    const logging::Logger& logger,
                                                    bool is_quantized_model,
                                                    bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(logger);
  NodeAttrHelper node_helper(node_unit);
  const std::string& resize_mode = node_helper.Get("mode", "nearest");
  std::string qnn_node_type = "ResizeNearestNeighbor";
  if ("linear" == resize_mode) {
    qnn_node_type = "ResizeBilinear";
  }

  const std::string& coordinate_mode = node_helper.Get("coordinate_transformation_mode", "half_pixel");

  Qnn_Scalar_t qnn_align_corners = QNN_SCALAR_INIT;
  qnn_align_corners.dataType = QNN_DATATYPE_BOOL_8;
  qnn_align_corners.bool8Value = static_cast<uint8_t>(0);

  Qnn_Scalar_t qnn_half_pixel = QNN_SCALAR_INIT;
  qnn_half_pixel.dataType = QNN_DATATYPE_BOOL_8;
  qnn_half_pixel.bool8Value = static_cast<uint8_t>(0);

  if ("align_corners" == coordinate_mode) {
    qnn_align_corners.bool8Value = static_cast<uint8_t>(1);
  } else if ("half_pixel" == coordinate_mode) {
    qnn_half_pixel.bool8Value = static_cast<uint8_t>(1);
  }
  QnnParamWrapper qnn_align_corners_param(node_unit.Index(), node_unit.Name(),
                                          qnn_def::align_corners, qnn_align_corners);
  QnnParamWrapper qnn_half_pixel_param(node_unit.Index(), node_unit.Name(),
                                       qnn_def::half_pixel_centers, qnn_half_pixel);

  std::vector<std::string> param_tensor_names;
  param_tensor_names.push_back(qnn_align_corners_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(qnn_align_corners_param));
  param_tensor_names.push_back(qnn_half_pixel_param.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(qnn_half_pixel_param));

  const auto& resize_output = node_unit.Outputs()[0];

  const auto& output_name = resize_output.node_arg.Name();

  Qnn_QuantizeParams_t quantize_param = QNN_QUANTIZE_PARAMS_INIT;
  InitializeQuantizeParam(quantize_param, is_quantized_model);

  const auto* type_proto = resize_output.node_arg.TypeAsProto();
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  ORT_RETURN_IF_ERROR(GetQnnDataType(is_quantized_model, type_proto, qnn_data_type));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.ProcessQuantizationParameter(resize_output.quant_param,
                                                                   quantize_param.scaleOffsetEncoding.scale,
                                                                   quantize_param.scaleOffsetEncoding.offset),
                    "Cannot get quantization parameter");
  std::vector<uint32_t> output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(resize_output.node_arg, output_shape),
                    "Cannot get shape");

  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
  Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper output_tensorwrapper(output_name,
                                        tensor_type,
                                        qnn_data_type,
                                        quantize_param,
                                        std::move(output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(GetNodeName(node_unit),
                                                    qnn_def::package_name,
                                                    qnn_node_type,
                                                    std::move(input_names),
                                                    {output_name},
                                                    std::move(param_tensor_names),
                                                    do_op_validation),
                    "Failed to add node.");

  return Status::OK();
}

void CreateResizeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ResizeOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
