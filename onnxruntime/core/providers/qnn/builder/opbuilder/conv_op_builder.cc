// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/common/safeint.h"
#include "core/providers/qnn/builder/qnn_utils.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace qnn {

class ConvOpBuilder : public BaseOpBuilder {
 public:
  ConvOpBuilder() : BaseOpBuilder("ConvOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ConvOpBuilder);

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

 private:
  Status GetInputChannelNumber(QnnModelWrapper& qnn_model_wrapper,
                               const NodeUnit& node_unit,
                               uint32_t& input_channel_number) const;
  Status AddDefaultBias(QnnModelWrapper& qnn_model_wrapper,
                        const uint32_t weight_m,
                        const std::string& node_name,
                        const bool is_quantized_model,
                        const logging::Logger& logger,
                        std::vector<std::string>& input_names) const;
};

// Conv, ConvTranspose ops are sensitive with data layout, no special validation so far
// The nodes from 1st call of GetCapability do not get layout transformer applied, it's still NCHW
// The nodes from 2nd call of GetCapability get layout transformer applied, it's NHWC
// Need to do op validation in 1st call of GetCapability
// TODO: Check if node domain == kMSInternalNHWCDomain to determine if the layout has been transformed.
Status ConvOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger,
                                    bool is_quantized_model) const {
  if (node_unit.Domain() == kMSInternalNHWCDomain) {  // Use QNN validation API if layout is NHWC.
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, is_quantized_model, true);
  }

  const auto& input_0 = node_unit.Inputs()[0];
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_0.node_arg, input_shape), "Cannot get shape");
  if (input_shape.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN Conv only support 2D!");
  }

  ONNX_NAMESPACE::DataType input_data_type = input_0.node_arg.Type();
  ORT_RETURN_IF(!is_quantized_model && input_data_type != ONNX_NAMESPACE::Utils::DataTypeUtils::ToType("float"),
                "QNN EP: Data type ", input_data_type->c_str(),
                " is not supported for Conv operator in CPU backend.");

  NodeAttrHelper node_helper(node_unit);
  auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
  ORT_RETURN_IF(auto_pad != "NOTSET" && auto_pad != "SAME_LOWER" && auto_pad != "SAME_UPPER",
                "QNN Conv operators do not support 'auto_pad' value: ", auto_pad.c_str());

  return Status::OK();
}

Status ConvOpBuilder::GetInputChannelNumber(QnnModelWrapper& qnn_model_wrapper,
                                            const NodeUnit& node_unit,
                                            uint32_t& input_channel_number) const {
  auto input_0 = node_unit.Inputs()[0];
  input_channel_number = 0;
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_0.node_arg, input_shape), "Cannot get shape");
  // Conv input 0 is NHWC layout now, get the channel data from input_shape[3]
  input_channel_number = input_shape.at(3);

  return Status::OK();
}

// TODO: bias is not required in QNN, but it failed for some case if remove this. That case has Weight as dynamic input
// e.g. the Conv node test in Onnx repo like test_conv_with_autopad_same. Could be QNN issue, Still need to dig out more
Status ConvOpBuilder::AddDefaultBias(QnnModelWrapper& qnn_model_wrapper,
                                     const uint32_t weight_m,
                                     const std::string& node_name,
                                     const bool is_quantized_model,
                                     const logging::Logger& logger,
                                     std::vector<std::string>& input_names) const {
  std::vector<uint32_t> bias_shape(1, weight_m);
  std::string bias_name = node_name + "_bias";

  Qnn_DataType_t qnn_data_type = is_quantized_model ? QNN_DATATYPE_SFIXED_POINT_32 : QNN_DATATYPE_FLOAT_32;
  uint32_t data_size = weight_m * static_cast<uint32_t>(utils::GetElementSizeByType(qnn_data_type));
  std::vector<uint8_t> default_bias_data(data_size, 0);

  LOGS(logger, VERBOSE) << "Add default bias: " << bias_name;
  Qnn_TensorType_t tensor_type = QNN_TENSOR_TYPE_STATIC;
  Qnn_QuantizeParams_t quantize_param = QNN_QUANTIZE_PARAMS_INIT;
  InitializeQuantizeParam(quantize_param, is_quantized_model);
  QnnTensorWrapper bias_tensorwrapper(bias_name, tensor_type, qnn_data_type, quantize_param,
                                      std::move(bias_shape), std::move(default_bias_data));

  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(bias_tensorwrapper)), "Failed to add tensor.");
  input_names.push_back(bias_name);
  return Status::OK();
}

Status ConvOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger,
                                    bool is_quantized_model,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  Qnn_QuantizeParams_t quantize_param = QNN_QUANTIZE_PARAMS_INIT;
  InitializeQuantizeParam(quantize_param, is_quantized_model);
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  uint32_t weight_m = 0;
  auto inputs = node_unit.Inputs();

  for (size_t input_i = 0; input_i < inputs.size(); ++input_i) {
    const auto& input_name = inputs[input_i].node_arg.Name();

    const auto* type_proto = inputs[input_i].node_arg.TypeAsProto();
    ORT_RETURN_IF_ERROR(GetQnnDataType(is_quantized_model, type_proto, qnn_data_type));

    std::vector<uint32_t> input_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[input_i].node_arg, input_shape), "Cannot get shape");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.ProcessQuantizationParameter(inputs[input_i].quant_param,
                                                                     quantize_param.scaleOffsetEncoding.scale,
                                                                     quantize_param.scaleOffsetEncoding.offset),
                      "Cannot get quantization parameter");

    std::vector<uint8_t> unpacked_tensor;
    bool is_initializer_input = qnn_model_wrapper.IsInitializerInput(input_name);
    if (is_initializer_input) {
      const auto& input_tensor = qnn_model_wrapper.GetInitializerTensors().at(input_name);
      if (1 == input_i) {  // qnn Conv weight requires HWCN
        if (node_unit.OpType() == "Conv") {
          ORT_RETURN_IF_ERROR(TransposeFromNchwToHwcn(qnn_model_wrapper, *input_tensor, unpacked_tensor));
        } else if (node_unit.OpType() == "ConvTranspose") {
          ORT_RETURN_IF_ERROR(TransposeFromCnhwToHwcn(qnn_model_wrapper, *input_tensor, unpacked_tensor));
        } else {
          ORT_THROW("Unexpected operator %s", node_unit.OpType());
        }
      } else {
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_tensor, unpacked_tensor));
      }
    }

    std::string input_tensor_name(input_name);
    std::vector<uint32_t> new_input_shape;
    if (1 == input_i) {
      new_input_shape.resize(input_shape.size());
      // Change shape to HWCN, it could be initializer or normal input
      if (node_unit.OpType() == "Conv") {
        ORT_RETURN_IF_ERROR(NchwShapeToHwcn(input_shape, new_input_shape));
      } else if (node_unit.OpType() == "ConvTranspose") {
        ORT_RETURN_IF_ERROR(CnhwShapeToHwcn(input_shape, new_input_shape));
      } else {
        ORT_THROW("Unexpected operator %s", node_unit.OpType());
      }
      weight_m = new_input_shape.at(3);
      // Add Transpose node NCHW->HWCN after the graph input (exclude initializer) for Conv weight input
      if (!is_initializer_input) {
        bool is_graph_input = qnn_model_wrapper.IsGraphInput(input_name);
        std::string transpose_output_name = input_name + "_trans";
        LOGS(logger, VERBOSE) << "Add HWCN Transpose node after input: " << input_name;
        if (node_unit.OpType() == "Conv") {
          ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddNchwToHwcnTranspose(node_unit.Index(),
                                                                       input_name,
                                                                       transpose_output_name,
                                                                       input_shape,
                                                                       new_input_shape,
                                                                       qnn_data_type,
                                                                       quantize_param,
                                                                       is_graph_input));
        } else if (node_unit.OpType() == "ConvTranspose") {
          ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddCnhwToHwcnTranspose(node_unit.Index(),
                                                                       input_name,
                                                                       transpose_output_name,
                                                                       input_shape,
                                                                       new_input_shape,
                                                                       qnn_data_type,
                                                                       quantize_param,
                                                                       is_graph_input));
        } else {
          ORT_THROW("Unexpected operator %s", node_unit.OpType());
        }
        input_tensor_name = transpose_output_name;
      }
      input_shape = new_input_shape;
    }
    input_names.push_back(input_tensor_name);

    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name)) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_name;
      continue;
    }

    Qnn_TensorType_t tensor_type = GetInputTensorType(qnn_model_wrapper, input_name);

    QnnTensorWrapper input_tensorwrapper(input_name, tensor_type, qnn_data_type, quantize_param,
                                         std::move(input_shape), std::move(unpacked_tensor));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  }

  if (inputs.size() < 3) {  // Add default bais
    ORT_RETURN_IF_ERROR(AddDefaultBias(qnn_model_wrapper, weight_m, GetNodeName(node_unit),
                                       is_quantized_model, logger, input_names));
  }

  return Status::OK();
}

Status ConvOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool is_quantized_model,
                                                  bool do_op_validation) const {
  ORT_UNUSED_PARAMETER(do_op_validation);
  NodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_tensor_names;
  std::vector<uint32_t> output_padding;
  uint32_t output_padding_0 = 0;
  uint32_t output_padding_1 = 0;
  // Conv attribute dilations
  auto dilation_values = node_helper.Get("dilations", std::vector<int32_t>{1, 1});
  std::vector<uint32_t> dilations;
  std::transform(dilation_values.cbegin(), dilation_values.cend(), std::back_inserter(dilations),
                 [](int32_t item) { return SafeInt<uint32_t>(item); });
  // keep a copy for later user since it will be invalid after move
  uint32_t dilations_0 = dilations[0];
  uint32_t dilations_1 = dilations[1];
  uint32_t dialitions_size = static_cast<uint32_t>(dilations.size());
  std::vector<uint32_t> dialitions_dim;
  dialitions_dim.push_back(dialitions_size);
  if (node_unit.OpType() == "Conv") {
    QnnParamWrapper dilation_paramwrapper(node_unit.Index(), node_unit.Name(), qnn_def::dilation,
                                          std::move(dialitions_dim), std::move(dilations));
    param_tensor_names.push_back(dilation_paramwrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(dilation_paramwrapper));
  } else if (node_unit.OpType() == "ConvTranspose") {
    // Add output_padding param
    auto output_padding_values = node_helper.Get("output_padding", std::vector<int32_t>{0, 0});
    std::transform(output_padding_values.cbegin(), output_padding_values.cend(), std::back_inserter(output_padding),
                   [](int32_t item) { return SafeInt<uint32_t>(item); });
    // keep a copy for later user since it will be invalid after move
    output_padding_0 = output_padding[0];
    output_padding_1 = output_padding[1];
    std::vector<uint32_t> output_padding_dim{static_cast<uint32_t>(output_padding.size())};
    QnnParamWrapper output_padding_paramwrapper(node_unit.Index(), node_unit.Name(), qnn_def::output_padding,
                                                std::move(output_padding_dim), std::move(output_padding));
    param_tensor_names.push_back(output_padding_paramwrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(output_padding_paramwrapper));
  } else {
    ORT_THROW("Unexpected operator %s", node_unit.OpType());
  }
  // Conv/ConvTranspose output
  const auto& outputs = node_unit.Outputs();
  const auto& output_name = outputs[0].node_arg.Name();

  std::vector<uint32_t> output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(outputs[0].node_arg, output_shape), "Cannot get shape");

  // Conv attribute strides
  auto stride_values = node_helper.Get("strides", std::vector<int32_t>{1, 1});
  std::vector<uint32_t> strides;
  std::transform(stride_values.cbegin(), stride_values.cend(), std::back_inserter(strides),
                 [](int32_t item) { return SafeInt<uint32_t>(item); });
  uint32_t strides_size = static_cast<uint32_t>(strides.size());
  std::vector<uint32_t> strides_dim{strides_size};
  QnnParamWrapper stride_amount_paramwrapper(node_unit.Index(), node_unit.Name(), qnn_def::stride,
                                             std::move(strides_dim), std::move(strides));
  param_tensor_names.push_back(stride_amount_paramwrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(stride_amount_paramwrapper));

  std::vector<int32_t> pad_values = {0, 0, 0, 0};
  auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
  ORT_RETURN_IF(auto_pad != "NOTSET" && auto_pad != "SAME_LOWER" && auto_pad != "SAME_UPPER",
                "QNN Conv operators do not support 'auto_pad' value: ", auto_pad.c_str());

  if (auto_pad.compare("NOTSET") != 0) {
    auto input_0 = node_unit.Inputs()[0];
    auto input_1 = node_unit.Inputs()[1];
    std::vector<uint32_t> input_0_shape;
    std::vector<uint32_t> input_1_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_0.node_arg, input_0_shape), "Cannot get shape");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_1.node_arg, input_1_shape), "Cannot get shape");
    int32_t total_padding[2];
    if (node_unit.OpType() == "ConvTranspose") {
      total_padding[0] = stride_values[0] * (input_0_shape[1] - 1) + output_padding_0 + (input_1_shape[2] - 1) * dilations_0 + 1 - output_shape[1];
      total_padding[1] = stride_values[1] * (input_0_shape[2] - 1) + output_padding_1 + (input_1_shape[3] - 1) * dilations_1 + 1 - output_shape[2];
    } else {
      total_padding[0] = output_shape[1] * stride_values[0] - input_0_shape[1] + 1;
      total_padding[1] = output_shape[1] * stride_values[0] - input_0_shape[2] + 1;
    }
    if (auto_pad.compare("SAME_UPPER")) {
      pad_values[0] = total_padding[0] / 2;
      pad_values[1] = total_padding[1] / 2;
      pad_values[2] = total_padding[0] - pad_values[0];
      pad_values[3] = total_padding[1] - pad_values[1];
    } else if (auto_pad.compare("SAME_LOWER")) {
      pad_values[2] = total_padding[0] / 2;
      pad_values[3] = total_padding[1] / 2;
      pad_values[0] = total_padding[0] - pad_values[2];
      pad_values[1] = total_padding[1] - pad_values[3];
    }
  } else {
    // Conv/ConvTranspose attribute pads
    pad_values = node_helper.Get("pads", pad_values);
  }
  ReArranagePads(pad_values);
  std::vector<uint32_t> pads;
  std::transform(pad_values.cbegin(), pad_values.cend(), std::back_inserter(pads),
                 [](int32_t item) { return SafeInt<uint32_t>(item); });
  // Qnn Conv2d must use dims {2, 2}
  std::vector<uint32_t> pad_dims{2, 2};
  QnnParamWrapper pad_amount_paramwrapper(node_unit.Index(), node_unit.Name(), qnn_def::pad_amount,
                                          std::move(pad_dims), std::move(pads));
  param_tensor_names.push_back(pad_amount_paramwrapper.GetParamTensorName());
  qnn_model_wrapper.AddParamWrapper(std::move(pad_amount_paramwrapper));

  Qnn_QuantizeParams_t output_quantize_param = QNN_QUANTIZE_PARAMS_INIT;
  InitializeQuantizeParam(output_quantize_param, is_quantized_model);

  const auto* type_proto = outputs[0].node_arg.TypeAsProto();
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  ORT_RETURN_IF_ERROR(GetQnnDataType(is_quantized_model, type_proto, qnn_data_type));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.ProcessQuantizationParameter(outputs[0].quant_param,
                                                                   output_quantize_param.scaleOffsetEncoding.scale,
                                                                   output_quantize_param.scaleOffsetEncoding.offset),
                    "Cannot get quantization parameter");

  const uint32_t group = SafeInt<uint32_t>(node_helper.Get("group", static_cast<int64_t>(1)));
  uint32_t num_output = output_shape[3];
  uint32_t num_input_channel = 0;
  ORT_RETURN_IF_ERROR(GetInputChannelNumber(qnn_model_wrapper, node_unit, num_input_channel));
  LOGS(logger, VERBOSE) << (node_unit.OpType() == "Conv" ? "Conv:" : "ConvTranspose:")
                        << " num_output: " << num_output << ", num_input_channel: " << num_input_channel << ", group: " << group;
  const static std::string depthwise_conv2d = "DepthWiseConv2d";
  bool is_depthwise_conv2d = false;
  if ((node_unit.OpType() == "Conv") && (num_input_channel == num_output) && (group == num_output)) {
    is_depthwise_conv2d = true;
  } else {  // DepthWiseConv2d does not need group
    // Conv attribute group
    Qnn_Scalar_t group_qnn_scalar = QNN_SCALAR_INIT;
    group_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
    group_qnn_scalar.uint32Value = group;
    QnnParamWrapper group_paramwrapper(node_unit.Index(), node_unit.Name(), qnn_def::group, group_qnn_scalar);
    param_tensor_names.push_back(group_paramwrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(group_paramwrapper));
  }
  const std::string& output_node_type = is_depthwise_conv2d ? depthwise_conv2d : GetQnnOpType(node_unit.OpType());

  bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
  Qnn_TensorType_t tensor_type = is_graph_output ? QNN_TENSOR_TYPE_APP_READ : QNN_TENSOR_TYPE_NATIVE;
  QnnTensorWrapper output_tensorwrapper(output_name, tensor_type, qnn_data_type, output_quantize_param,
                                        std::move(output_shape));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");

  ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(GetNodeName(node_unit),
                                                    qnn_def::package_name,
                                                    output_node_type,
                                                    std::move(input_names),
                                                    {output_name},
                                                    std::move(param_tensor_names)),
                    "Failed to add node.");

  return Status::OK();
}

void CreateConvOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ConvOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
