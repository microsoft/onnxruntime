// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/framework/tensorprotoutils.h"
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
  Status ProcessConv1DInputs(QnnModelWrapper& qnn_model_wrapper,
                             const NodeUnit& node_unit,
                             const logging::Logger& logger,
                             bool is_quantized_model,
                             std::vector<std::string>& input_names,
                             bool do_op_validation) const ORT_MUST_USE_RESULT;
  Status ProcessConv2DInputs(QnnModelWrapper& qnn_model_wrapper,
                             const NodeUnit& node_unit,
                             const logging::Logger& logger,
                             bool is_quantized_model,
                             std::vector<std::string>& input_names,
                             bool do_op_validation) const ORT_MUST_USE_RESULT;
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
};

// Conv, ConvTranspose ops are sensitive with data layout, no special validation so far
// The nodes from 1st call of GetCapability do not get layout transformer applied, it's still NCHW
// The nodes from 2nd call of GetCapability get layout transformer applied, it's NHWC
// Need to do op validation in 1st call of GetCapability
Status ConvOpBuilder::IsOpSupported(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger,
                                    bool is_quantized_model) const {
  if (node_unit.Domain() == kMSInternalNHWCDomain) {  // Use QNN validation API if layout is NHWC.
    return AddToModelBuilder(qnn_model_wrapper, node_unit, logger, is_quantized_model, true);
  }

  const auto& inputs = node_unit.Inputs();
  ORT_RETURN_IF(inputs.size() < 2, "QNN Conv and ConvTranspose must have at least 2 inputs.");

  const auto& input_0 = inputs[0];
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_0.node_arg, input_shape), "Cannot get shape");
  if (input_shape.size() != 4 && input_shape.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN Conv only supports 2D (rank 4) or 1D (rank 3) inputs.");
  }

  ONNX_NAMESPACE::DataType input_data_type = input_0.node_arg.Type();
  ORT_RETURN_IF(!is_quantized_model && input_data_type != ONNX_NAMESPACE::Utils::DataTypeUtils::ToType("float"),
                "QNN EP: Data type ", input_data_type->c_str(),
                " is not supported for Conv operator in CPU backend.");

  NodeAttrHelper node_helper(node_unit);
  auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
  ORT_RETURN_IF(auto_pad != "NOTSET" && auto_pad != "SAME_LOWER" && auto_pad != "SAME_UPPER",
                "QNN Conv operators do not support 'auto_pad' value: ", auto_pad.c_str());

  // QNN's TransposeConv2d only supports default dilation values of 1.
  if (node_unit.OpType() == "ConvTranspose") {
    constexpr int32_t default_dilation = 1;
    auto dilations = node_helper.Get("dilations", std::vector<int32_t>{default_dilation, default_dilation});

    for (auto dilation : dilations) {
      ORT_RETURN_IF(dilation != default_dilation,
                    "QNN EP: QNN's TransposeConv2d operator only supports default dilation values of 1.");
    }
  }

  return Status::OK();
}

Status ConvOpBuilder::GetInputChannelNumber(QnnModelWrapper& qnn_model_wrapper,
                                            const NodeUnit& node_unit,
                                            uint32_t& input_channel_number) const {
  const auto& input_0 = node_unit.Inputs()[0];
  input_channel_number = 0;
  std::vector<uint32_t> input_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_0.node_arg, input_shape), "Cannot get shape");
  // Conv input 0 is NHWC layout now, get the channel data from the last dim.
  input_channel_number = input_shape.back();

  return Status::OK();
}

Status ConvOpBuilder::ProcessConv2DInputs(QnnModelWrapper& qnn_model_wrapper,
                                          const NodeUnit& node_unit,
                                          const logging::Logger& logger,
                                          bool is_quantized_model,
                                          std::vector<std::string>& input_names,
                                          bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const size_t num_inputs = inputs.size();
  const auto& onnx_op_type = node_unit.OpType();

  assert(num_inputs >= 2);  // Checked by IsOpSupported.

  //
  // Input 0
  //
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, is_quantized_model, input_names));

  //
  // Input 1: weight
  //
  uint32_t weight_m = 0;
  {
    const std::string& input1_name = inputs[1].node_arg.Name();
    OnnxInputInfo input_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(inputs[1], is_quantized_model, input_info));

    std::string actual_name = input_info.is_initializer ? input1_name : input1_name + "_trans";
    input_names.push_back(actual_name);

    std::vector<uint32_t> actual_shape;
    actual_shape.resize(input_info.shape.size());

    // Change shape to HWCN, it could be initializer or normal input
    if (node_unit.OpType() == "Conv") {
      ORT_RETURN_IF_ERROR(NchwShapeToHwcn(input_info.shape, actual_shape));
    } else if (node_unit.OpType() == "ConvTranspose") {
      ORT_RETURN_IF_ERROR(CnhwShapeToHwcn(input_info.shape, actual_shape));
    } else {
      ORT_THROW("Unexpected operator %s", onnx_op_type);
    }
    weight_m = actual_shape.at(3);

    std::vector<uint8_t> unpacked_tensor;
    if (input_info.is_initializer) {
      // Get transposed initializer bytes.
      if (node_unit.OpType() == "Conv") {
        ORT_RETURN_IF_ERROR(TransposeFromNchwToHwcn(qnn_model_wrapper, *input_info.initializer_tensor, unpacked_tensor));
      } else if (node_unit.OpType() == "ConvTranspose") {
        ORT_RETURN_IF_ERROR(TransposeFromCnhwToHwcn(qnn_model_wrapper, *input_info.initializer_tensor, unpacked_tensor));
      } else {
        ORT_THROW("Unexpected operator %s", node_unit.OpType());
      }
    } else {
      // Add transpose node above weight input.
      bool is_graph_input = qnn_model_wrapper.IsGraphInput(input1_name);
      LOGS(logger, VERBOSE) << "Add HWCN Transpose node after input: " << input1_name;
      if (node_unit.OpType() == "Conv") {
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddNchwToHwcnTranspose(node_unit.Index(),
                                                                     input1_name,
                                                                     actual_name,
                                                                     input_info.shape,
                                                                     actual_shape,
                                                                     input_info.qnn_data_type,
                                                                     input_info.quant_param,
                                                                     do_op_validation,
                                                                     is_graph_input));
      } else if (node_unit.OpType() == "ConvTranspose") {
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddCnhwToHwcnTranspose(node_unit.Index(),
                                                                     input1_name,
                                                                     actual_name,
                                                                     input_info.shape,
                                                                     actual_shape,
                                                                     input_info.qnn_data_type,
                                                                     input_info.quant_param,
                                                                     do_op_validation,
                                                                     is_graph_input));
      } else {
        ORT_THROW("Unexpected operator %s", node_unit.OpType());
      }
    }

    Qnn_TensorType_t tensor_type = GetInputTensorType(qnn_model_wrapper, actual_name);
    QnnTensorWrapper input_tensorwrapper(actual_name, tensor_type, input_info.qnn_data_type, input_info.quant_param,
                                         std::move(actual_shape), std::move(unpacked_tensor));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  }

  //
  // Input 2: bias
  //
  if (inputs.size() == 3) {
    ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[2], logger, is_quantized_model, input_names));
  }

  return Status::OK();
}

Status ConvOpBuilder::ProcessConv1DInputs(QnnModelWrapper& qnn_model_wrapper,
                                          const NodeUnit& node_unit,
                                          const logging::Logger& logger,
                                          bool is_quantized_model,
                                          std::vector<std::string>& input_names,
                                          bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const size_t num_inputs = inputs.size();
  const auto& onnx_op_type = node_unit.OpType();

  assert(num_inputs >= 2);  // Checked by IsOpSupported.

  //
  // Input 0
  //

  {
    const std::string& input0_name = inputs[0].node_arg.Name();
    OnnxInputInfo input0_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(inputs[0], is_quantized_model, input0_info));

    const std::string conv_input0_name = input0_info.is_initializer ? input0_name
                                                                    : input0_name + "_reshape_as_conv2d";
    input_names.push_back(conv_input0_name);

    if (!qnn_model_wrapper.IsQnnTensorWrapperExist(conv_input0_name)) {
      std::vector<uint8_t> unpacked_tensor;
      if (input0_info.is_initializer) {
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input0_info.initializer_tensor, unpacked_tensor));
      }

      std::vector<uint32_t> shape = {
          input0_info.shape[0],  // N
          1,                     // Height == 1
          input0_info.shape[1],  // Width
          input0_info.shape[2]   // Channels
      };

      if (!input0_info.is_initializer) {
        // Add Reshape node to transform 1D input to 2D (i.e., set height to 1).
        // We don't need to do this for initializers, because the number of elements does not change. We can just
        // modify the shape dimensions.
        bool is_graph_input = qnn_model_wrapper.IsGraphInput(input0_name);
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(input0_name,
                                                             conv_input0_name,
                                                             input0_info.shape,
                                                             shape,
                                                             input0_info.qnn_data_type,
                                                             input0_info.quant_param,
                                                             do_op_validation,
                                                             is_graph_input));
      }

      Qnn_TensorType_t tensor_type = GetInputTensorType(qnn_model_wrapper, conv_input0_name);
      QnnTensorWrapper input_tensorwrapper(conv_input0_name, tensor_type, input0_info.qnn_data_type, input0_info.quant_param,
                                           std::move(shape), std::move(unpacked_tensor));
      ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
    } else {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input0_name;
    }
  }

  //
  // Input 1: weight
  // We need to first reshape the weight inorder to handle 1D convolutions with the Conv2d operator.
  // Next, we have to transpose the weight because ORT layout transformations do not change the weight layout.
  //
  {
    const std::string& input1_name = inputs[1].node_arg.Name();
    OnnxInputInfo input_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(inputs[1], is_quantized_model, input_info));

    std::string conv_weight_input_name = input_info.is_initializer ? input1_name : input1_name + "_trans_qnn_ep";
    input_names.push_back(conv_weight_input_name);

    // Create the shape after reshaping.
    // Set height to 1 to be able to use 2D convolution.
    std::vector<uint32_t> shape_2d = {
        input_info.shape[0],  // N
        input_info.shape[1],  // Channels
        1,                    // Height == 1
        input_info.shape[2],  // Width
    };

    std::vector<uint32_t> final_shape;
    final_shape.resize(4);

    // Create the final shape after the weights are transposed to HWCN.
    if (node_unit.OpType() == "Conv") {
      ORT_RETURN_IF_ERROR(NchwShapeToHwcn(shape_2d, final_shape));
    } else if (node_unit.OpType() == "ConvTranspose") {
      ORT_RETURN_IF_ERROR(CnhwShapeToHwcn(shape_2d, final_shape));
    } else {
      ORT_THROW("Unexpected operator %s", onnx_op_type);
    }

    const std::string reshape_output = input1_name + "_reshape_qnn_ep";
    std::vector<uint8_t> unpacked_tensor;
    if (input_info.is_initializer) {
      //
      // Create a reshaped "view" of the initializer tensor with [N, C, 1, W] dims.
      //
      std::vector<int64_t> shape_2d_int64;
      shape_2d_int64.resize(4);

      std::transform(shape_2d.begin(), shape_2d.end(), shape_2d_int64.begin(), [](uint32_t dim) -> int64_t {
        return static_cast<int64_t>(dim);
      });

      const TensorShape tensor_shape = TensorShape::FromExistingBuffer(shape_2d_int64);  // Does not own shape data.
      const DataTypeImpl* tensor_dtype = DataTypeImpl::TensorTypeFromONNXEnum(
                                             input_info.initializer_tensor->data_type())
                                             ->GetElementType();
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_info.initializer_tensor, unpacked_tensor));

      Tensor tensor_2d(tensor_dtype, tensor_shape, unpacked_tensor.data(), OrtMemoryInfo{});  // Does not own data.
      ONNX_NAMESPACE::TensorProto reshaped_initializer = onnxruntime::utils::TensorToTensorProto(tensor_2d,
                                                                                                 reshape_output);

      //
      // Get transposed initializer bytes.
      //
      if (node_unit.OpType() == "Conv") {
        ORT_RETURN_IF_ERROR(TransposeFromNchwToHwcn(qnn_model_wrapper, reshaped_initializer, unpacked_tensor));
      } else if (node_unit.OpType() == "ConvTranspose") {
        ORT_RETURN_IF_ERROR(TransposeFromCnhwToHwcn(qnn_model_wrapper, reshaped_initializer, unpacked_tensor));
      } else {
        ORT_THROW("Unexpected operator %s", node_unit.OpType());
      }
    } else {
      // Dynamic weight: Add nodes to reshape to 2D, and then transpose.
      bool is_graph_input = qnn_model_wrapper.IsGraphInput(input1_name);
      LOGS(logger, VERBOSE) << "Adding Reshape (to 2D) and HWCN Transpose node after input: " << input1_name;
      if (node_unit.OpType() == "Conv") {
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(input1_name,
                                                             reshape_output,
                                                             input_info.shape,
                                                             shape_2d,
                                                             input_info.qnn_data_type,
                                                             input_info.quant_param,
                                                             do_op_validation,
                                                             is_graph_input));
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddNchwToHwcnTranspose(node_unit.Index(),
                                                                     reshape_output,
                                                                     conv_weight_input_name,
                                                                     shape_2d,
                                                                     final_shape,
                                                                     input_info.qnn_data_type,
                                                                     input_info.quant_param,
                                                                     do_op_validation,
                                                                     false));
      } else if (node_unit.OpType() == "ConvTranspose") {
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(input1_name,
                                                             reshape_output,
                                                             input_info.shape,
                                                             shape_2d,
                                                             input_info.qnn_data_type,
                                                             input_info.quant_param,
                                                             do_op_validation,
                                                             is_graph_input));
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddCnhwToHwcnTranspose(node_unit.Index(),
                                                                     reshape_output,
                                                                     conv_weight_input_name,
                                                                     shape_2d,
                                                                     final_shape,
                                                                     input_info.qnn_data_type,
                                                                     input_info.quant_param,
                                                                     do_op_validation,
                                                                     false));
      } else {
        ORT_THROW("Unexpected operator %s", node_unit.OpType());
      }
    }

    Qnn_TensorType_t tensor_type = GetInputTensorType(qnn_model_wrapper, conv_weight_input_name);
    QnnTensorWrapper input_tensorwrapper(conv_weight_input_name, tensor_type, input_info.qnn_data_type,
                                         input_info.quant_param, std::move(final_shape), std::move(unpacked_tensor));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");
  }

  //
  // Input 2: bias
  //
  if (inputs.size() == 3) {
    ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[2], logger, is_quantized_model, input_names));
  }

  return Status::OK();
}

Status ConvOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger,
                                    bool is_quantized_model,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  assert(inputs.size() >= 2);

  std::vector<uint32_t> input0_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[0].node_arg, input0_shape),
                    "QNN EP: Cannot get shape for first input");

  const bool is_1d_conv = input0_shape.size() == 3;

  if (is_1d_conv) {
    return ProcessConv1DInputs(qnn_model_wrapper, node_unit, logger, is_quantized_model, input_names, do_op_validation);
  }

  return ProcessConv2DInputs(qnn_model_wrapper, node_unit, logger, is_quantized_model, input_names, do_op_validation);
}

static Status GetAutoPadding(std::vector<int32_t>& pads, const std::string& op_type, const std::string& auto_pad,
                             const std::array<uint32_t, 2>& strides, const std::array<uint32_t, 2>& dilations,
                             const std::array<uint32_t, 2>& input_dims, const std::array<uint32_t, 2>& filter_dims,
                             const std::array<uint32_t, 2>& output_dims, const std::array<uint32_t, 2>& output_padding = {0, 0}) {
  constexpr size_t HEIGHT_IDX = 0;
  constexpr size_t WIDTH_IDX = 1;

  std::array<uint32_t, 2> total_padding = {};

  if (op_type == "ConvTranspose") {
    // height_out = floor(stride[0] * (shape(in[0])[height] - 1) + shape(in[1])[height] - pad_amount[0,0] - pad_amount[0,1] + output_padding[0])
    //
    // Set total_height_padding equal to pad_amount[0,0] + pad_amount[0,1] and solve for it.
    total_padding[HEIGHT_IDX] = strides[HEIGHT_IDX] * (input_dims[HEIGHT_IDX] - 1) + output_padding[HEIGHT_IDX] + filter_dims[HEIGHT_IDX] - output_dims[HEIGHT_IDX];

    // width_out = floor(stride[1] * (shape(in[0])[width] - 1) + shape(in[1])[width] - pad_amount[1,0] - pad_amount[1,1] + output_padding[1])
    //
    // Set total_width_padding equal to pad_amount[1,0] + pad_amount[1,1] and solve for it.
    total_padding[WIDTH_IDX] = strides[WIDTH_IDX] * (input_dims[WIDTH_IDX] - 1) + output_padding[WIDTH_IDX] + filter_dims[WIDTH_IDX] - output_dims[WIDTH_IDX];
  } else if (op_type == "Conv") {
    // dilated_filter_height = (shape(in[1])[height] - 1) * dilation[0] + 1
    // height_out = floor((pad_amount[0,0] + shape(in[0])[height] + pad_amount[0,1] - dilated_filter_height) / stride[0] + 1)
    //
    // Set total_height_padding equal to pad_amount[0,0] + pad_amount[0,1] and solve for it.
    uint32_t dilated_filter_height = (filter_dims[HEIGHT_IDX] - 1) * dilations[HEIGHT_IDX] + 1;
    total_padding[HEIGHT_IDX] = (output_dims[HEIGHT_IDX] - 1) * strides[HEIGHT_IDX] + dilated_filter_height - input_dims[HEIGHT_IDX];  // Total height padding

    // dilated_filter_width = (shape(in[1])[width] - 1) * dilation[1] + 1
    // width_out = floor((pad_amount[1,0] + shape(in[0])[width] + pad_amount[1,1] - dilated_filter_width) / stride[1] + 1)
    //
    // Set total_width_padding equal to pad_amount[1,0] + pad_amount[1,1] and solve for it.
    uint32_t dilated_filter_width = (filter_dims[WIDTH_IDX] - 1) * dilations[WIDTH_IDX] + 1;
    total_padding[WIDTH_IDX] = (output_dims[WIDTH_IDX] - 1) * strides[WIDTH_IDX] + dilated_filter_width - input_dims[WIDTH_IDX];  // Total width padding
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN EP: Cannot calculate auto-padding for unsupported operator type: ",
                           op_type.c_str());
  }

  pads.resize(4);  // Make room.

  if (auto_pad == "SAME_UPPER") {
    pads[0] = total_padding[0] / 2;
    pads[1] = total_padding[1] / 2;
    pads[2] = total_padding[0] - pads[0];
    pads[3] = total_padding[1] - pads[1];
  } else if (auto_pad == "SAME_LOWER") {
    pads[2] = total_padding[0] / 2;
    pads[3] = total_padding[1] / 2;
    pads[0] = total_padding[0] - pads[2];
    pads[1] = total_padding[1] - pads[3];
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN EP: Cannot calculate auto-padding for unsupported auto_pad setting: ",
                           auto_pad.c_str());
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

  // Handle 1D conv.
  if (dilation_values.size() == 1) {
    const int32_t width_dilation = dilation_values[0];
    dilation_values.resize(2);
    dilation_values[0] = 1;  // Height == 1
    dilation_values[1] = width_dilation;
  }

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

    // Handle 1D conv.
    if (output_padding_values.size() == 1) {
      const int32_t width_out_pad = output_padding_values[0];
      output_padding_values.resize(2);
      output_padding_values[0] = 0;  // Height: default output padding of 0
      output_padding_values[1] = width_out_pad;
    }

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
  const bool is_1d_conv = output_shape.size() == 3;

  // Conv attribute strides
  auto stride_values = node_helper.Get("strides", std::vector<int32_t>{1, 1});

  // Handle 1D conv.
  if (stride_values.size() == 1) {
    const int32_t width_stride = stride_values[0];
    stride_values.resize(2);
    stride_values[0] = 1;  // Height: default stride of 1
    stride_values[1] = width_stride;
  }

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
    const auto& input_0 = node_unit.Inputs()[0];
    const auto& input_1 = node_unit.Inputs()[1];
    std::vector<uint32_t> input_0_shape;  // NHWC
    std::vector<uint32_t> input_1_shape;  // NCHW
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_0.node_arg, input_0_shape), "Cannot get shape");
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(input_1.node_arg, input_1_shape), "Cannot get shape");

    std::array<uint32_t, 2> input_dims = {};
    std::array<uint32_t, 2> filter_dims = {};
    std::array<uint32_t, 2> output_dims = {};

    if (is_1d_conv) {
      input_dims[0] = 1;
      input_dims[1] = input_0_shape[1];

      filter_dims[0] = 1;
      filter_dims[1] = input_1_shape[2];

      output_dims[0] = 1;
      output_dims[1] = output_shape[1];
    } else {
      input_dims[0] = input_0_shape[1];
      input_dims[1] = input_0_shape[2];

      filter_dims[0] = input_1_shape[2];
      filter_dims[1] = input_1_shape[3];

      output_dims[0] = output_shape[1];
      output_dims[1] = output_shape[2];
    }

    ORT_RETURN_IF_ERROR(GetAutoPadding(pad_values, node_unit.OpType(), auto_pad,
                                       {SafeInt<uint32_t>(stride_values[0]), SafeInt<uint32_t>(stride_values[1])},
                                       {dilations_0, dilations_1}, input_dims, filter_dims, output_dims,
                                       {output_padding_0, output_padding_1}));
  } else {
    // Conv/ConvTranspose attribute pads
    pad_values = node_helper.Get("pads", pad_values);

    // Handle 1D conv.
    if (pad_values.size() == 2) {
      const int32_t width_pad_begin = pad_values[0];
      const int32_t width_pad_end = pad_values[1];
      pad_values.resize(4);
      pad_values[0] = 0;  // Height pad begin: 0
      pad_values[1] = width_pad_begin;
      pad_values[2] = 0;  // Height pad end: 0
      pad_values[3] = width_pad_end;
    }
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
  utils::InitializeQuantizeParam(output_quantize_param, is_quantized_model);

  const auto* type_proto = outputs[0].node_arg.TypeAsProto();
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  ORT_RETURN_IF_ERROR(utils::GetQnnDataType(is_quantized_model, type_proto, qnn_data_type));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.ProcessQuantizationParameter(outputs[0].quant_param,
                                                                   output_quantize_param.scaleOffsetEncoding.scale,
                                                                   output_quantize_param.scaleOffsetEncoding.offset),
                    "Cannot get quantization parameter");

  const uint32_t group = SafeInt<uint32_t>(node_helper.Get("group", static_cast<int64_t>(1)));
  uint32_t num_output_channels = is_1d_conv ? output_shape[2] : output_shape[3];
  uint32_t num_input_channels = 0;
  ORT_RETURN_IF_ERROR(GetInputChannelNumber(qnn_model_wrapper, node_unit, num_input_channels));
  LOGS(logger, VERBOSE) << (node_unit.OpType() == "Conv" ? "Conv:" : "ConvTranspose:")
                        << " num_output_channels: " << num_output_channels
                        << ", num_input_channels: " << num_input_channels << ", group: " << group;
  const static std::string depthwise_conv2d = "DepthWiseConv2d";
  bool is_depthwise_conv2d = false;
  if ((node_unit.OpType() == "Conv") && (num_input_channels == num_output_channels) && (group == num_output_channels)) {
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

  if (is_1d_conv) {
    bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
    std::vector<uint32_t> output_shape_2d = {
        output_shape[0],  // N
        1,                // H == 1
        output_shape[1],  // W
        output_shape[2],  // C
    };
    const std::string conv_output_name = output_name + "_2d_qnn_ep";
    QnnTensorWrapper output_tensorwrapper(conv_output_name, QNN_TENSOR_TYPE_NATIVE, qnn_data_type, output_quantize_param,
                                          std::vector<uint32_t>(output_shape_2d));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(output_tensorwrapper)), "Failed to add tensor.");

    ORT_RETURN_IF_NOT(qnn_model_wrapper.CreateQnnNode(GetNodeName(node_unit),
                                                      qnn_def::package_name,
                                                      output_node_type,
                                                      std::move(input_names),
                                                      {conv_output_name},
                                                      std::move(param_tensor_names)),
                      "Failed to add node.");
    // Add Reshape to convert QNN Conv2d/TransposeConv2d/DepthWiseConv2d output to 1D.
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(conv_output_name,
                                                         output_name,
                                                         output_shape_2d,
                                                         output_shape,
                                                         qnn_data_type,
                                                         output_quantize_param,
                                                         do_op_validation,
                                                         false,
                                                         is_graph_output));
  } else {
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
  }

  return Status::OK();
}

void CreateConvOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<ConvOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime
