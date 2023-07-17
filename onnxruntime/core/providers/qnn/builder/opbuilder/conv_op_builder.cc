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

// ONNX convolution types supported by this builder.
// We translate node_unit.OpType() into this enum to avoid repeated string comparisons.
enum class OnnxConvType {
  kConv,
  kConvTranspose,
};

static Status GetOnnxConvType(const std::string& onnx_op_type, OnnxConvType& conv_type) {
  if (onnx_op_type == "Conv") {
    conv_type = OnnxConvType::kConv;
  } else if (onnx_op_type == "ConvTranspose") {
    conv_type = OnnxConvType::kConvTranspose;
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN EP: Unsupported ONNX convolution op type: ", onnx_op_type.c_str());
  }

  return Status::OK();
}

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

// Conv/ConvTranspose ops are sensitive with data layout, no special validation so far
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
  ORT_RETURN_IF(inputs.size() < 2, "QNN Conv must have at least 2 inputs.");

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

  OnnxConvType conv_type = {};
  ORT_RETURN_IF_ERROR(GetOnnxConvType(node_unit.OpType(), conv_type));

  if (conv_type == OnnxConvType::kConvTranspose) {
    // QNN's TransposeConv2d only supports default dilation values of 1.
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

Status ConvOpBuilder::ProcessConv2DInputs(QnnModelWrapper& qnn_model_wrapper,
                                          const NodeUnit& node_unit,
                                          const logging::Logger& logger,
                                          bool is_quantized_model,
                                          std::vector<std::string>& input_names,
                                          bool do_op_validation) const {
  const auto& inputs = node_unit.Inputs();
  const size_t num_inputs = inputs.size();
  OnnxConvType conv_type = {};
  ORT_RETURN_IF_ERROR(GetOnnxConvType(node_unit.OpType(), conv_type));

  assert(num_inputs >= 2);  // Checked by IsOpSupported.

  //
  // Input 0
  //
  ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[0], logger, is_quantized_model, input_names));

  //
  // Input 1: weight
  //
  {
    const std::string& input1_name = inputs[1].node_arg.Name();
    OnnxInputInfo input_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(inputs[1], is_quantized_model, input_info));

    std::string actual_name = input_info.is_initializer ? input1_name : input1_name + "_ort_qnn_ep_transpose";
    input_names.push_back(actual_name);

    std::vector<uint32_t> actual_shape;
    actual_shape.resize(input_info.shape.size());

    // Change shape to HWCN, it could be initializer or normal input
    if (conv_type == OnnxConvType::kConv) {
      ORT_RETURN_IF_ERROR(NchwShapeToHwcn(input_info.shape, actual_shape));
    } else if (conv_type == OnnxConvType::kConvTranspose) {
      ORT_RETURN_IF_ERROR(CnhwShapeToHwcn(input_info.shape, actual_shape));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN EP: Unexpected convolution op type: ", node_unit.OpType().c_str());
    }

    std::vector<uint8_t> unpacked_tensor;
    if (input_info.is_initializer) {
      // Get transposed initializer bytes.
      if (conv_type == OnnxConvType::kConv) {
        ORT_RETURN_IF_ERROR(TransposeFromNchwToHwcn(qnn_model_wrapper, *input_info.initializer_tensor, unpacked_tensor));
      } else if (conv_type == OnnxConvType::kConvTranspose) {
        ORT_RETURN_IF_ERROR(TransposeFromCnhwToHwcn(qnn_model_wrapper, *input_info.initializer_tensor, unpacked_tensor));
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN EP: Unexpected convolution op type: ", node_unit.OpType().c_str());
      }
    } else {
      // Add transpose node above weight input.
      bool is_graph_input = qnn_model_wrapper.IsGraphInput(input1_name);
      LOGS(logger, VERBOSE) << "Add HWCN Transpose node after input: " << input1_name;

      if (conv_type == OnnxConvType::kConv) {
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddNchwToHwcnTranspose(node_unit.Index(),
                                                                     input1_name,
                                                                     actual_name,
                                                                     input_info.shape,
                                                                     actual_shape,
                                                                     input_info.qnn_data_type,
                                                                     input_info.quant_param,
                                                                     do_op_validation,
                                                                     is_graph_input));
      } else if (conv_type == OnnxConvType::kConvTranspose) {
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
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN EP: Unexpected convolution op type: ", node_unit.OpType().c_str());
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
  if (num_inputs == 3) {
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
  OnnxConvType conv_type = {};
  ORT_RETURN_IF_ERROR(GetOnnxConvType(node_unit.OpType(), conv_type));

  assert(num_inputs >= 2);  // Checked by IsOpSupported.

  //
  // Input 0
  //

  {
    const std::string& input0_name = inputs[0].node_arg.Name();
    OnnxInputInfo input0_info = {};
    ORT_RETURN_IF_ERROR(qnn_model_wrapper.GetOnnxInputInfo(inputs[0], is_quantized_model, input0_info));

    const std::string conv_input0_name = input0_info.is_initializer ? input0_name
                                                                    : input0_name + "_ort_qnn_ep_reshape";
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

    std::string conv_weight_input_name = input_info.is_initializer ? input1_name : input1_name + "_ort_qnn_ep_transpose";
    input_names.push_back(conv_weight_input_name);

    // Create the shape after reshaping.
    // Set height to 1 to be able to use 2D convolution.
    // Note: Conv shape is [N,C,1,W]. ConvTranspose shape is [C,N,1,W]
    std::vector<uint32_t> shape_2d = {
        input_info.shape[0],  // N
        input_info.shape[1],  // Channels
        1,                    // Height == 1
        input_info.shape[2],  // Width
    };

    std::vector<uint32_t> final_shape;
    final_shape.resize(4);

    // Create the final shape after the weights are transposed to HWCN.
    if (conv_type == OnnxConvType::kConv) {
      ORT_RETURN_IF_ERROR(NchwShapeToHwcn(shape_2d, final_shape));
    } else if (conv_type == OnnxConvType::kConvTranspose) {
      ORT_RETURN_IF_ERROR(CnhwShapeToHwcn(shape_2d, final_shape));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN EP: Unexpected convolution op type: ", node_unit.OpType().c_str());
    }

    const std::string reshape_output = input1_name + "_ort_qnn_ep_reshape";
    std::vector<uint8_t> unpacked_tensor;
    if (input_info.is_initializer) {
      //
      // Create a reshaped "view" of the initializer tensor with [N, C, 1, W] dims for Conv
      // ([C, N, 1, W] for ConvTranspose).
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
      if (conv_type == OnnxConvType::kConv) {
        ORT_RETURN_IF_ERROR(TransposeFromNchwToHwcn(qnn_model_wrapper, reshaped_initializer, unpacked_tensor));
      } else if (conv_type == OnnxConvType::kConvTranspose) {
        ORT_RETURN_IF_ERROR(TransposeFromCnhwToHwcn(qnn_model_wrapper, reshaped_initializer, unpacked_tensor));
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN EP: Unexpected convolution op type: ", node_unit.OpType().c_str());
      }
    } else {
      // Dynamic weight: Add nodes to reshape to 2D, and then transpose.
      bool is_graph_input = qnn_model_wrapper.IsGraphInput(input1_name);
      LOGS(logger, VERBOSE) << "Adding Reshape (to 2D) and HWCN Transpose node after input: " << input1_name;
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddReshapeNode(input1_name,
                                                           reshape_output,
                                                           input_info.shape,
                                                           shape_2d,
                                                           input_info.qnn_data_type,
                                                           input_info.quant_param,
                                                           do_op_validation,
                                                           is_graph_input));
      if (conv_type == OnnxConvType::kConv) {
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddNchwToHwcnTranspose(node_unit.Index(),
                                                                     reshape_output,
                                                                     conv_weight_input_name,
                                                                     shape_2d,
                                                                     final_shape,
                                                                     input_info.qnn_data_type,
                                                                     input_info.quant_param,
                                                                     do_op_validation,
                                                                     false));
      } else if (conv_type == OnnxConvType::kConvTranspose) {
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
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN EP: Unexpected convolution op type: ", node_unit.OpType().c_str());
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
  if (num_inputs == 3) {
    ORT_RETURN_IF_ERROR(ProcessInput(qnn_model_wrapper, inputs[2], logger, is_quantized_model, input_names));
  }

  return Status::OK();
}

static Status GetAutoPadding(std::vector<uint32_t>& pads, const std::string& auto_pad, OnnxConvType conv_type,
                             const std::array<uint32_t, 2>& strides, const std::array<uint32_t, 2>& dilations,
                             const std::array<uint32_t, 2>& input_dims, const std::array<uint32_t, 2>& filter_dims,
                             const std::array<uint32_t, 2>& output_dims, const std::array<uint32_t, 2>& output_padding) {
  constexpr size_t HEIGHT_IDX = 0;
  constexpr size_t WIDTH_IDX = 1;

  std::array<uint32_t, 2> total_padding = {};

  if (conv_type == OnnxConvType::kConv) {
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
  } else if (conv_type == OnnxConvType::kConvTranspose) {
    // height_out = floor(stride[0] * (shape(in[0])[height] - 1) + shape(in[1])[height] - pad_amount[0,0] - pad_amount[0,1] + output_padding[0])
    //
    // Set total_height_padding equal to pad_amount[0,0] + pad_amount[0,1] and solve for it.
    total_padding[HEIGHT_IDX] = strides[HEIGHT_IDX] * (input_dims[HEIGHT_IDX] - 1) + output_padding[HEIGHT_IDX] + filter_dims[HEIGHT_IDX] - output_dims[HEIGHT_IDX];

    // width_out = floor(stride[1] * (shape(in[0])[width] - 1) + shape(in[1])[width] - pad_amount[1,0] - pad_amount[1,1] + output_padding[1])
    //
    // Set total_width_padding equal to pad_amount[1,0] + pad_amount[1,1] and solve for it.
    total_padding[WIDTH_IDX] = strides[WIDTH_IDX] * (input_dims[WIDTH_IDX] - 1) + output_padding[WIDTH_IDX] + filter_dims[WIDTH_IDX] - output_dims[WIDTH_IDX];
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN EP: Unexpected conv op type when computing auto-padding");
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
  OnnxConvType conv_type = {};
  ORT_RETURN_IF_ERROR(GetOnnxConvType(node_unit.OpType(), conv_type));

  NodeAttrHelper node_helper(node_unit);
  std::vector<std::string> param_tensor_names;

  // Dilations parameter
  std::vector<uint32_t> dilations = {1, 1};

  if (conv_type == OnnxConvType::kConv) {
    dilations = node_helper.Get("dilations", std::vector<uint32_t>{1, 1});

    // Handle 1D conv by setting height dilation to 1.
    if (dilations.size() == 1) {
      const uint32_t width_dilation = dilations[0];
      dilations.resize(2);
      dilations[0] = 1;  // Height == 1
      dilations[1] = width_dilation;
    }

    QnnParamWrapper dilation_paramwrapper(node_unit.Index(), node_unit.Name(), qnn_def::dilation,
                                          {SafeInt<uint32_t>(dilations.size())}, std::vector<uint32_t>(dilations));
    param_tensor_names.push_back(dilation_paramwrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(dilation_paramwrapper));
  }

  // Strides parameter.
  auto strides = node_helper.Get("strides", std::vector<uint32_t>{1, 1});
  {
    // Handle 1D conv by setting the height stride to 1.
    if (strides.size() == 1) {
      const uint32_t width_stride = strides[0];
      strides.resize(2);
      strides[0] = 1;  // Height
      strides[1] = width_stride;
    }

    QnnParamWrapper stride_amount_paramwrapper(node_unit.Index(), node_unit.Name(), qnn_def::stride,
                                               {SafeInt<uint32_t>(strides.size())}, std::vector<uint32_t>(strides));
    param_tensor_names.push_back(stride_amount_paramwrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(stride_amount_paramwrapper));
  }

  // Output padding parameter. (Only for ConvTranspose)
  std::vector<uint32_t> output_padding = {0, 0};

  if (conv_type == OnnxConvType::kConvTranspose) {
    output_padding = node_helper.Get("output_padding", std::vector<uint32_t>{0, 0});

    // Handle 1D conv.
    if (output_padding.size() == 1) {
      const uint32_t width_out_pad = output_padding[0];
      output_padding.resize(2);
      output_padding[0] = 0;  // Height: default output padding of 0
      output_padding[1] = width_out_pad;
    }

    QnnParamWrapper output_padding_paramwrapper(node_unit.Index(), node_unit.Name(), qnn_def::output_padding,
                                                {static_cast<uint32_t>(output_padding.size())}, std::vector<uint32_t>(output_padding));
    param_tensor_names.push_back(output_padding_paramwrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(output_padding_paramwrapper));
  }

  const auto& outputs = node_unit.Outputs();
  const auto& output_name = outputs[0].node_arg.Name();

  std::vector<uint32_t> output_shape;
  ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(outputs[0].node_arg, output_shape), "Cannot get shape");
  const bool is_1d_conv = output_shape.size() == 3;

  // Pads attribute
  {
    std::vector<uint32_t> pads = node_helper.Get("pads", std::vector<uint32_t>({0, 0, 0, 0}));
    auto auto_pad = node_helper.Get("auto_pad", std::string("NOTSET"));
    ORT_RETURN_IF(auto_pad != "NOTSET" && auto_pad != "SAME_LOWER" && auto_pad != "SAME_UPPER",
                  "QNN Conv operators do not support 'auto_pad' value: ", auto_pad.c_str());

    if (auto_pad != "NOTSET") {
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

      ORT_RETURN_IF_ERROR(GetAutoPadding(pads, auto_pad, conv_type, {strides[0], strides[1]}, {dilations[0], dilations[1]},
                                         input_dims, filter_dims, output_dims, {output_padding[0], output_padding[1]}));
    } else {
      // Handle 1D conv by setting padding for height to 0.
      if (pads.size() == 2) {
        const uint32_t width_pad_begin = pads[0];
        const uint32_t width_pad_end = pads[1];
        pads.resize(4);
        pads[0] = 0;  // Height pad begin: 0
        pads[1] = width_pad_begin;
        pads[2] = 0;  // Height pad end: 0
        pads[3] = width_pad_end;
      }
    }

    ReArranagePads(pads);
    QnnParamWrapper pad_amount_paramwrapper(node_unit.Index(), node_unit.Name(), qnn_def::pad_amount,
                                            {2, 2}, std::move(pads));
    param_tensor_names.push_back(pad_amount_paramwrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(pad_amount_paramwrapper));
  }

  const uint32_t group = node_helper.Get("group", static_cast<uint32_t>(1));
  const uint32_t num_output_channels = output_shape.back();
  uint32_t num_input_channels = 0;
  ORT_RETURN_IF_ERROR(GetInputChannelNumber(qnn_model_wrapper, node_unit, num_input_channels));

  const bool is_depthwise_conv2d = (conv_type == OnnxConvType::kConv) && (num_input_channels == num_output_channels) &&
                                   (group == num_output_channels);

  if (!is_depthwise_conv2d) {  // DepthWiseConv2d does not need a group parameter.
    Qnn_Scalar_t group_qnn_scalar = QNN_SCALAR_INIT;
    group_qnn_scalar.dataType = QNN_DATATYPE_UINT_32;
    group_qnn_scalar.uint32Value = group;
    QnnParamWrapper group_paramwrapper(node_unit.Index(), node_unit.Name(), qnn_def::group, group_qnn_scalar);
    param_tensor_names.push_back(group_paramwrapper.GetParamTensorName());
    qnn_model_wrapper.AddParamWrapper(std::move(group_paramwrapper));
  } else {
    LOGS(logger, VERBOSE) << "Using DepthWiseConv2d instead of Conv2d for node " << node_unit.Name();
  }

  const std::string& output_node_type = is_depthwise_conv2d ? QNN_OP_DEPTH_WISE_CONV_2D : GetQnnOpType(node_unit.OpType());

  Qnn_QuantizeParams_t output_quantize_param = QNN_QUANTIZE_PARAMS_INIT;
  utils::InitializeQuantizeParam(output_quantize_param, is_quantized_model);

  const auto* type_proto = outputs[0].node_arg.TypeAsProto();
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;
  ORT_RETURN_IF_ERROR(utils::GetQnnDataType(is_quantized_model, type_proto, qnn_data_type));
  ORT_RETURN_IF_NOT(qnn_model_wrapper.ProcessQuantizationParameter(outputs[0].quant_param,
                                                                   output_quantize_param.scaleOffsetEncoding.scale,
                                                                   output_quantize_param.scaleOffsetEncoding.offset),
                    "Cannot get quantization parameter");

  if (is_1d_conv) {
    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
    std::vector<uint32_t> output_shape_2d = {
        output_shape[0],  // N
        1,                // H == 1
        output_shape[1],  // W
        output_shape[2],  // C
    };
    const std::string conv_output_name = output_name + "_ort_qnn_ep_conv2d";
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

    // Add Reshape to convert QNN Conv2d/TransposeConv2d/DepthWiseConv2d output back to 1D.
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
    const bool is_graph_output = qnn_model_wrapper.IsGraphOutput(output_name);
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
