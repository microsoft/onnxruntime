// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"
#include "builder_utils.h"

namespace onnxruntime {
namespace webnn {

class ConvOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const WebnnDeviceType device_type, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const Node& node, const emscripten::val& wnn_limits,
                              const logging::Logger& logger) const override;
};

void ConvOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // skip the weight for conv as we need to transpose for preferred layout NHWC.
  if (model_builder.GetPreferredLayout() == DataLayout::NHWC) {
    model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());  // W
  }
}

// Helper functions
common::Status SetConvBaseOptions(ModelBuilder& model_builder,
                                  const Node& node, emscripten::val& options,
                                  const std::vector<int64_t> input_shape,
                                  const std::vector<int64_t> weight_shape,
                                  const std::vector<int64_t>& strides,
                                  const std::vector<int64_t>& dilations,
                                  std::vector<int64_t>& pads,
                                  const bool is_nhwc,
                                  const bool is_conv1d,
                                  const logging::Logger& logger) {
  NodeAttrHelper helper(node);
  const auto& input_defs = node.InputDefs();

  // Add Padding.
  AutoPadType auto_pad_type = StringToAutoPadType(helper.Get("auto_pad", "NOTSET"));
  std::vector<int64_t> pads_out;
  if (node.OpType() == "Conv" || node.OpType() == "ConvInteger") {
    // Calculate explicit padding for autoPad.
    if (AutoPadType::SAME_UPPER == auto_pad_type || AutoPadType::SAME_LOWER == auto_pad_type) {
      ORT_RETURN_IF_ERROR(HandleAutoPad(input_shape, weight_shape[2], weight_shape[3],
                                        pads, strides, dilations, auto_pad_type, pads_out, !is_nhwc));
      pads = pads_out;
    }
  } else if (node.OpType() == "ConvTranspose") {
    std::vector<int64_t> output_shape = helper.Get("output_shape", std::vector<int64_t>{-1, -1});
    // Appending 1's if it is ConvTranspose 1d and output shape is provided.
    if (output_shape.size() == 1 && is_conv1d && output_shape[0] != -1) {
      output_shape.push_back(1);
    }

    std::vector<int64_t> output_padding = helper.Get("output_padding", std::vector<int64_t>{0, 0});
    // Appending 0's if it is ConvTranspose 1d.
    if (output_padding.size() == 1 && is_conv1d) {
      output_padding.push_back(0);
    }
    options.set("outputPadding", emscripten::val::array(GetVecUint32FromVecInt64(output_padding)));

    // If output shape is explicitly provided, compute the pads.
    // Otherwise compute the output shape, as well as the pads if the auto_pad attribute is SAME_UPPER/SAME_LOWER.
    ORT_RETURN_IF_ERROR(ComputeConvTransposePadsAndOutputShape(input_shape, weight_shape[2], weight_shape[3],
                                                               pads, strides, dilations, output_padding,
                                                               auto_pad_type, pads_out, output_shape, !is_nhwc));

    if (output_shape[0] != -1 && output_shape[1] != -1) {
      options.set("outputSizes", emscripten::val::array(GetVecUint32FromVecInt64(output_shape)));
    }
    pads = pads_out;
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "conv_op_builder only supports Op Conv, ConvInteger and ConvTranspose.");
  }

  const auto group = helper.Get("group", static_cast<uint32_t>(1));
  options.set("groups", group);
  options.set("strides", emscripten::val::array(GetVecUint32FromVecInt64(strides)));
  options.set("dilations", emscripten::val::array(GetVecUint32FromVecInt64(dilations)));

  // Permute the ONNX's pads, which is [beginning_height, beginning_width, ending_height, ending_width],
  // while WebNN's padding is [beginning_height, ending_height, beginning_width, ending_width].
  const std::vector<int64_t> padding{pads[0], pads[2], pads[1], pads[3]};
  options.set("padding", emscripten::val::array(GetVecUint32FromVecInt64(padding)));

  // Add bias if present.
  if (input_defs.size() > 2) {
    options.set("bias", model_builder.GetOperand(input_defs[2]->Name()));
  }

  return Status::OK();
}

// Both depthwise Conv and ConvTranspose share the same logic to add the layout.
Status AddInitializerInNewLayout(ModelBuilder& model_builder,
                                 const std::string& name,
                                 bool is_conv,
                                 bool is_conv1d) {
  const auto& tensor = *model_builder.GetInitializerTensors().at(name);
  auto data_type = tensor.data_type();

  const auto& shape = tensor.dims();
  std::vector<uint32_t> dims = GetVecUint32FromVecInt64(std::vector<int64_t>(std::begin(shape), std::end(shape)));

  if (is_conv1d) {
    // Support conv1d by prepending a 1 size dimension.
    dims.push_back(1);
  }

  const uint8_t* src = nullptr;
  Initializer unpacked_tensor(tensor, model_builder.GetGraphViewer().ModelPath());
  src = unpacked_tensor.DataAsByteSpan().data();
  const auto out_t = dims[0], in_t = dims[1],
             h_t = dims[2], w_t = dims[3];
  std::vector<uint32_t> dest_shape;
  if (is_conv == 1)
    dest_shape = {out_t, h_t, w_t, in_t};  // L_0231
  else
    dest_shape = {in_t, h_t, w_t, out_t};  // L_1230 for depthwise conv and convTranspose weight

  SafeInt<size_t> num_elements = SafeInt<size_t>(Product(dest_shape));

  size_t element_size{0};
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      element_size = sizeof(uint8_t);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      element_size = sizeof(int8_t);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      element_size = sizeof(uint16_t);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      element_size = sizeof(float);
      break;
    default:
      break;
  }
  std::unique_ptr<uint8_t[]> buffer_holder(new uint8_t[element_size * num_elements]);
  uint8_t* buffer = buffer_holder.get();

  for (uint32_t out = 0; out < out_t; out++) {
    for (uint32_t in = 0; in < in_t; in++) {
      for (uint32_t h = 0; h < h_t; h++) {
        for (uint32_t w = 0; w < w_t; w++) {
          auto onnx_idx = out * in_t * h_t * w_t +
                          in * h_t * w_t +
                          h * w_t +
                          w;

          uint32_t nnapi_idx;
          if (is_conv == 1) {  // L_0231
            nnapi_idx = out * h_t * w_t * in_t +
                        h * w_t * in_t +
                        w * in_t +
                        in;
          } else {  // L_1230 for depthwise conv weight
            nnapi_idx = in * h_t * w_t * out_t +
                        h * w_t * out_t +
                        w * out_t +
                        out;
          }

          for (size_t i = 0; i < element_size; i++) {
            buffer[element_size * nnapi_idx + i] = src[element_size * onnx_idx + i];
          }
        }
      }
    }
  }
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(name, buffer, num_elements * element_size,
                                                                      dest_shape, data_type));
  return Status::OK();
}

// Add operator related.

Status ConvOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                            const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  const auto& op_type = node.OpType();
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val output = emscripten::val::object();
  const auto& initializers(model_builder.GetInitializerTensors());

  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get input shape");
  std::vector<int64_t> weight_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[1], weight_shape, logger), "Cannot get weight shape");
  const auto& weight_name = input_defs[1]->Name();

  NodeAttrHelper helper(node);
  auto strides = helper.Get("strides", std::vector<int64_t>{1, 1});
  auto dilations = helper.Get("dilations", std::vector<int64_t>{1, 1});
  auto pads = helper.Get("pads", std::vector<int64_t>{0, 0, 0, 0});

  const bool is_nhwc = model_builder.GetPreferredLayout() == DataLayout::NHWC;
  const bool is_conv1d = input_shape.size() == 3 && weight_shape.size() == 3;
  const bool is_constant_weight = Contains(initializers, weight_name);
  // Support conv1d by prepending a 1 or 2 size dimensions.
  if (is_conv1d) {
    // Reshape input.
    if (is_nhwc) {
      // For NHWC preferred layout, the input has been transposed.
      // For conv1d it is NCD1 -> ND1C, so we need to prepend 1 to the index 2.
      input_shape.insert(input_shape.begin() + 2, 1);
    } else {
      input_shape.push_back(1);
    }
    std::vector<uint32_t> new_shape = GetVecUint32FromVecInt64(input_shape);
    input = model_builder.GetBuilder().call<emscripten::val>("reshape", input, emscripten::val::array(new_shape));

    weight_shape.resize(4, 1);  // Ensure 4D by appending 1's if needed.
    strides.resize(2, 1);       // Ensure 2D by appending 1's if needed.
    dilations.resize(2, 1);     // Ensure 2D by appending 1's if needed.
    if (pads.size() == 2) {
      pads.insert(pads.begin() + 1, 0);
      pads.push_back(0);
    }
  }

  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());
  ORT_RETURN_IF_ERROR(SetConvBaseOptions(
      model_builder, node, options, input_shape, weight_shape, strides, dilations, pads, is_nhwc, is_conv1d, logger));
  bool depthwise = false;
  if (op_type == "Conv" || op_type == "ConvInteger") {
    int groups = options["groups"].as<int>();
    if (is_nhwc) {
      depthwise = (groups == input_shape[3] && groups != 1);
      options.set("inputLayout", emscripten::val("nhwc"));
      if (is_constant_weight) {
        ORT_RETURN_IF_ERROR(AddInitializerInNewLayout(model_builder, weight_name, !depthwise, is_conv1d));
      }
      if (!depthwise) {
        options.set("filterLayout", emscripten::val("ohwi"));
      } else {
        options.set("filterLayout", emscripten::val("ihwo"));
      }
    }
  } else {  // ConvTranspose
    if (is_nhwc) {
      options.set("inputLayout", emscripten::val("nhwc"));
      options.set("filterLayout", emscripten::val("ohwi"));
      if (is_constant_weight) {
        ORT_RETURN_IF_ERROR(AddInitializerInNewLayout(model_builder, weight_name, true, is_conv1d));
      }
    }
  }

  emscripten::val filter = model_builder.GetOperand(weight_name);

  if (is_conv1d) {
    // Reshape weight to 4D for conv1d.
    if (!is_nhwc || !is_constant_weight) {
      // The weight_shape has been appended 1's, reshape weight operand.
      std::vector<uint32_t> new_shape = GetVecUint32FromVecInt64(weight_shape);
      emscripten::val reshape_options = emscripten::val::object();
      reshape_options.set("label", node.Name() + "_reshape_filter");
      filter = model_builder.GetBuilder().call<emscripten::val>("reshape",
                                                                filter,
                                                                emscripten::val::array(new_shape),
                                                                reshape_options);
    }
  }

  emscripten::val transpose_options = emscripten::val::object();
  if (is_nhwc && !is_constant_weight) {
    // For NHWC preferred layout, if the weight is input:
    // - Transpose it from iohw -> ohwi for convTranspose.
    // - Transpose it from oihw -> ihwo for depthwise conv.
    // - Transpose it from oihw -> ohwi for conv.
    std::vector<uint32_t> perm(4);
    if (op_type == "ConvTranspose" || depthwise) {
      perm = {1, 2, 3, 0};  // L_1230 for depthwise conv and convTranspose weight
    } else {
      perm = {0, 2, 3, 1};  // L_0231
    }
    transpose_options.set("permutation", emscripten::val::array(perm));
    transpose_options.set("label", node.Name() + "_transpose_filter");
    filter = model_builder.GetBuilder().call<emscripten::val>("transpose", filter, transpose_options);
  }

  if (op_type == "Conv") {
    output = model_builder.GetBuilder().call<emscripten::val>("conv2d", input, filter, options);
  } else if (op_type == "ConvInteger") {
    emscripten::val x_zero_point = emscripten::val::null();
    emscripten::val w_zero_point = emscripten::val::null();
    if (input_defs.size() >= 3) {
      x_zero_point = model_builder.GetOperand(node.InputDefs()[2]->Name());
    } else {
      x_zero_point = model_builder.GetZeroConstant("uint8");
    }
    if (input_defs.size() >= 4) {
      w_zero_point = model_builder.GetOperand(node.InputDefs()[3]->Name());
    } else {
      w_zero_point = model_builder.GetZeroConstant("uint8");
    }
    output = model_builder.GetBuilder().call<emscripten::val>("conv2dInteger",
                                                              input, x_zero_point, filter, w_zero_point, options);
  } else {
    output = model_builder.GetBuilder().call<emscripten::val>("convTranspose2d", input, filter, options);
  }

  // If it's a conv1d, reshape it back.
  if (is_conv1d) {
    const auto& output_defs = node.OutputDefs();
    std::vector<int64_t> output_shape;
    ORT_RETURN_IF_NOT(GetShape(*output_defs[0], output_shape, logger), "Cannot get output shape");
    std::vector<uint32_t> new_shape = GetVecUint32FromVecInt64(output_shape);
    emscripten::val reshape_options = emscripten::val::object();
    reshape_options.set("label", node.Name() + "_reshape_output");
    output = model_builder.GetBuilder().call<emscripten::val>("reshape",
                                                              output,
                                                              emscripten::val::array(new_shape),
                                                              reshape_options);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool ConvOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers,
                                      const Node& node,
                                      const WebnnDeviceType device_type,
                                      const logging::Logger& logger) const {
  const auto& name = node.Name();
  const auto& op_type = node.OpType();
  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger)) {
    LOGS(logger, VERBOSE) << "Cannot get input's shape.";
    return false;
  }

  const auto input_size = input_shape.size();
  if (input_size != 4 && input_size != 3) {
    LOGS(logger, VERBOSE) << op_type << " [" << name << "]'s input dimension: " << input_size
                          << ". Only conv 1d / 2d is supported.";
    return false;
  }

  std::vector<int64_t> weight_shape;
  if (!GetShape(*input_defs[1], weight_shape, logger)) {
    LOGS(logger, VERBOSE) << "Cannot get weight's shape.";
    return false;
  }

  const auto weight_size = weight_shape.size();
  if (weight_size != 4 && weight_size != 3) {
    LOGS(logger, VERBOSE) << op_type << " [" << name << "]'s weight dimension: " << weight_size
                          << ". Only conv 1d / 2d is supported.";
    return false;
  }

  // WebNN CPU backend (TFLite) only supports default dilations and group.
  // https://source.chromium.org/chromium/chromium/src/+/main:services/webnn/tflite/graph_builder_tflite.cc;l=1040
  if (device_type == WebnnDeviceType::CPU && op_type == "ConvTranspose") {
    NodeAttrHelper helper(node);
    const auto dilations = helper.Get("dilations", std::vector<int64_t>{1, 1});
    const auto group = helper.Get("group", 1);
    if (dilations[0] != 1 || (dilations.size() > 1 && dilations[1] != 1)) {
      LOGS(logger, VERBOSE) << op_type << " for WebNN CPU backend only supports default dilation 1.";
      return false;
    }
    if (group != 1) {
      LOGS(logger, VERBOSE) << op_type << " for WebNN CPU backend only supports default group 1.";
      return false;
    }
  }

  return true;
}

bool ConvOpBuilder::HasSupportedInputsImpl(const Node& node, const emscripten::val& wnn_limits,
                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  int32_t input0_type;  // input data type
  int32_t input1_type;  // weight data type
  int32_t input2_type;  // bias or x_zero_point data type
  int32_t input3_type;  // w_zero_point data type
  bool has_input2 = input_defs.size() > 2 && input_defs[2]->Exists();
  bool has_input3 = input_defs.size() > 3 && input_defs[3]->Exists();

  if (!GetType(*input_defs[0], input0_type, logger) ||
      !GetType(*input_defs[1], input1_type, logger) ||
      (has_input2 && !GetType(*input_defs[2], input2_type, logger)) ||
      (has_input3 && !GetType(*input_defs[3], input3_type, logger))) {
    return false;
  }

  std::string webnn_op_type;
  if (!GetWebNNOpType(op_type, webnn_op_type))
    return false;

  if (!IsSupportedDataType(input0_type, wnn_limits[webnn_op_type]["input"]["dataTypes"])) {
    LOGS(logger, VERBOSE) << "[" << op_type
                          << "] Input type: [" << input0_type
                          << "] is not supported for now";
    return false;
  }

  if (input0_type != input1_type ||
      (has_input2 && input0_type != input2_type) ||
      (has_input3 && input0_type != input3_type)) {
    LOGS(logger, VERBOSE) << "[" << op_type
                          << "] Input data types should be the same.";
    return false;
  }

  return true;
}

void CreateConvOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  static std::vector<std::string> op_types =
      {
          "Conv",
          "ConvInteger",
          "ConvTranspose",
      };

  op_registrations.builders.push_back(std::make_unique<ConvOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
