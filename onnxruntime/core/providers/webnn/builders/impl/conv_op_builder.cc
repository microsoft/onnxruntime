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
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

void ConvOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // skip the weight for conv as we need to transpose for preferred layout NHWC.
  if (model_builder.GetPreferredLayout() == DataLayout::NHWC) {
    model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());  // W
    model_builder.AddInputToSkip(node.InputDefs()[1]->Name());
  }
}

// Helper functions
common::Status SetConvBaseOptions(ModelBuilder& model_builder,
                                  const Node& node, emscripten::val& options,
                                  const std::vector<int32_t>& strides,
                                  const std::vector<int32_t>& dilations,
                                  std::vector<int32_t>& pads,
                                  const logging::Logger& logger) {
  NodeAttrHelper helper(node);
  const auto group = helper.Get("group", static_cast<int32_t>(1));
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> weight_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[1], weight_shape, logger), "Cannot get weight shape");
  options.set("strides", emscripten::val::array(strides));
  options.set("dilations", emscripten::val::array(dilations));
  options.set("groups", group);
  // Add Padding.
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
  AutoPadType auto_pad_type = StringToAutoPadType(helper.Get("auto_pad", "NOTSET"));
  if (AutoPadType::SAME_UPPER == auto_pad_type || AutoPadType::SAME_LOWER == auto_pad_type) {
    std::vector<int64_t> pads_out;
    ORT_RETURN_IF_ERROR(HandleAutoPad(input_shape, weight_shape[2], weight_shape[3],
                                      helper.Get("pads", std::vector<int64_t>{0, 0, 0, 0}),
                                      helper.Get("strides", std::vector<int64_t>{1, 1}),
                                      helper.Get("dilations", std::vector<int64_t>{1, 1}),
                                      auto_pad_type,
                                      pads_out,
                                      model_builder.GetPreferredLayout() == DataLayout::NCHW));
    std::transform(pads_out.begin(), pads_out.end(), pads.begin(),
                   [](int64_t pad) -> int32_t { return static_cast<int32_t>(pad); });
  }
  // Permute the ONNX's pads, which is [beginning_height, beginning_width, ending_height, ending_width],
  // while WebNN's padding is [beginning_height, ending_height, beginning_width, ending_width].
  const std::vector<int32_t> padding{pads[0], pads[2], pads[1], pads[3]};
  options.set("padding", emscripten::val::array(padding));

  // Add bias if present.
  if (input_defs.size() > 2) {
    options.set("bias", model_builder.GetOperand(input_defs[2]->Name()));
  }
  InlinedHashSet<std::string> supported_nodes{"Clip", "Relu"};
  emscripten::val activation = model_builder.FindActivation(node, *node.OutputDefs()[0], supported_nodes);
  if (emscripten::val::null() != activation) {
    options.set("activation", activation);
  }

  return Status::OK();
}

// Both depthwise Conv and ConvTranspose share the same logic to add the layout.
Status AddInitializerInNewLayout(ModelBuilder& model_builder,
                                 const std::string& name,
                                 bool is_conv) {
  const auto& tensor = *model_builder.GetInitializerTensors().at(name);
  auto data_type = tensor.data_type();
  if (!IsSupportedDataType(data_type, model_builder.GetWebnnDeviceType())) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "The initializer of graph has unsupported type, name: ",
                           tensor.name(), " type: ", data_type);
  }

  const auto& shape = tensor.dims();
  std::vector<uint32_t> dims;
  std::transform(shape.cbegin(), shape.cend(),
                 std::back_inserter(dims),
                 [](int64_t dim) -> int32_t { return SafeInt<int32_t>(dim); });

  ORT_RETURN_IF_NOT(dims.size() == 4,
                    "The initializer is not 4D: ", name, " actual dim ", dims.size());
  const uint8_t* src = nullptr;
  Initializer unpacked_tensor(tensor, model_builder.GetGraphViewer().ModelPath());
  src = unpacked_tensor.DataAsByteSpan().data();
  const auto out_t = dims[0], in_t = dims[1],
             h_t = dims[2], w_t = dims[3];
  std::vector<uint32_t> dest_shape;
  if (is_conv == 1)
    dest_shape = {out_t, h_t, w_t, in_t};  // L_0231
  else
    dest_shape = {in_t, h_t, w_t, out_t};  // L_1230 for depthwise conv weight

  SafeInt<size_t> num_elements = SafeInt<size_t>(Product(dest_shape));

  size_t element_size{0};
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      element_size = sizeof(uint16_t);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      element_size = sizeof(float);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      element_size = sizeof(int32_t);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      element_size = sizeof(int64_t);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      element_size = sizeof(uint32_t);
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      element_size = sizeof(uint64_t);
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

  NodeAttrHelper helper(node);
  const auto strides = helper.Get("strides", std::vector<int32_t>{1, 1});
  const auto dilations = helper.Get("dilations", std::vector<int32_t>{1, 1});
  auto pads = helper.Get("pads", std::vector<int32_t>{0, 0, 0, 0});
  const auto& weight = input_defs[1]->Name();
  if (op_type == "Conv") {
    emscripten::val options = emscripten::val::object();
    ORT_RETURN_IF_ERROR(SetConvBaseOptions(model_builder, node, options, strides, dilations, pads, logger));
    int groups = options["groups"].as<int>();
    std::vector<int64_t> input_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
    if (model_builder.GetPreferredLayout() == DataLayout::NHWC) {
      bool depthwise = (groups == input_shape[3] && groups != 1);
      options.set("inputLayout", emscripten::val("nhwc"));
      ORT_RETURN_IF_ERROR(AddInitializerInNewLayout(model_builder, weight, !depthwise));
      if (!depthwise) {
        options.set("filterLayout", emscripten::val("ohwi"));
      } else {
        options.set("filterLayout", emscripten::val("ihwo"));
      }
    }
    emscripten::val filter = model_builder.GetOperand(input_defs[1]->Name());

    output = model_builder.GetBuilder().call<emscripten::val>("conv2d", input, filter, options);
  } else {
    emscripten::val options = emscripten::val::object();
    ORT_RETURN_IF_ERROR(SetConvBaseOptions(model_builder, node, options, strides, dilations, pads, logger));
    if (model_builder.GetPreferredLayout() == DataLayout::NHWC) {
      options.set("inputLayout", emscripten::val("nhwc"));
      options.set("filterLayout", emscripten::val("ohwi"));
      ORT_RETURN_IF_ERROR(AddInitializerInNewLayout(model_builder, weight, false));
    }

    // When the 'output_shape' is specificed, the 'output_padding' values
    // in options.outputPadding are ignored.
    std::vector<int32_t> dim;
    std::vector<int32_t> output_padding{0, 0};
    if (helper.HasAttr("output_shape")) {
      // Default value of 'output_shape' will be ignore as we already check if
      // it's existed.
      dim = helper.Get("output_shape", std::vector<int32_t>{-1, -1});
      // Extract the height and width.
      std::vector<int32_t> output_shape;
      if (dim.size() == 2) {
        output_shape = dim;
      } else if (dim.size() == 4) {
        output_shape = {dim[2], dim[3]};
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid output shape");
      }
      // Padding values are auto generated.
      if (helper.HasAttr("kernel_shape")) {
        std::vector<int32_t> kernel_shape = helper.Get("kernel_shape", std::vector<int32_t>{-1, -1});
        std::vector<int32_t> total_padding(2);
        std::vector<int64_t> input_shape;
        ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
        for (size_t i = 0; i < 2; i++) {
          // Get the dimensions of H and W.
          // For NHWC layout, the dimensions of H and W correspond to index 1 and 2.
          // For NCHW layout, the dimensions of H and W correspond to index 2 and 3.
          if (model_builder.GetPreferredLayout() == DataLayout::NHWC) {
            total_padding[i] = strides[i] * (narrow<size_t>(input_shape[i + 1]) - 1) +
                               output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i];
          } else {
            ORT_RETURN_IF_NOT(model_builder.GetPreferredLayout() == DataLayout::NCHW,
                              "WebNN GPU backend preferred layout should be NCHW.");
            total_padding[i] = strides[i] * (narrow<size_t>(input_shape[i + 2]) - 1) +
                               output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i];
          }
        }
        AutoPadType auto_pad_type = StringToAutoPadType(helper.Get("auto_pad", "NOTSET"));
        if (AutoPadType::SAME_UPPER == auto_pad_type || AutoPadType::SAME_LOWER == auto_pad_type) {
          pads[0] = total_padding[0] / 2;
          pads[1] = total_padding[0] - pads[0];
          pads[2] = total_padding[1] / 2;
          pads[3] = total_padding[1] - pads[2];
          if (AutoPadType::SAME_LOWER == auto_pad_type) {
            std::swap(pads[0], pads[1]);
            std::swap(pads[2], pads[3]);
          }
          options.set("padding", emscripten::val::array(pads));
        }

      }
      options.set("outputSizes", emscripten::val::array(output_shape));
    } else {
      output_padding = helper.Get("output_padding", std::vector<int32_t>{0, 0});
      options.set("outputPadding", emscripten::val::array(output_padding));
    }
    emscripten::val filter = model_builder.GetOperand(input_defs[1]->Name());
    output = model_builder.GetBuilder().call<emscripten::val>("convTranspose2d", input, filter, options);
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

  const auto& weight_name = input_defs[1]->Name();
  // WebNN CPU backend (XNNPACK) requires the filter operand to be a constant.
  // https://github.com/google/XNNPACK/blob/master/src/subgraph/convolution-2d.c#L739
  if (device_type == WebnnDeviceType::CPU) {
    if (Contains(initializers, weight_name)) {
      const auto& tensor = *initializers.at(weight_name);
      if (tensor.dims().size() != 4) {
        LOGS(logger, VERBOSE) << op_type << " [" << name << "] dimension: " << tensor.dims().size()
                              << " Only conv 2d is supported.";
        return false;
      }
    } else {
      LOGS(logger, VERBOSE) << "The weight of " << op_type << " [" << name << "] must be known";
      return false;
    }
  }
  return true;
}

void CreateConvOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  static std::vector<std::string> op_types =
      {
          "Conv",
          "ConvTranspose",
      };

  op_registrations.builders.push_back(std::make_unique<ConvOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
