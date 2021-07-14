// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"

#include "base_op_builder.h"
#include "builder_utils.h"

namespace onnxruntime {
namespace coreml {

class ConvOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const Node& /* node */, const OpBuilderInputParams& /* input_params */,
                         const logging::Logger& /* logger */) const override;
};

// Add operator related

void ConvOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& input_defs = node.InputDefs();

  // skip the weight and bias (if has it) for conv as we will directly set those as part of the NN layer
  model_builder.AddInitializerToSkip(input_defs[1]->Name());  // w

  if (input_defs.size() > 2) {
    model_builder.AddInitializerToSkip(input_defs[2]->Name());  // b
  }
}

Status ConvOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                            const logging::Logger& logger) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(node);

  const auto& input_defs = node.InputDefs();
  const auto& output_defs = node.OutputDefs();
  const auto& input_name = input_defs[0]->Name();
  const auto& output_name = output_defs[0]->Name();

  const auto& weight_tensor = *model_builder.GetInitializerTensors().at(input_defs[1]->Name());
  const auto& weight_shape = weight_tensor.dims();

  const bool is_1d_conv = (weight_shape.size() == 3);

  NodeAttrHelper helper(node);
  const auto strides = helper.Get("strides", std::vector<int64_t>{1, 1});
  const auto dilations = helper.Get("dilations", std::vector<int64_t>{1, 1});
  const auto onnx_pads = helper.Get("pads", std::vector<int64_t>{0, 0, 0, 0});
  auto strides_prime = strides;
  auto dilations_prime = dilations;
  auto onnx_pads_prime = onnx_pads;
  if (is_1d_conv) {
    if (strides.size() < 2) {
      strides_prime.push_back(1);
    }
    if (dilations.size() < 2) {
      dilations_prime.push_back(1);
    }
    if (onnx_pads.size() < 4 && onnx_pads != std::vector<int64_t>{0, 0}) {
      onnx_pads_prime.insert(onnx_pads_prime.begin() + 1, 1);
      onnx_pads_prime.push_back(1);
    }
  }
  const auto group = helper.Get("group", static_cast<int64_t>(1));

  auto* coreml_conv = layer->mutable_convolution();

  std::string expand_output_name = model_builder.GetUniqueName(node.Name() + "_expandDims");
  auto weight_tensor_prime = weight_tensor;

  if (is_1d_conv) {
    std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> expand_layer = CreateNNLayer(node);

    expand_layer->mutable_expanddims()->add_axes(0);
    *expand_layer->mutable_input()->Add() = input_name;
    *expand_layer->mutable_output()->Add() = expand_output_name;
    model_builder.AddLayer(std::move(expand_layer));

    weight_tensor_prime.add_dims(1);
    // const bool is_1d_conv_updated = (weight_tensor_prime.dims().size() == 4);
    // if (is_1d_conv_updated) {
    //   LOGS(logger, VERBOSE) << "weight_tensor gets updated.";
    // }
    const auto& weight_shape_prime = weight_tensor_prime.dims();
    coreml_conv->set_outputchannels(weight_shape_prime[0]);  // M
    coreml_conv->set_kernelchannels(weight_shape_prime[1]);  // C/Group
    coreml_conv->add_kernelsize(weight_shape_prime[2]);      // H
    coreml_conv->add_kernelsize(weight_shape_prime[3]);      // W:1
    *coreml_conv->mutable_stride() = {strides_prime.cbegin(), strides_prime.cend()};
    *coreml_conv->mutable_dilationfactor() = {dilations_prime.cbegin(), dilations_prime.cend()};
  } else {
    coreml_conv->set_outputchannels(weight_shape[0]);  // M
    coreml_conv->set_kernelchannels(weight_shape[1]);  // C/Group
    coreml_conv->add_kernelsize(weight_shape[2]);      // H
    coreml_conv->add_kernelsize(weight_shape[3]);      // W
    *coreml_conv->mutable_stride() = {strides.cbegin(), strides.cend()};
    *coreml_conv->mutable_dilationfactor() = {dilations.cbegin(), dilations.cend()};
  }
  coreml_conv->set_ngroups(group);

  coreml_conv->set_isdeconvolution(false);

  // Add Padding
  // Usually using autopadding is more efficient than using explicit padding
  // Try to see if we can map explicit padding to auto padding
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");
  AutoPadType auto_pad_type;

  if (is_1d_conv) {
    ORT_RETURN_IF_ERROR(HandleAutoPad(input_shape, weight_shape[2], 1,
                                      onnx_pads_prime, strides_prime, dilations_prime,
                                      StringToAutoPadType(helper.Get("auto_pad", "NOTSET")),
                                      auto_pad_type));
  } else {
    ORT_RETURN_IF_ERROR(HandleAutoPad(input_shape, weight_shape[2], weight_shape[3],
                                      onnx_pads, strides, dilations,
                                      StringToAutoPadType(helper.Get("auto_pad", "NOTSET")),
                                      auto_pad_type));
  }

  if (AutoPadType::SAME_UPPER == auto_pad_type || AutoPadType::SAME_LOWER == auto_pad_type) {
    auto* padding_type = coreml_conv->mutable_same();
    if (AutoPadType::SAME_LOWER == auto_pad_type) {  // default is SAME_UPPER
      padding_type->set_asymmetrymode(COREML_SPEC::SamePadding_SamePaddingMode_TOP_LEFT_HEAVY);
    }
  } else {
    auto* padding_type = coreml_conv->mutable_valid();

    if (is_1d_conv) {
      if (AutoPadType::NOTSET == auto_pad_type && onnx_pads_prime != std::vector<int64_t>{0, 0, 0, 0}) {
        // NOTSET is adding the explicit padding to the ValidPadding.paddingAmounts
        auto* height_border = padding_type->mutable_paddingamounts()->add_borderamounts();
        height_border->set_startedgesize(onnx_pads_prime[0]);
        height_border->set_endedgesize(onnx_pads_prime[2]);
        auto* width_border = padding_type->mutable_paddingamounts()->add_borderamounts();
        width_border->set_startedgesize(onnx_pads_prime[1]);
        width_border->set_endedgesize(onnx_pads_prime[3]);
      }
    } else {
      if (AutoPadType::NOTSET == auto_pad_type && onnx_pads != std::vector<int64_t>{0, 0, 0, 0}) {
        auto* height_border = padding_type->mutable_paddingamounts()->add_borderamounts();
        height_border->set_startedgesize(onnx_pads[0]);
        height_border->set_endedgesize(onnx_pads[2]);
        auto* width_border = padding_type->mutable_paddingamounts()->add_borderamounts();
        width_border->set_startedgesize(onnx_pads[1]);
        width_border->set_endedgesize(onnx_pads[3]);
      }
    }
  }

  // Add weight
  if (is_1d_conv) {
    auto& weight_tensor_prime_ref = weight_tensor_prime;
    CreateCoreMLWeight(*coreml_conv->mutable_weights(), weight_tensor_prime_ref);
  } else {
    CreateCoreMLWeight(*coreml_conv->mutable_weights(), weight_tensor);
  }

  // Add bias if present
  if (input_defs.size() > 2) {
    coreml_conv->set_hasbias(true);
    const auto& bias_tensor = *model_builder.GetInitializerTensors().at(input_defs[2]->Name());
    CreateCoreMLWeight(*coreml_conv->mutable_bias(), bias_tensor);
  }

  if (is_1d_conv) {
    std::string conv_output_name = model_builder.GetUniqueName(node.Name() + "_conv_output");
    *layer->mutable_input()->Add() = expand_output_name;
    *layer->mutable_output()->Add() = conv_output_name;
    model_builder.AddLayer(std::move(layer));

    std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> squeeze_layer = CreateNNLayer(node);
    squeeze_layer->mutable_squeeze()->add_axes(0);
    *squeeze_layer->mutable_input()->Add() = conv_output_name;
    *squeeze_layer->mutable_output()->Add() = output_name;
    model_builder.AddLayer(std::move(squeeze_layer));
  } else {
    *layer->mutable_input()->Add() = input_name;
    *layer->mutable_output()->Add() = output_name;
    model_builder.AddLayer(std::move(layer));
  }

  return Status::OK();
}

// Operator support related

bool ConvOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                      const logging::Logger& logger) const {
  const auto& name = node.Name();
  const auto& input_defs = node.InputDefs();

  const auto& weight_name = input_defs[1]->Name();
  const auto& initializers = input_params.graph_viewer.GetAllInitializedTensors();
  if (Contains(initializers, weight_name)) {
    const auto& tensor = *initializers.at(weight_name);
    if (tensor.dims().size() != 4 && tensor.dims().size() != 3) {
      LOGS(logger, VERBOSE) << "Conv [" << name << "] dimension: " << tensor.dims().size()
                            << " Only conv 2d and conv 1d are supported.";
      return false;
    }
  } else {
    LOGS(logger, VERBOSE) << "The weight of Conv [" << name << "] must be known";
    return false;
  }

  if (input_defs.size() > 2) {
    const auto& bias_name = input_defs[2]->Name();
    if (!Contains(initializers, bias_name)) {
      LOGS(logger, VERBOSE) << "The bias of Conv [" << name << "] must be a constant initializer";
      return false;
    }
  }

  return true;
}

void CreateConvOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ConvOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
