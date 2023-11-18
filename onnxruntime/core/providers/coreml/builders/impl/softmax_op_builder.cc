// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/impl/base_op_builder.h"

#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

#ifdef __APPLE__
#include "core/providers/coreml/builders/model_builder.h"
#endif
#include "core/providers/coreml/builders/op_builder_factory.h"

namespace onnxruntime {
namespace coreml {

class SoftmaxOpBuilder : public BaseOpBuilder {
  // Add operator related
#ifdef __APPLE__
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;
#endif

  // Operator support related
 private:
  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
};

// Add operator related

#ifdef __APPLE__

Status SoftmaxOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& logger) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(model_builder, node);
  const auto& input_name = node.InputDefs()[0]->Name();
  const auto& output_name = node.OutputDefs()[0]->Name();

  std::vector<int64_t> data_shape;
  ORT_RETURN_IF_NOT(GetStaticShape(*node.InputDefs()[0], data_shape, logger), "Failed to get input shape.");

  NodeAttrHelper helper(node);
  int32_t axis_default_value = (node.SinceVersion() < 13) ? 1 : -1;
  const auto axis = helper.Get("axis", axis_default_value);
  const auto axis_nonnegative = HandleNegativeAxis(axis, data_shape.size());

  if (node.SinceVersion() >= 13 || (data_shape.size() == 2)) {
    auto* coreml_softmaxnd = layer->mutable_softmaxnd();
    coreml_softmaxnd->set_axis(axis);
    *layer->mutable_input()->Add() = input_name;
    *layer->mutable_output()->Add() = output_name;
    model_builder.AddLayer(std::move(layer));
  } else {
    // note: if opsets < 13, onnx Softmax coerces the input shape to be 2D based on axis.
    // we need to manually reshape to 2D and apply SoftmaxND to axis -1 to achieve equivalent results for CoreML.
    TensorShape input_shape(data_shape);
    const auto size_to_dimension = input_shape.SizeToDimension(axis_nonnegative);
    const auto size_from_dimension = input_shape.SizeFromDimension(axis_nonnegative);

    std::vector<int64_t> target_shape;
    target_shape.push_back(size_to_dimension);
    target_shape.push_back(size_from_dimension);

    const auto reshape1_output_name = model_builder.GetUniqueName(MakeString(node.Name(), "reshape1_output"));
    {  // Add reshape layer
      const auto softmax_reshape1_layer_name =
          model_builder.GetUniqueName(MakeString(node.Name(), "_Softmax_reshape1"));
      auto reshape_layer = CreateNNLayer(softmax_reshape1_layer_name);
      *reshape_layer->mutable_reshapestatic()->mutable_targetshape() = {target_shape.cbegin(), target_shape.cend()};
      *reshape_layer->mutable_input()->Add() = input_name;
      *reshape_layer->mutable_output()->Add() = reshape1_output_name;
      model_builder.AddLayer(std::move(reshape_layer));
    }
    const auto softmax_output_name = model_builder.GetUniqueName(MakeString(node.Name(), "softmax_output"));
    {
      auto* coreml_softmaxnd = layer->mutable_softmaxnd();
      coreml_softmaxnd->set_axis(-1);
      *layer->mutable_input()->Add() = reshape1_output_name;
      *layer->mutable_output()->Add() = softmax_output_name;
      model_builder.AddLayer(std::move(layer));
    }
    {
      // Add reshape back layer
      const auto softmax_reshape2_layer_name =
          model_builder.GetUniqueName(MakeString(node.Name(), "_Softmax_reshape2"));
      auto reshape_layer = CreateNNLayer(softmax_reshape2_layer_name);
      *reshape_layer->mutable_reshapestatic()->mutable_targetshape() = {data_shape.cbegin(), data_shape.cend()};
      *reshape_layer->mutable_input()->Add() = softmax_output_name;
      *reshape_layer->mutable_output()->Add() = output_name;
      model_builder.AddLayer(std::move(reshape_layer));
    }
  }

  return Status::OK();
}

#endif

// Operator support related

bool SoftmaxOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& /* input_params */,
                                         const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetStaticShape(*input_defs[0], input_shape, logger))
    return false;

  const TensorShape shape(input_shape);
  if (shape.Size() == 0) {
    LOGS(logger, VERBOSE) << "Empty input data is not supported.";
    return false;
  }

  return true;
}

void CreateSoftmaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SoftmaxOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
