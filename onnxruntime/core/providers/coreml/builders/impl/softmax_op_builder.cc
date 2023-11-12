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

  if (node.SinceVersion() >= 13 || (data_shape.size() == 2 && axis == 1)) {
    auto* coreml_softmaxnd = layer->mutable_softmaxnd();
    coreml_softmaxnd->set_axis(axis);
    *layer->mutable_input()->Add() = input_name;
    *layer->mutable_output()->Add() = output_name;
    model_builder.AddLayer(std::move(layer));
  } else {
    // note: if opsets < 13, onnx Softmax coerces the input shape to be 2D based on axis.
    // we need to manually reshape to make sure the rank is >=3 and get the right number of dims.
    const auto num_elements_from_axis = data_shape.size() - axis_nonnegative;
    if (num_elements_from_axis < 3) {
      const auto expand_output_name = model_builder.GetUniqueName(MakeString(node.Name(), "expand_output"));
      {  // Add expand layer
        const auto softmax_expand_layer_name =
            model_builder.GetUniqueName(MakeString(node.Name(), "_Softmax_expand"));
        auto expand_layer = CreateNNLayer(softmax_expand_layer_name);
        for (uint64_t i = 0; i < 3 - num_elements_from_axis; i++) {
          expand_layer->mutable_expanddims()->add_axes(-1-i);
        }
        *expand_layer->mutable_input()->Add() = input_name;
        *expand_layer->mutable_output()->Add() = expand_output_name;
        model_builder.AddLayer(std::move(expand_layer));
      }
      layer->mutable_softmax();
      *layer->mutable_input()->Add() = expand_output_name;
      *layer->mutable_output()->Add() = output_name;
      model_builder.AddLayer(std::move(layer));
    } else {
      layer->mutable_softmax();
      *layer->mutable_input()->Add() = input_name;
      *layer->mutable_output()->Add() = output_name;
      model_builder.AddLayer(std::move(layer));
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
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  const TensorShape shape(input_shape);
  if (shape.Size() == 0) {
    LOGS(logger, VERBOSE) << "Cases that input data being empty due to a dimension with value of 0 is not supported";
    return false;
  }

  NodeAttrHelper helper(node);
  int32_t axis_default_value = (node.SinceVersion() < 13) ? 1 : -1;
  const auto axis = helper.Get("axis", axis_default_value);
  if (node.SinceVersion() < 13 && !(input_shape.size() == 2 && axis == 1)) {
    if (input_shape.size() >= 4) {
      LOGS(logger, VERBOSE) << "Cases that Softmax with version 13- with > 4d input and is not supported. Current input rank: "
                            << input_shape.size();
      return false;
    }
  }

  return true;
}

void CreateSoftmaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SoftmaxOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
