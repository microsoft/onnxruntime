// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

class SoftmaxOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
  bool SupportsMLProgram() const override { return true; }
};

Status SoftmaxOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& logger) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = model_builder.CreateNNLayer(node);
  const auto& input_name = node.InputDefs()[0]->Name();
  const auto& output_name = node.OutputDefs()[0]->Name();

  std::vector<int64_t> data_shape;
  ORT_RETURN_IF_NOT(GetStaticShape(*node.InputDefs()[0], data_shape, logger), "Failed to get input shape.");

  NodeAttrHelper helper(node);
  int32_t axis_default_value = (node.SinceVersion() < 13) ? 1 : -1;
  const auto axis = helper.Get("axis", axis_default_value);
  auto axis_nonnegative = HandleNegativeAxis(axis, data_shape.size());

#if defined(COREML_ENABLE_MLPROGRAM)
  // CoreML's softmax match onnx's softmax behavior since opset 13.
  // For opset < 13, we need to reshape to 2D and set axis to -1 to simulate onnx softmax behavior.
  // [B,D,...](onnx softmax opset 12, axis=1)->[B,D*...](CoreML softmax, axis=-1)->[B,D,...](reshape back)
  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;
    auto input_dtype = node.InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    const int32_t elem_type = static_cast<int32_t>(input_dtype);

    std::string_view layer_input_name_x = node.InputDefs()[0]->Name();
    const bool need_reshape = node.SinceVersion() < 13 && axis_nonnegative != static_cast<int64_t>(data_shape.size()) - 1;
    std::vector<int64_t> target_shape;
    if (need_reshape) {
      // reshape to 2D to simulate onnx softmax behavior
      auto reshape1 = model_builder.CreateOperation(node, "reshape", "pre");
      TensorShape input_shape(data_shape);
      target_shape.push_back(input_shape.SizeToDimension(axis_nonnegative));
      target_shape.push_back(input_shape.SizeFromDimension(axis_nonnegative));
      axis_nonnegative = 1;
      AddOperationInput(*reshape1, "x", layer_input_name_x);
      AddOperationInput(*reshape1, "shape", model_builder.AddConstant(reshape1->type(), "shape1", target_shape));
      layer_input_name_x = model_builder.GetUniqueName(node, "ln_reshape1_");
      AddIntermediateOperationOutput(*reshape1, layer_input_name_x, elem_type, target_shape);
      model_builder.AddOperation(std::move(reshape1));
    }
    std::unique_ptr<Operation> op = model_builder.CreateOperation(node, "softmax");
    AddOperationInput(*op, "x", layer_input_name_x);
    AddOperationInput(*op, "axis", model_builder.AddScalarConstant(op->type(), "axis", axis_nonnegative));
    if (!need_reshape) {
      AddOperationOutput(*op, *node.OutputDefs()[0]);
      model_builder.AddOperation(std::move(op));
    } else {
      std::string_view ln_output_name = model_builder.GetUniqueName(node, "ln_reshape1_");
      AddIntermediateOperationOutput(*op, ln_output_name, elem_type, target_shape);
      model_builder.AddOperation(std::move(op));
      auto reshape2 = model_builder.CreateOperation(node, "reshape", "post");
      AddOperationInput(*reshape2, "x", ln_output_name);
      AddOperationInput(*reshape2, "shape", model_builder.AddConstant(reshape2->type(), "shape2", data_shape));
      AddOperationOutput(*reshape2, *node.OutputDefs()[0]);
      model_builder.AddOperation(std::move(reshape2));
    }
  } else  // NOLINT
#endif
  {
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

      TensorShapeVector target_shape;
      target_shape.push_back(size_to_dimension);
      target_shape.push_back(size_from_dimension);

      const auto reshape1_output_name = model_builder.GetUniqueName(node, "reshape1_output");
      {  // Add reshape layer
        auto reshape_layer = model_builder.CreateNNLayer(node, "_Softmax_reshape1");
        *reshape_layer->mutable_reshapestatic()->mutable_targetshape() = {target_shape.cbegin(), target_shape.cend()};
        *reshape_layer->mutable_input()->Add() = input_name;
        *reshape_layer->mutable_output()->Add() = reshape1_output_name;
        model_builder.AddLayer(std::move(reshape_layer));
      }
      const auto softmax_output_name = model_builder.GetUniqueName(node, "softmax_output");
      {
        auto* coreml_softmaxnd = layer->mutable_softmaxnd();
        coreml_softmaxnd->set_axis(-1);
        *layer->mutable_input()->Add() = reshape1_output_name;
        *layer->mutable_output()->Add() = softmax_output_name;
        model_builder.AddLayer(std::move(layer));
      }
      {
        // Add reshape back layer
        auto reshape_layer = model_builder.CreateNNLayer(node, "_Softmax_reshape2");
        *reshape_layer->mutable_reshapestatic()->mutable_targetshape() = {data_shape.cbegin(), data_shape.cend()};
        *reshape_layer->mutable_input()->Add() = softmax_output_name;
        *reshape_layer->mutable_output()->Add() = output_name;
        model_builder.AddLayer(std::move(reshape_layer));
      }
    }
  }

  return Status::OK();
}

bool SoftmaxOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& /*input_params*/,
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
