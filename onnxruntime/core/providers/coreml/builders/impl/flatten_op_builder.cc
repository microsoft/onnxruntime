// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

#ifdef __APPLE__
#include "core/providers/coreml/builders/model_builder.h"
#endif

namespace onnxruntime {
namespace coreml {

class FlattenOpBuilder : public BaseOpBuilder {
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

Status FlattenOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& logger) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(model_builder, node);

  // Note: ONNX Flatten corresponds to CoreML FlattenTo2DLayerParams
  auto* coreml_flatten = layer->mutable_flattento2d();

  NodeAttrHelper helper(node);
  const int64_t axis = helper.Get("axis", 1);
  coreml_flatten->set_axis(axis);

  *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
  *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

  model_builder.AddLayer(std::move(layer));

  return Status::OK();
}
#endif

// Operator support related

bool FlattenOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& /*input_params*/,
                                         const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  if (input_shape.empty()) {
    LOGS(logger, VERBOSE) << "Flatten does not support empty input shape";
    return false;
  }

  return true;
}

void CreateFlattenOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<FlattenOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
