// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/cpu/tensor/reshape_helper.h"
#include "core/providers/shared/utils/utils.h"

#ifdef __APPLE__
#include "core/providers/coreml/builders/model_builder.h"
#endif

namespace onnxruntime {
namespace coreml {

class ReshapeOpBuilder : public BaseOpBuilder {
  // Add operator related
#ifdef __APPLE__
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;
#endif

  // Operator support related
 private:
  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  // Reshape opset 4- uses attributes for new shape which we do not support for now
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 5; }
};

// Add operator related

#ifdef __APPLE__
void ReshapeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());
}

Status ReshapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& logger) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(model_builder, node);

  const auto& input_defs = node.InputDefs();
  const auto& initializers(model_builder.GetInitializerTensors());
  const auto& target_shape_tensor = *initializers.at(input_defs[1]->Name());
  const int64_t* raw_target_shape = target_shape_tensor.int64_data().empty()
                                        ? reinterpret_cast<const int64_t*>(target_shape_tensor.raw_data().data())
                                        : target_shape_tensor.int64_data().data();

  const auto size = target_shape_tensor.dims()[0];
  TensorShapeVector target_shape{raw_target_shape, raw_target_shape + size};
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetStaticShape(*input_defs[0], input_shape, logger), "Cannot get shape");
  ReshapeHelper helper(TensorShape(input_shape), target_shape);
  *layer->mutable_reshapestatic()->mutable_targetshape() = {target_shape.cbegin(), target_shape.cend()};
  *layer->mutable_input()->Add() = input_defs[0]->Name();
  *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

  model_builder.AddLayer(std::move(layer));
  return Status::OK();
}
#endif

// Operator support related

bool ReshapeOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                         const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& new_shape_name = input_defs[1]->Name();
  const auto& initializers = input_params.graph_viewer.GetAllInitializedTensors();
  if (!Contains(initializers, new_shape_name)) {
    LOGS(logger, VERBOSE) << "New shape of reshape must be a constant initializer";
    return false;
  }

  const auto& new_shape_tensor = *initializers.at(new_shape_name);
  Initializer unpacked_tensor(new_shape_tensor);
  auto new_shape = unpacked_tensor.DataAsSpan<int64_t>();
  if (new_shape.empty()) {
    LOGS(logger, VERBOSE) << "New shape of reshape cannot be empty";
    return false;
  }

  std::vector<int64_t> input_shape;
  if (!GetStaticShape(*input_defs[0], input_shape, logger))
    return false;

  if (input_shape.empty()) {
    LOGS(logger, VERBOSE) << "Reshape does not support empty input shape";
    return false;
  }

  // CoreML reshape doesn't support new shape with more than 5 dimensions
  if (new_shape.size() > 5) {
    LOGS(logger, VERBOSE) << "Reshape does not support new shape with rank greater than 5. Input shape: "
                          << Shape2String(input_shape) << ", new shape: " << Shape2String(new_shape);
    return false;
  }

  // CoreML reshape does not support 0 as dimension
  NodeAttrHelper helper(node);
  const bool allow_zero = helper.Get("allowzero ", 0) == 1;
  if (allow_zero) {
    if (std::find(new_shape.begin(), new_shape.end(), int64_t{0}) != new_shape.end()) {
      LOGS(logger, VERBOSE) << "Reshape does not support new shape with 0 as dimension when allowzero is enabled. "
                               "Input shape: "
                            << Shape2String(input_shape) << ", new shape: " << Shape2String(new_shape);
      return false;
    }
  }

  return true;
}

void CreateReshapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ReshapeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
