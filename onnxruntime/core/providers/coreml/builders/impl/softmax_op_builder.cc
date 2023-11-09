// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"

#ifdef __APPLE__
#include "core/providers/coreml/builders/model_builder.h"
#endif
#include "core/providers/coreml/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace coreml {

class SoftmaxOpBuilder : public BaseOpBuilder {
  // Add operator related
#ifdef __APPLE__
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;
#endif

};

// Add operator related

#ifdef __APPLE__

Status SoftmaxOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& /* logger */) const {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(model_builder, node);

  NodeAttrHelper helper(node);
  const auto axis = helper.Get("axis", -1);

  auto* coreml_softmax = layer->mutable_softmaxnd();
  coreml_softmax->set_axis(axis);

  *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
  *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

  model_builder.AddLayer(std::move(layer));
  return Status::OK();
}

#endif

void CreateSoftmaxOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<SoftmaxOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
