// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"

namespace onnxruntime {
namespace coreml {

class IdentityOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

Status IdentityOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                                const logging::Logger& /*logger*/) const {
  const auto& input_defs = node.InputDefs();
  const auto& output_def = *node.OutputDefs()[0];

  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;
    auto op = model_builder.CreateOperation(node, "identity");
    AddOperationInput(*op, "x", input_defs[0]->Name());
    AddOperationOutput(*op, output_def);
    model_builder.AddOperation(std::move(op));
  } else {
    // NeuralNetwork: emulate via activation LINEAR(alpha=1, beta=0).
    auto layer = model_builder.CreateNNLayer(node);
    auto* linear = layer->mutable_activation()->mutable_linear();
    linear->set_alpha(1.0f);
    linear->set_beta(0.0f);
    *layer->mutable_input()->Add() = input_defs[0]->Name();
    *layer->mutable_output()->Add() = output_def.Name();
    model_builder.AddLayer(std::move(layer));
  }
  return Status::OK();
}

void CreateIdentityOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<IdentityOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
