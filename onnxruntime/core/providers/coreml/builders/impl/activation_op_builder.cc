// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace coreml {

class ActivationOpBuilder : public BaseOpBuilder {
 private:
  int GetMinSupportedOpSet(const Node& node) const override;
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) const override ORT_MUST_USE_RESULT;
};

int ActivationOpBuilder::GetMinSupportedOpSet(const Node& /* node */) const {
  // All ops opset 5- uses consumed_inputs attribute which is not supported for now
  return 6;
}

Status ActivationOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) const {
  const auto& op_type(node.OpType());

  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = std::make_unique<COREML_SPEC::NeuralNetworkLayer>();
  layer->set_name(node.Name());

  if (op_type == "Sigmoid") {
    layer->mutable_activation()->mutable_sigmoid();
  } else if (op_type == "Tanh") {
    layer->mutable_activation()->mutable_tanh();
  } else if (op_type == "Relu") {
    layer->mutable_activation()->mutable_relu();
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "ActivationOpBuilder::AddToModelBuilderImpl, unknown op: ", op_type);
  }

  *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
  *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

  model_builder.AddLayer(layer.release());
  return Status::OK();
}

void CreateActivationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.find(op_type) != op_registrations.op_builder_map.cend())
    return;

  static std::vector<std::string> op_types =
      {
          "Sigmoid",
          "Tanh",
          "Relu",
      };

  op_registrations.builders.push_back(onnxruntime::make_unique<ActivationOpBuilder>());
  for (const auto& op_type : op_types) {
    op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
  }
}

}  // namespace coreml
}  // namespace onnxruntime