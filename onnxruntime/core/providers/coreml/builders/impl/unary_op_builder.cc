// Copyright (c) Shukant Pal.
// Licensed under the MIT License.

#include "core/providers/common.h"

#ifdef __APPLE__
#include "core/providers/coreml/builders/model_builder.h"
#endif
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace coreml {

class UnaryOpBuilder : public BaseOpBuilder {
 private:
#ifdef __APPLE__
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;
#endif
};

#ifdef __APPLE__
Status UnaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                             const logging::Logger& /* logger */) const {
  const auto& op_type(node.OpType());
  const auto& input_defs(node.InputDefs());

  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(model_builder, node);

  if (op_type == "Sqrt") {
    layer->mutable_unary()->set_type(COREML_SPEC::UnaryFunctionLayerParams::SQRT);
  } else if (op_type == "Reciprocal") {
    layer->mutable_unary()->set_type(COREML_SPEC::UnaryFunctionLayerParams::INVERSE);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "UnaryOpBuilder::AddToModelBuilderImpl, unknown op: ", op_type);
  }

  *layer->mutable_input()->Add() = input_defs[0]->Name();
  *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

  model_builder.AddLayer(std::move(layer));
  return Status::OK();
}
#endif

// Operator support related

void CreateUnaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<UnaryOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime