// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"

namespace onnxruntime {
namespace coreml {

class ExpandOpBuilder : public BaseOpBuilder {
 public:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true;}
};

Status ExpandOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                const Node& node,
                                                const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  ORT_RETURN_IF_NOT(input_defs.size() == 2, "Expand requires exactly two inputs.");

  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetStaticShape(*input_defs[0], input_shape, logger), "Cannot get shape of data");

  const auto& data_name = input_defs[0]->Name();
  const auto& shape_name = input_defs[1]->Name();
  Initializer unpacked_tensor(*model_builder.GetConstantInitializer(shape_name));
  TensorShapeVector shape = ToShapeVector(unpacked_tensor.DataAsSpan<int64_t>());

  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;

    std::unique_ptr<Operation> expand_op = model_builder.CreateOperation(node, "expand");
    AddOperationInput(*expand_op, "x", data_name);
    AddOperationInput(*expand_op, "shape", model_builder.AddConstant(expand_op->type(), "shape", shape));
    model_builder.AddOperation(std::move(expand_op));
  } else {
    // For non-ML Program mode, we create a simple layer
    auto layer = model_builder.CreateNNLayer(node);
    *layer->mutable_expand()->mutable_input()->Add() = data_name;
    *layer->mutable_expand()->mutable_shape()->Add() = shape_name;
    *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();
    model_builder.AddLayer(std::move(layer));
  }

  return Status::OK();
}


void CreateExpandOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ExpandOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}
}
