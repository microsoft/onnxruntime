// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/impl/base_op_builder.h"

#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"  // for NodeAttrHelper

#if defined(__APPLE__)
#include "core/providers/coreml/builders/model_builder.h"
#endif

namespace onnxruntime::coreml {

class ShapeOpBuilder : public BaseOpBuilder {
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
#if defined(__APPLE__)
Status ShapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                             const logging::Logger& logger) const {
  auto layer = CreateNNLayer(model_builder, node);
  layer->mutable_getshape();
  *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();
  *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();
  model_builder.AddLayer(std::move(layer));
  return Status::OK();
}
#endif  // defined(__APPLE__)

// Operator support related
bool ShapeOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& /*input_params*/,
                                       const logging::Logger& logger) const {
  NodeAttrHelper node_attr_helper{node};
  if (node_attr_helper.Get("start", 0) != 0) {
    LOGS(logger, VERBOSE) << "Shape does not support 'start' attribute with value other than 0";
    return false;
  }

  if (node_attr_helper.HasAttr("end")) {
    LOGS(logger, VERBOSE) << "Shape does not support 'end' attribute";
    return false;
  }

  return true;
}

void CreateShapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ShapeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace onnxruntime::coreml
