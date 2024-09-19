// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime::coreml {

class GatherOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
};

namespace {
int64_t GetAxisAttribute(const Node& node) {
  NodeAttrHelper node_attr_helper{node};
  return node_attr_helper.Get("axis", int64_t{0});
}
}  // namespace

Status GatherOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                              const logging::Logger& /*logger*/) const {
  auto layer = model_builder.CreateNNLayer(node);
  layer->mutable_gather()->set_axis(GetAxisAttribute(node));
  *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();    // data
  *layer->mutable_input()->Add() = node.InputDefs()[1]->Name();    // indices
  *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();  // output
  model_builder.AddLayer(std::move(layer));
  return Status::OK();
}

bool GatherOpBuilder::HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& /*input_params*/,
                                             const logging::Logger& logger) const {
  int32_t input_type;
  if (!GetType(*node.InputDefs()[0], input_type, logger))
    return false;

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      input_type != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    LOGS(logger, VERBOSE) << "[" << node.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

bool GatherOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& /*input_params*/,
                                        const logging::Logger& logger) const {
  std::vector<int64_t> data_shape, indices_shape;
  if (!GetShape(*node.InputDefs()[0], data_shape, logger)) {
    LOGS(logger, VERBOSE) << "Failed to get 'data' shape";
    return false;
  }

  if (!GetShape(*node.InputDefs()[1], indices_shape, logger)) {
    LOGS(logger, VERBOSE) << "Failed to get 'indices' shape";
    return false;
  }

  // Don't allow scalar 'indices' input.
  // We convert scalar inputs to tensors with shape [1] before providing them to CoreML.
  // This modification changes the shape of the Gather output.
  if (indices_shape.empty()) {
    LOGS(logger, VERBOSE) << "Gather does not support scalar 'indices'";
    return false;
  }

  if (data_shape.size() + indices_shape.size() - 1 > 5) {
    LOGS(logger, VERBOSE) << "Gather does not support output with rank greater than 5";
    return false;
  }

  return true;
}

void CreateGatherOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<GatherOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace onnxruntime::coreml
