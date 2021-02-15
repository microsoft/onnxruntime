// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/common.h>

#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/helper.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace coreml {

// Shared functions

// TODO, move this to shared_library
bool HasExternalInitializer(const InitializedTensorSet& initializers, const Node& node,
                            const logging::Logger& logger) {
  for (const auto* node_arg : node.InputDefs()) {
    const auto& input_name(node_arg->Name());
    if (!Contains(initializers, input_name))
      continue;

    const auto& tensor = *initializers.at(input_name);
    if (tensor.has_data_location() &&
        tensor.data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      LOGS(logger, VERBOSE) << "Initializer [" << input_name
                            << "] with external data location are not currently supported";
      return true;
    }
  }

  return false;
}

// Add operator related

Status BaseOpBuilder::AddToModelBuilder(ModelBuilder& model_builder, const Node& node,
                                        const logging::Logger& logger) const {
  ORT_RETURN_IF_NOT(
      IsOpSupported(model_builder.GetInitializerTensors(), node, logger),
      "Unsupported operator ",
      node.OpType());

  ORT_RETURN_IF_ERROR(AddToModelBuilderImpl(model_builder, node, logger));
  LOGS(logger, VERBOSE) << "Operator name: [" << node.Name()
                        << "] type: [" << node.OpType() << "] was added";
  return Status::OK();
}

/* static */ std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> BaseOpBuilder::CreateNNLayer(const Node& node) {
  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer =
      onnxruntime::make_unique<COREML_SPEC::NeuralNetworkLayer>();
  layer->set_name(node.Name());
  return layer;
}

// Operator support related

bool BaseOpBuilder::IsOpSupported(const InitializedTensorSet& initializers, const Node& node,
                                  const logging::Logger& logger) const {
  if (!HasSupportedInputs(node, logger))
    return false;

  // We do not support external initializers for now
  if (HasExternalInitializer(initializers, node, logger))
    return false;

  if (!HasSupportedOpSet(node, logger))
    return false;

  return IsOpSupportedImpl(initializers, node, logger);
}

bool BaseOpBuilder::HasSupportedInputs(const Node& node, const logging::Logger& logger) const {
  const auto node_name = MakeString("Node [", node.Name(), "] type [", node.OpType(), "]");
  for (const auto* input : node.InputDefs()) {
    if (!IsInputSupported(*input, node_name, logger)) {
      return false;
    }
  }

  return HasSupportedInputsImpl(node, logger);
}

bool BaseOpBuilder::HasSupportedInputsImpl(const Node& node, const logging::Logger& logger) const {
  // We only check the type of input 0 by default
  // specific op builder can override this
  const auto& input = *node.InputDefs()[0];

  int32_t input_type;
  if (!GetType(input, input_type, logger))
    return false;

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    LOGS(logger, VERBOSE) << "[" << node.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

bool BaseOpBuilder::HasSupportedOpSet(const Node& node,
                                      const logging::Logger& logger) const {
  auto since_version = node.SinceVersion();
  if (since_version < GetMinSupportedOpSet(node) || since_version > GetMaxSupportedOpSet(node)) {
    LOGS(logger, VERBOSE) << node.OpType() << "is only supported for opset ["
                          << GetMinSupportedOpSet(node) << ", "
                          << GetMaxSupportedOpSet(node) << "]";
    return false;
  }

  return true;
}

}  // namespace coreml
}  // namespace onnxruntime