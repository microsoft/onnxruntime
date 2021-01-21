// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/common.h>

#include "../model_builder.h"
#include "../helper.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace coreml {

// Shared functions

// TODO, move this to shared_library
bool HasExternalInitializer(const InitializedTensorSet& initializers, const Node& node) {
  for (const auto* node_arg : node.InputDefs()) {
    const auto& input_name(node_arg->Name());
    if (!Contains(initializers, input_name))
      continue;

    const auto& tensor = *initializers.at(input_name);
    if (tensor.has_data_location() &&
        tensor.data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      LOGS_DEFAULT(VERBOSE) << "Initializer [" << input_name
                            << "] with external data location are not currently supported";
      return true;
    }
  }

  return false;
}

// Add operator related

Status BaseOpBuilder::AddToModelBuilder(ModelBuilder& model_builder, const Node& node) const {
  ORT_RETURN_IF_NOT(
      IsOpSupported(model_builder.GetInitializerTensors(), node),
      "Unsupported operator ",
      node.OpType());

  ORT_RETURN_IF_ERROR(AddToModelBuilderImpl(model_builder, node));
  LOGS_DEFAULT(VERBOSE) << "Operator name: [" << node.Name()
                        << "] type: [" << node.OpType() << "] was added";
  return Status::OK();
}

// Operator support related

bool BaseOpBuilder::IsOpSupported(const InitializedTensorSet& initializers, const Node& node) const {
  if (!HasSupportedInputs(node))
    return false;

  // We do not support external initializers for now
  if (HasExternalInitializer(initializers, node))
    return false;

  if (!HasSupportedOpSet(node))
    return false;

  return IsOpSupportedImpl(initializers, node);
}

bool BaseOpBuilder::HasSupportedInputs(const Node& node) const {
  // We do not support unknown(null) input shape
  for (const auto* input : node.InputDefs()) {
    if (!input->Shape()) {
      LOGS_DEFAULT(VERBOSE) << "Node [" << node.Name() << "] type [" << node.OpType()
                            << "] Input [" << input->Name() << "] has no shape";
      return false;
    }
  }

  return HasSupportedInputsImpl(node);
}

bool BaseOpBuilder::HasSupportedInputsImpl(const Node& node) const {
  // We only check the type of input 0 by default
  // specific op builder can override this
  const auto& input = *node.InputDefs()[0];

  int32_t input_type;
  if (!GetType(input, input_type))
    return false;

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    LOGS_DEFAULT(VERBOSE) << "[" << node.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

bool BaseOpBuilder::HasSupportedOpSet(const Node& node) const {
  auto since_version = node.SinceVersion();
  if (since_version < GetMinSupportedOpSet(node) || since_version > GetMaxSupportedOpSet(node)) {
    LOGS_DEFAULT(VERBOSE) << node.OpType() << "is only supported for opset ["
                          << GetMinSupportedOpSet(node) << ", "
                          << GetMaxSupportedOpSet(node) << "]";
    return false;
  }

  return true;
}

}  // namespace coreml
}  // namespace onnxruntime