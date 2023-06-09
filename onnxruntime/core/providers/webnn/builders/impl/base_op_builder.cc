// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/common.h>

#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/shared/utils/utils.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

// Shared functions.
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

// Add operator related.

Status BaseOpBuilder::AddToModelBuilder(ModelBuilder& model_builder, const Node& node,
                                        const logging::Logger& logger) const {
  ORT_RETURN_IF_NOT(
      IsOpSupported(model_builder.GetInitializerTensors(), node, model_builder.GetWebnnDeviceType(), logger),
      "Unsupported operator ",
      node.OpType());
  ORT_RETURN_IF_ERROR(AddToModelBuilderImpl(model_builder, node, logger));
  LOGS(logger, VERBOSE) << "Operator name: [" << node.Name()
                        << "] type: [" << node.OpType() << "] was added";
  return Status::OK();
}

// Operator support related.

bool BaseOpBuilder::IsOpSupported(const InitializedTensorSet& initializers, const Node& node,
                                  const WebnnDeviceType device_type, const logging::Logger& logger) const {
  if (!HasSupportedInputs(node, device_type, logger))
    return false;

  // We do not support external initializers for now.
  if (HasExternalInitializer(initializers, node, logger))
    return false;

  if (!HasSupportedOpSet(node, logger))
    return false;

  return IsOpSupportedImpl(initializers, node, device_type, logger);
}

bool BaseOpBuilder::HasSupportedInputs(const Node& node, const WebnnDeviceType device_type,
                                       const logging::Logger& logger) const {
  const auto node_name = MakeString("Node [", node.Name(), "] type [", node.OpType(), "]");
  for (const auto* input : node.InputDefs()) {
    if (!IsInputSupported(*input, node_name, logger)) {
      return false;
    }
  }

  return HasSupportedInputsImpl(node, device_type, logger);
}

bool BaseOpBuilder::HasSupportedInputsImpl(const Node& node,
                                           const WebnnDeviceType device_type,
                                           const logging::Logger& logger) const {
  // We only check the type of input 0 by default, specific op builder can override this.
  const auto& input = *node.InputDefs()[0];

  int32_t input_type;
  if (!GetType(input, input_type, logger))
    return false;

  if (!IsSupportedDataType(input_type, device_type)) {
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
    LOGS(logger, VERBOSE) << "Current opset is " << since_version << ", "
                          << node.OpType() << " is only supported for opset ["
                          << GetMinSupportedOpSet(node) << ", "
                          << GetMaxSupportedOpSet(node) << "]";
    return false;
  }

  return true;
}

}  // namespace webnn
}  // namespace onnxruntime
