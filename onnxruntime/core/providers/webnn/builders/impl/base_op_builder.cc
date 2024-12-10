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
// Add operator related.

Status BaseOpBuilder::AddToModelBuilder(ModelBuilder& model_builder, const Node& node,
                                        const logging::Logger& logger) const {
  ORT_RETURN_IF_NOT(
      IsOpSupported(model_builder.GetInitializerTensors(), node, model_builder.GetWebnnDeviceType(),
                    model_builder.GetOpSupportLimits(), logger),
      "Unsupported operator ", node.OpType());
  ORT_RETURN_IF_ERROR(AddToModelBuilderImpl(model_builder, node, logger));
  return Status::OK();
}

// Operator support related.

bool BaseOpBuilder::IsOpSupported(const InitializedTensorSet& initializers, const Node& node,
                                  const WebnnDeviceType device_type, const emscripten::val& wnn_limits,
                                  const logging::Logger& logger) const {
  if (!HasSupportedInputs(node, wnn_limits, logger))
    return false;

  if (!HasSupportedOutputs(node, wnn_limits, logger))
    return false;

  if (!HasSupportedOpSet(node, logger))
    return false;

  return IsOpSupportedImpl(initializers, node, device_type, logger);
}

bool BaseOpBuilder::HasSupportedInputs(const Node& node, const emscripten::val& wnn_limits,
                                       const logging::Logger& logger) const {
  const auto node_name = MakeString("Node [", node.Name(), "] type [", node.OpType(), "]");
  for (const auto* input : node.InputDefs()) {
    if (!IsTensorShapeSupported(*input, node_name, logger, allow_empty_tensor_as_input_)) {
      return false;
    }
  }

  return HasSupportedInputsImpl(node, wnn_limits, logger);
}

bool BaseOpBuilder::HasSupportedInputsImpl(const Node& node,
                                           const emscripten::val& wnn_limits,
                                           const logging::Logger& logger) const {
  // We only check the type of input 0 by default, specific op builder can override this.
  const auto& input = *node.InputDefs()[0];
  const auto& op_type = node.OpType();
  int32_t input_type;
  if (!GetType(input, input_type, logger))
    return false;

  return IsDataTypeSupportedByOp(op_type, input_type, wnn_limits, "input", "Input", logger);
}

bool BaseOpBuilder::HasSupportedOutputs(const Node& node, const emscripten::val& wnn_limits,
                                        const logging::Logger& logger) const {
  const auto node_name = MakeString("Node [", node.Name(), "] type [", node.OpType(), "]");
  for (const auto* output : node.OutputDefs()) {
    if (!IsTensorShapeSupported(*output, node_name, logger)) {
      return false;
    }
  }

  return HasSupportedOutputsImpl(node, wnn_limits, logger);
}

bool BaseOpBuilder::HasSupportedOutputsImpl(const Node& node,
                                            const emscripten::val& wnn_limits,
                                            const logging::Logger& logger) const {
  // We only check the type of output 0 by default, specific op builder can override this.
  const auto& output = *node.OutputDefs()[0];
  const auto& op_type = node.OpType();
  int32_t output_type;
  if (!GetType(output, output_type, logger))
    return false;

  return IsDataTypeSupportedByOp(op_type, output_type, wnn_limits, "output", "Output", logger);
}

bool BaseOpBuilder::HasSupportedOpSet(const Node& node,
                                      const logging::Logger& logger) const {
  auto since_version = node.SinceVersion();
  if (since_version < GetMinSupportedOpSet(node) || since_version > GetMaxSupportedOpSet(node)) {
    LOGS(logger, VERBOSE) << "Current opset since version of "
                          << node.OpType() << " is " << since_version
                          << ", WebNN EP only supports for its opset ["
                          << GetMinSupportedOpSet(node) << ", "
                          << GetMaxSupportedOpSet(node) << "]";
    return false;
  }

  return true;
}

}  // namespace webnn
}  // namespace onnxruntime
