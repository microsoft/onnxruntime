// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class ScatterNDOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
  bool IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
  bool HasSupportedInputsImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                              const emscripten::val& wnn_limits, const logging::Logger& logger) const override;
};

// Add operator related.

Status ScatterNDOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                                 const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  emscripten::val data = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val indices = model_builder.GetOperand(input_defs[1]->Name());
  emscripten::val updates = model_builder.GetOperand(input_defs[2]->Name());
  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());
  emscripten::val output =
      model_builder.GetBuilder().call<emscripten::val>("scatterND", data, indices, updates, options);

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

bool ScatterNDOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                                           const WebnnDeviceType /* device_type */,
                                           const logging::Logger& logger) const {
  NodeAttrHelper helper(node);
  if (helper.Get("reduction", "none") != "none") {
    LOGS(logger, VERBOSE) << "ScatterND: WebNN only supports reduction type none (default)";
    return false;
  }

  return true;
}

bool ScatterNDOpBuilder::HasSupportedInputsImpl(const InitializedTensorSet& /* initializers */, const Node& node,
                                                const emscripten::val& wnn_limits,
                                                const logging::Logger& logger) const {
  const auto& data = *node.InputDefs()[0];
  const auto& indices = *node.InputDefs()[1];
  const auto& updates = *node.InputDefs()[2];
  const auto& op_type = node.OpType();

  int32_t data_type;
  int32_t indices_type;
  int32_t updates_type;
  if (!GetType(data, data_type, logger) || !GetType(indices, indices_type, logger) ||
      !GetType(updates, updates_type, logger)) {
    return false;
  }

  if (data_type != updates_type) {
    return false;
  }

  return IsDataTypeSupportedByOp(op_type, data_type, wnn_limits, "input", "data", logger) &&
         IsDataTypeSupportedByOp(op_type, indices_type, wnn_limits, "indices", "indices", logger);
}

void CreateScatterNDOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ScatterNDOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
