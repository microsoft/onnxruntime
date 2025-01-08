// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class ActivationOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         WebnnDeviceType device_type, const logging::Logger& logger) const override;
};

// Add operator related.

Status ActivationOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                  const Node& node,
                                                  const logging::Logger& /* logger */) const {
  const auto& op_type(node.OpType());
  emscripten::val input = model_builder.GetOperand(node.InputDefs()[0]->Name());
  emscripten::val output = emscripten::val::object();

  NodeAttrHelper helper(node);
  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());
  if (op_type == "Elu") {
    options.set("alpha", helper.Get("alpha", 1.0f));
    output = model_builder.GetBuilder().call<emscripten::val>("elu", input, options);
  } else if (op_type == "Gelu") {
    output = model_builder.GetBuilder().call<emscripten::val>("gelu", input, options);
  } else if (op_type == "HardSigmoid") {
    options.set("alpha", helper.Get("alpha", 0.2f));
    options.set("beta", helper.Get("beta", 0.5f));
    output = model_builder.GetBuilder().call<emscripten::val>("hardSigmoid", input, options);
  } else if (op_type == "HardSwish") {
    output = model_builder.GetBuilder().call<emscripten::val>("hardSwish", input, options);
  } else if (op_type == "LeakyRelu") {
    options.set("alpha", helper.Get("alpha", 0.0f));
    output = model_builder.GetBuilder().call<emscripten::val>("leakyRelu", input, options);
  } else if (op_type == "Relu") {
    output = model_builder.GetBuilder().call<emscripten::val>("relu", input, options);
  } else if (op_type == "Sigmoid") {
    output = model_builder.GetBuilder().call<emscripten::val>("sigmoid", input, options);
  } else if (op_type == "Softplus") {
    output = model_builder.GetBuilder().call<emscripten::val>("softplus", input, options);
  } else if (op_type == "Softsign") {
    output = model_builder.GetBuilder().call<emscripten::val>("softsign", input, options);
  } else if (op_type == "Tanh") {
    output = model_builder.GetBuilder().call<emscripten::val>("tanh", input, options);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "ActivationOpBuilder::AddToModelBuilderImpl, unknown op: ", op_type);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.
bool ActivationOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */,
                                            const Node& node,
                                            WebnnDeviceType device_type,
                                            const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& op_type = node.OpType();

  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  if (op_type == "Elu" && device_type == WebnnDeviceType::CPU) {
    NodeAttrHelper helper(node);
    float alpha = helper.Get("alpha", 1.0f);
    if (alpha != 1.0f) {
      LOGS(logger, VERBOSE) << "WebNN CPU backend only supports Elu's alpha == 1.0";
      return false;
    }
  }

  return true;
}

void CreateActivationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  static std::vector<std::string> op_types =
      {
          "Elu",
          "Gelu",
          "HardSigmoid",
          "HardSwish",
          "LeakyRelu",
          "Relu",
          "Sigmoid",
          "Softplus",
          "Softsign",
          "Tanh",
      };

  op_registrations.builders.push_back(std::make_unique<ActivationOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
