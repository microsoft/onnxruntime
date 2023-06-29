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
 private:
  int GetMinSupportedOpSet(const Node& node) const override;
};

// Add operator related.

Status ActivationOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                  const Node& node,
                                                  const logging::Logger& /* logger */) const {
  const auto& op_type(node.OpType());
  emscripten::val input = model_builder.GetOperand(node.InputDefs()[0]->Name());
  emscripten::val output = emscripten::val::object();
  if (op_type == "Relu") {
    if (Contains(model_builder.GetFusedActivations(), node.InputDefs()[0]->Name())) {
      LOGS_DEFAULT(VERBOSE) << "Relu Node [" << node.Name() << "] fused";
      output = input;
    } else {
      output = model_builder.GetBuilder().call<emscripten::val>("relu", input);
    }
  } else if (op_type == "LeakyRelu") {
    if (Contains(model_builder.GetFusedActivations(), node.InputDefs()[0]->Name())) {
      LOGS_DEFAULT(VERBOSE) << "LeakyRelu Node [" << node.Name() << "] fused";
      output = input;
    } else {
      NodeAttrHelper helper(node);
      emscripten::val options = emscripten::val::object();
      options.set("alpha", helper.Get("alpha", (float)0.0));
      output = model_builder.GetBuilder().call<emscripten::val>("leakyRelu", input, options);
    }
  } else if (op_type == "Sigmoid") {
    if (Contains(model_builder.GetFusedActivations(), node.InputDefs()[0]->Name())) {
      LOGS_DEFAULT(VERBOSE) << "Sigmoid Node [" << node.Name() << "] fused";
      output = input;
    } else {
      output = model_builder.GetBuilder().call<emscripten::val>("sigmoid", input);
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "ActivationOpBuilder::AddToModelBuilderImpl, unknown op: ", op_type);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related.

int ActivationOpBuilder::GetMinSupportedOpSet(const Node& /* node */) const {
  // All ops opset 5- uses consumed_inputs attribute which is not supported for now.
  return 6;
}

void CreateActivationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.find(op_type) != op_registrations.op_builder_map.cend())
    return;

  static std::vector<std::string> op_types = {"Relu", "LeakyRelu", "Sigmoid"};

  op_registrations.builders.push_back(std::make_unique<ActivationOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
