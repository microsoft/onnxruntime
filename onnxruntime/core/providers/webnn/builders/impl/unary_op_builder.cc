// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/common.h>

#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"
#include "core/providers/webnn/builders/helper.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class UnaryOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;
};

// Add operator related.

Status UnaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                             const logging::Logger& /* logger */) const {
  const auto& op_type(node.OpType());

  emscripten::val input = model_builder.GetOperand(node.InputDefs()[0]->Name());
  emscripten::val output = emscripten::val::object();
  if (op_type == "Cos") {
    output = model_builder.GetBuilder().call<emscripten::val>("cos", input);
  } else if (op_type == "Erf") {
    output = model_builder.GetBuilder().call<emscripten::val>("erf", input);
  } else if (op_type == "Floor") {
    output = model_builder.GetBuilder().call<emscripten::val>("floor", input);
  } else if (op_type == "Not") {
    output = model_builder.GetBuilder().call<emscripten::val>("logicalNot", input);
  } else if (op_type == "Sin") {
    output = model_builder.GetBuilder().call<emscripten::val>("sin", input);
  } else if (op_type == "Sqrt") {
    output = model_builder.GetBuilder().call<emscripten::val>("sqrt", input);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "UnaryOpBuilder::AddToModelBuilderImpl, unknown op: ", op_type);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

void CreateUnaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.find(op_type) != op_registrations.op_builder_map.cend())
    return;

  static std::vector<std::string> op_types =
      {
          "Cos",
          "Erf",
          "Floor",
          "Not",
          "Sin",
          "Sqrt",
      };

  op_registrations.builders.push_back(std::make_unique<UnaryOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
