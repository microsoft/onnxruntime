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

class TernaryOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;
};

// Add operator related.

Status TernaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                               const logging::Logger& /* logger */) const {
  const auto& op_type(node.OpType());
  ORT_RETURN_IF(node.InputDefs().size() < 3, "Operator requires at least three inputs");

  emscripten::val input0 = model_builder.GetOperand(node.InputDefs()[0]->Name());
  emscripten::val input1 = model_builder.GetOperand(node.InputDefs()[1]->Name());
  emscripten::val input2 = model_builder.GetOperand(node.InputDefs()[2]->Name());
  emscripten::val output = emscripten::val::object();
  if (op_type == "Where") {
    output = model_builder.GetBuilder().call<emscripten::val>("elementwiseIf", input0, input1, input2);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "TernaryOpBuilder::AddToModelBuilderImpl, unknown op: ", op_type);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

void CreateTernaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  static std::vector<std::string> op_types =
      {
          "Where",
      };

  op_registrations.builders.push_back(std::make_unique<TernaryOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
