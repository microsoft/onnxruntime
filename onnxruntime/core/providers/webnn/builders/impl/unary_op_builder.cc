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
  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());

  const std::string_view webnn_op_type = GetWebNNOpType(op_type);
  ORT_RETURN_IF(webnn_op_type.empty(), "Cannot get WebNN op type");

  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>(
      std::string(webnn_op_type).c_str(), input, options);

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

void CreateUnaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.count(op_type) > 0)
    return;

  static std::vector<std::string> op_types =
      {
          "Abs",
          "Ceil",
          "Cos",
          "Erf",
          "Exp",
          "Floor",
          "Identity",
          "Log",
          "Neg",
          "Reciprocal",
          "Round",
          "Sign",
          "Sin",
          "Sqrt",
          "Tan",
      };

  op_registrations.builders.push_back(std::make_unique<UnaryOpBuilder>());
  for (const auto& type : op_types) {
    op_registrations.op_builder_map.emplace(type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
