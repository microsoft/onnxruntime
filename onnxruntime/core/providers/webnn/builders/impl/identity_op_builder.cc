// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class IdentityOpBuilder : public BaseOpBuilder {
  // Add operator related.
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;
};

// Add operator related.
Status IdentityOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                const Node& node,
                                                const logging::Logger& logger) const {
  const auto& input_name = node.InputDefs()[0]->Name();
  const auto& output_name = node.OutputDefs()[0]->Name();
  emscripten::val input = model_builder.GetOperand(input_name);
  emscripten::val output = emscripten::val::object();

  output = model_builder.GetBuilder().call<emscripten::val>("identity", input);

  model_builder.AddOperand(output_name, std::move(output));
  return Status::OK();
}

void CreateIdentityOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<IdentityOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
