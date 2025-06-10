// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"

namespace onnxruntime {
namespace coreml {

class WhereOpBuilder : public BaseOpBuilder {
 public:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

Status WhereOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                             const logging::Logger& /* logger */) const {
  const auto& input_defs = node.InputDefs();

  using namespace CoreML::Specification::MILSpec;

  std::unique_ptr<Operation> op = model_builder.CreateOperation(node, "select");
  AddOperationInput(*op, "cond", input_defs[0]->Name());
  AddOperationInput(*op, "a", input_defs[1]->Name());
  AddOperationInput(*op, "b", input_defs[2]->Name());
  AddOperationOutput(*op, *node.OutputDefs()[0]);

  model_builder.AddOperation(std::move(op));

  return Status::OK();
}

bool WhereOpBuilder::HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                                            const logging::Logger& logger) const {
  // ONNX spec specifies that cond input must be a boolean tensor
  if (!IsInputDtypeSupport(node, 1, input_params, logger)) {
    return false;
  }

  if (!IsInputDtypeSupport(node, 2, input_params, logger)) {
    return false;
  }

  return true;
}

bool WhereOpBuilder::IsOpSupportedImpl(const Node& /*node*/, const OpBuilderInputParams& input_params,
                                       const logging::Logger& logger) const {
  if (!input_params.create_mlprogram) {
    LOGS(logger, VERBOSE) << "WhereOpBuilder: ML Program is required for 'Where' operator.";
    return false;
  }
  return true;
}

void CreateWhereOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<WhereOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
