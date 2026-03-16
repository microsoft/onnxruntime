// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

class WhereOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;

  // Where opset 9 is the first version with broadcasting support
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 9; }

  bool SupportsMLProgram() const override { return true; }
};

Status WhereOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                             const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;

    // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_binary.select
    //
    // CoreML select: return cond ? a : b (element-wise with broadcasting)
    // Inputs: cond (bool tensor), a (true branch), b (false branch)
    std::unique_ptr<Operation> op = model_builder.CreateOperation(node, "select");
    AddOperationInput(*op, "cond", input_defs[0]->Name());
    AddOperationInput(*op, "a", input_defs[1]->Name());
    AddOperationInput(*op, "b", input_defs[2]->Name());

    AddOperationOutput(*op, *node.OutputDefs()[0]);
    model_builder.AddOperation(std::move(op));
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "WhereOpBuilder: Where is only supported in ML Program mode");
  }

  ORT_UNUSED_PARAMETER(logger);
  return Status::OK();
}

bool WhereOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                       const logging::Logger& logger) const {
  if (!input_params.create_mlprogram) {
    LOGS(logger, VERBOSE) << "Where is only supported in ML Program mode";
    return false;
  }

  const auto& input_defs = node.InputDefs();
  if (input_defs.size() < 3) {
    LOGS(logger, VERBOSE) << "Where requires 3 inputs (condition, X, Y)";
    return false;
  }

  // Validate output rank does not exceed 5 (CoreML limitation)
  const auto* output_shape = node.OutputDefs()[0]->Shape();
  if (output_shape && output_shape->dim_size() > 5) {
    LOGS(logger, VERBOSE) << "Where output rank " << output_shape->dim_size()
                          << " exceeds CoreML limit of 5";
    return false;
  }

  return true;
}

bool WhereOpBuilder::HasSupportedInputsImpl(const Node& node,
                                            [[maybe_unused]] const OpBuilderInputParams& input_params,
                                            const logging::Logger& logger) const {
  // Where has 3 inputs:
  //   input 0 (condition): must be BOOL
  //   input 1 (X / true branch): FLOAT, FLOAT16, or INT64 (in ML Program mode)
  //   input 2 (Y / false branch): same type as X
  const auto& input_defs = node.InputDefs();

  // Validate condition input is BOOL
  int32_t cond_type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  if (!GetType(*input_defs[0], cond_type, logger)) {
    return false;
  }
  if (cond_type != ONNX_NAMESPACE::TensorProto_DataType_BOOL) {
    LOGS(logger, VERBOSE) << "[Where] condition input must be BOOL, got type: " << cond_type;
    return false;
  }

  // Validate X and Y inputs have supported types
  if (!IsInputDtypeSupport(node, 1, input_params, logger)) {
    return false;
  }
  if (!IsInputDtypeSupport(node, 2, input_params, logger)) {
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
