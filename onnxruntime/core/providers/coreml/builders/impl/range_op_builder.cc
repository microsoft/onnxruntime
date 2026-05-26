// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

// ONNX Range(start, limit, delta) -> CoreML ML Program 'range_1d'. All three
// inputs are scalar tensors of the same dtype; produces a 1-D tensor of the
// generated range values.
class RangeOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

Status RangeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                             const logging::Logger& /*logger*/) const {
  using namespace CoreML::Specification::MILSpec;
  const auto& input_defs = node.InputDefs();

  // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.tensor_operation.range_1d
  // range_1d(end, start, step) -> tensor<[(end - start) / step], T>
  // ONNX Range is (start, limit, delta) -> remap accordingly.
  std::unique_ptr<Operation> op = model_builder.CreateOperation(node, "range_1d");
  AddOperationInput(*op, "end", input_defs[1]->Name());    // ONNX 'limit'
  AddOperationInput(*op, "start", input_defs[0]->Name());  // ONNX 'start'
  AddOperationInput(*op, "step", input_defs[2]->Name());   // ONNX 'delta'
  AddOperationOutput(*op, *node.OutputDefs()[0]);
  model_builder.AddOperation(std::move(op));
  return Status::OK();
}

bool RangeOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                       const logging::Logger& logger) const {
  if (!input_params.create_mlprogram) {
    LOGS(logger, VERBOSE) << node.OpType() << " is only supported for the ML Program format.";
    return false;
  }
  return true;
}

bool RangeOpBuilder::HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& /*input_params*/,
                                            const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  int32_t start_type = 0;
  if (!GetType(*input_defs[0], start_type, logger)) {
    return false;
  }
  // MIL range_1d supports int32, fp16, fp32.
  // ONNX uses int64 for integer ranges; treated as int32 inside the CoreML partition.
  if (start_type != ONNX_NAMESPACE::TensorProto_DataType_INT32 &&
      start_type != ONNX_NAMESPACE::TensorProto_DataType_INT64 &&
      start_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      start_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    LOGS(logger, VERBOSE) << node.OpType() << ": input dtype " << start_type << " not supported.";
    return false;
  }
  return true;
}

void CreateRangeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<RangeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
