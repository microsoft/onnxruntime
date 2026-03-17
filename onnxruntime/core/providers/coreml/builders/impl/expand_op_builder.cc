// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {

class ExpandOpBuilder : public BaseOpBuilder {
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;

  // Expand opset 8 is the first version with the shape input
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 8; }

  bool SupportsMLProgram() const override { return true; }
};

void ExpandOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  const auto& input_defs = node.InputDefs();
  // Skip the shape input if it is a constant initializer, as we will consume it directly
  if (input_defs.size() > 1) {
    const auto* shape_initializer = model_builder.GetConstantInitializer(input_defs[1]->Name());
    if (shape_initializer) {
      model_builder.AddInitializerToSkip(input_defs[1]->Name());
    }
  }
}

Status ExpandOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                              const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;

    // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.tensor_transformation.broadcast_to
    //
    // CoreML broadcast_to: broadcasts the input tensor to the given shape.
    // Inputs: x (tensor), shape (1D int32 tensor of target shape)
    std::unique_ptr<Operation> op = model_builder.CreateOperation(node, "broadcast_to");
    AddOperationInput(*op, "x", input_defs[0]->Name());

    const auto* shape_initializer = model_builder.GetConstantInitializer(input_defs[1]->Name());
    if (shape_initializer) {
      // Shape is a constant initializer — read it and add as a const operation
      Initializer unpacked_tensor(model_builder.GetGraphViewer().GetGraph(), *shape_initializer);
      auto shape_data = unpacked_tensor.DataAsSpan<int64_t>();
      AddOperationInput(*op, "shape",
                        model_builder.AddConstant(op->type(), "shape", shape_data));
    } else {
      // Shape is a dynamic runtime value. CoreML broadcast_to requires shape as int32,
      // but ONNX Expand provides shape as int64. Insert a cast op to convert.
      std::unique_ptr<Operation> cast_op = model_builder.CreateOperation(node, "cast", "shape_cast");
      AddOperationInput(*cast_op, "x", input_defs[1]->Name());
      AddOperationInput(*cast_op, "dtype",
                        model_builder.AddScalarConstant(cast_op->type(), "dtype", std::string("int32")));
      const auto& cast_output = model_builder.GetUniqueName(node, "shape_int32");
      AddIntermediateOperationOutput(*cast_op, cast_output,
                                     ONNX_NAMESPACE::TensorProto_DataType_INT32, std::nullopt);
      model_builder.AddOperation(std::move(cast_op));

      AddOperationInput(*op, "shape", cast_output);
    }

    AddOperationOutput(*op, *node.OutputDefs()[0]);
    model_builder.AddOperation(std::move(op));
  } else {
    // NeuralNetwork path is not supported for Expand
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "ExpandOpBuilder: Expand is only supported in ML Program mode");
  }

  ORT_UNUSED_PARAMETER(logger);
  return Status::OK();
}

bool ExpandOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                        const logging::Logger& logger) const {
  if (!input_params.create_mlprogram) {
    LOGS(logger, VERBOSE) << "Expand is only supported in ML Program mode";
    return false;
  }

  const auto& input_defs = node.InputDefs();
  if (input_defs.size() < 2) {
    LOGS(logger, VERBOSE) << "Expand requires 2 inputs (data and shape)";
    return false;
  }

  // Validate output rank does not exceed 5 (CoreML limitation)
  const auto* output_shape = node.OutputDefs()[0]->Shape();
  if (output_shape && output_shape->dim_size() > 5) {
    LOGS(logger, VERBOSE) << "Expand output rank " << output_shape->dim_size()
                          << " exceeds CoreML limit of 5";
    return false;
  }

  return true;
}

void CreateExpandOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ExpandOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
