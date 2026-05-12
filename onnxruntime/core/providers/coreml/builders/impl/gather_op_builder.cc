// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/shape_utils.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime::coreml {

class GatherOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;

  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
  bool SupportsMLProgram() const override { return true; }
};

namespace {
int64_t GetAxisAttribute(const Node& node) {
  NodeAttrHelper node_attr_helper{node};
  return node_attr_helper.Get("axis", int64_t{0});
}
}  // namespace

Status GatherOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                              const logging::Logger& /*logger*/) const {
  if (model_builder.CreateMLProgram()) {
    using CoreML::Specification::MILSpec::Operation;
    std::unique_ptr<Operation> op = model_builder.CreateOperation(node, "gather");

    const auto axis = GetAxisAttribute(node);
    // coreml docs claims validate_indices is optional but in practice it is required
    const auto validate_indices = false;
    AddOperationInput(*op, "x", node.InputDefs()[0]->Name());                                   // data
    AddOperationInput(*op, "indices", node.InputDefs()[1]->Name());                             // indices
    AddOperationInput(*op, "axis", model_builder.AddScalarConstant(op->type(), "axis", axis));  // axis attr
    AddOperationInput(*op, "validate_indices", model_builder.AddScalarConstant(op->type(), "validate_indices", validate_indices));
    AddOperationOutput(*op, *node.OutputDefs()[0]);  // output
    model_builder.AddOperation(std::move(op));
  } else {
    auto layer = model_builder.CreateNNLayer(node);
    layer->mutable_gather()->set_axis(GetAxisAttribute(node));
    *layer->mutable_input()->Add() = node.InputDefs()[0]->Name();    // data
    *layer->mutable_input()->Add() = node.InputDefs()[1]->Name();    // indices
    *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();  // output
    model_builder.AddLayer(std::move(layer));
  }
  return Status::OK();
}

bool GatherOpBuilder::HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                                             const logging::Logger& logger) const {
  int32_t input_type;
  if (!GetType(*node.InputDefs()[0], input_type, logger))
    return false;

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ||
       !input_params.create_mlprogram || input_params.coreml_version < 6) &&
      input_type != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    LOGS(logger, VERBOSE) << "[" << node.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

bool GatherOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                        const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> data_shape, indices_shape;
  if (!GetShape(*input_defs[0], data_shape, logger)) {
    LOGS(logger, VERBOSE) << "Failed to get 'data' shape";
    return false;
  }

  if (!GetShape(*input_defs[1], indices_shape, logger)) {
    LOGS(logger, VERBOSE) << "Failed to get 'indices' shape";
    return false;
  }

  // Scalar (rank-0) 'indices' input.
  //
  // MIL `gather` accepts rank-0 indices and produces the correct ONNX output
  // (the gathered axis is dropped). However, the CoreML EP reshapes any rank-0
  // *graph-boundary* tensor to {1} when it crosses into the CoreML subgraph
  // (see ModelBuilder::RegisterModelInputOutput) and a rank-0 input cannot be
  // represented as an MLMultiArray. So we only allow scalar indices when they
  // are a constant initializer: those flow through OnnxTensorToCoreMLTensor
  // with rank preserved and MIL gather can consume them directly.
  //
  // On the NeuralNetwork path scalar initializers are also reshaped to {1}
  // (ModelBuilder::RegisterInitializers, LoadConstantND requires rank >= 1),
  // so the gather output shape ends up wrong there. Keep rejecting that case.
  if (indices_shape.empty()) {
    if (!input_params.create_mlprogram) {
      LOGS(logger, VERBOSE) << "Gather does not support scalar 'indices' on the NeuralNetwork path";
      return false;
    }
    if (input_params.graph_viewer.GetConstantInitializer(input_defs[1]->Name()) == nullptr) {
      LOGS(logger, VERBOSE) << "Gather with scalar 'indices' is only supported when 'indices' is a constant initializer";
      return false;
    }
  }

  // ONNX Gather output rank = data_rank + indices_rank - 1.
  // For scalar indices (rank 0) this is data_rank - 1, which is what MIL also produces.
  if (data_shape.size() + indices_shape.size() - 1 > 5) {
    LOGS(logger, VERBOSE) << "Gather does not support output with rank greater than 5";
    return false;
  }

  return true;
}

void CreateGatherOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<GatherOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace onnxruntime::coreml
