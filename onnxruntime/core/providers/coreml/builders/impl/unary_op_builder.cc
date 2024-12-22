// Copyright (c) Shukant Pal.
// Licensed under the MIT License.

#include "core/providers/common.h"

#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"

namespace onnxruntime {
namespace coreml {

class UnaryOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;
  bool SupportsMLProgram() const override { return true; }
  bool IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                         const logging::Logger& logger) const override;
};

Status UnaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                             const logging::Logger& /* logger */) const {
  const auto& op_type(node.OpType());
  const auto& input_defs(node.InputDefs());

#if defined(COREML_ENABLE_MLPROGRAM)
  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;

    std::string_view coreml_op_type;
    if (op_type == "Sqrt") {
      coreml_op_type = "sqrt";
    } else if (op_type == "Reciprocal") {
      coreml_op_type = "inverse";
    } else if (op_type == "Erf") {
      coreml_op_type = "erf";
    } else if (op_type == "Round") {
      coreml_op_type = "round";
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "UnaryOpBuilder::AddToModelBuilderImpl, unexpected op: ", op_type);
    }

    std::unique_ptr<Operation> op = model_builder.CreateOperation(node, coreml_op_type);
    AddOperationInput(*op, "x", input_defs[0]->Name());
    if (op_type == "Reciprocal") {
      float epsilon = 1e-4f;  // epsilon: const T (Optional, default=1e-4)
      auto dtype = node.InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
      if (dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
        AddOperationInput(*op, "epsilon", model_builder.AddScalarConstant(op->type(), "epsilon", epsilon));
      } else if (dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
        AddOperationInput(*op, "epsilon", model_builder.AddScalarConstant(op->type(), "epsilon", MLFloat16(epsilon)));
      }
    }

    AddOperationOutput(*op, *node.OutputDefs()[0]);

    model_builder.AddOperation(std::move(op));
  } else  // NOLINT
#endif    // defined (COREML_ENABLE_MLPROGRAM)
  {
    std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = model_builder.CreateNNLayer(node);

    if (op_type == "Sqrt") {
      layer->mutable_unary()->set_type(COREML_SPEC::UnaryFunctionLayerParams::SQRT);
    } else if (op_type == "Reciprocal") {
      layer->mutable_unary()->set_type(COREML_SPEC::UnaryFunctionLayerParams::INVERSE);
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "UnaryOpBuilder::AddToModelBuilderImpl, unknown op: ", op_type);
    }

    *layer->mutable_input()->Add() = input_defs[0]->Name();
    *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

    model_builder.AddLayer(std::move(layer));
  }
  return Status::OK();
}

bool UnaryOpBuilder::IsOpSupportedImpl(const Node& node, const OpBuilderInputParams& input_params,
                                       const logging::Logger& /*logger*/) const {
  if (!input_params.create_mlprogram && (node.OpType() == "Erf" || node.OpType() == "Round")) {
    return false;
  }
  return true;
}

void CreateUnaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<UnaryOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
