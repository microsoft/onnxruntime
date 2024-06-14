// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/impl/base_op_builder.h"
#include "core/providers/coreml/builders/impl/builder_utils.h"
#include "core/providers/coreml/builders/model_builder.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime {
namespace coreml {
class BinaryOpBuilder : public BaseOpBuilder {
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;

  int GetMinSupportedOpSet(const Node& node) const override;

  bool HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                              const logging::Logger& logger) const override;

  bool SupportsMLProgram() const override { return true; }
};

namespace {
bool CheckIfBothInputShapesMatch(const Node& node, const logging::Logger& logger) {
  const auto& input_defs = node.InputDefs();

  const auto* x_shape_proto = input_defs[0]->Shape();
  const auto* y_shape_proto = input_defs[1]->Shape();

  if (!x_shape_proto || !y_shape_proto) {
    LOGS(logger, WARNING) << "[" << node.Name() << "] Input shape is missing";
    return false;
  }

  using Dimension = ONNX_NAMESPACE::TensorShapeProto::Dimension;
  auto dim_eq =
      [](const Dimension& x_dim, const Dimension& y_dim) {
        const bool x_has_dim_value = utils::HasDimValue(x_dim);
        if (x_has_dim_value != utils::HasDimValue(y_dim)) {
          return false;
        }
        if (x_has_dim_value) {
          return x_dim.dim_value() == y_dim.dim_value();
        }
        return x_dim.dim_param() == y_dim.dim_param();
      };

  return std::equal(x_shape_proto->dim().begin(), x_shape_proto->dim().end(),
                    y_shape_proto->dim().begin(), y_shape_proto->dim().end(),
                    dim_eq);
}
}  // namespace

Status BinaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                              const logging::Logger& logger) const {
  const auto& op_type(node.OpType());
  const auto& input_defs(node.InputDefs());

#if defined(COREML_ENABLE_MLPROGRAM)
  if (model_builder.CreateMLProgram()) {
    using namespace CoreML::Specification::MILSpec;

    // https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#module-coremltools.converters.mil.mil.ops.defs.iOS15.elementwise_binary
    std::string_view coreml_op_type;
    if (op_type == "Add") {
      coreml_op_type = "add";
    } else if (op_type == "Mul") {
      coreml_op_type = "mul";
    } else if (op_type == "Sub") {
      coreml_op_type = "sub";
    } else if (op_type == "Div") {
      // we only support fp32 currently. when we add support for integers we need to check the type and use
      // "floor_div" or "real_div" accordingly
      coreml_op_type = "real_div";
    } else if (op_type == "Pow") {
      coreml_op_type = "pow";
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "BinaryOpBuilder::AddToModelBuilderImpl, unexpected op: ", op_type);
    }

    std::unique_ptr<Operation> op = model_builder.CreateOperation(node, coreml_op_type);
    AddOperationInput(*op, "x", input_defs[0]->Name());
    AddOperationInput(*op, "y", input_defs[1]->Name());
    AddOperationOutput(*op, *node.OutputDefs()[0]);

    model_builder.AddOperation(std::move(op));
  } else
#endif  // defined (COREML_ENABLE_MLPROGRAM)
  {
    std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = model_builder.CreateNNLayer(node);

    if (op_type == "Add") {
      // original mutable_add() has limited broadcasting support
      // updated to use CoreML::AddBroadcastableLayerParams which has more general broadcasting support
      if (CheckIfBothInputShapesMatch(node, logger)) {
        layer->mutable_add();
      } else {
        layer->mutable_addbroadcastable();
      }
    } else if (op_type == "Mul") {
      if (CheckIfBothInputShapesMatch(node, logger)) {
        layer->mutable_multiply();
      } else {
        layer->mutable_multiplybroadcastable();
      }
    } else if (op_type == "Sub") {
      layer->mutable_subtractbroadcastable();
    } else if (op_type == "Div") {
      layer->mutable_dividebroadcastable();
    } else if (op_type == "Pow") {
      layer->mutable_powbroadcastable();
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "BinaryOpBuilder::AddToModelBuilderImpl, unexpected op: ", op_type);
    }

    *layer->mutable_input()->Add() = input_defs[0]->Name();
    *layer->mutable_input()->Add() = input_defs[1]->Name();
    *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

    model_builder.AddLayer(std::move(layer));
  }

  return Status::OK();
}

int BinaryOpBuilder::GetMinSupportedOpSet(const Node& /* node */) const {
  // Add/Sub/Mul/Div opset 6- has broadcast attributes we do not support now
  return 7;
}

bool BinaryOpBuilder::HasSupportedInputsImpl(const Node& node, const OpBuilderInputParams& input_params,
                                             const logging::Logger& logger) const {
  // Add/Sub/Mul/Div spec says inputs must be of the same type.
  // Pow spec says inputs can be different types.
  // We only support float for all of these inputs.
  if (!IsInputFloat(node, 0, input_params, logger) ||
      ((node.OpType() == "Pow") && !IsInputFloat(node, 1, input_params, logger))) {
    return false;
  }

  return true;
}

void CreateBinaryOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<BinaryOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace coreml
}  // namespace onnxruntime
