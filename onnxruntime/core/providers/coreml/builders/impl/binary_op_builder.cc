// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/coreml/builders/helper.h"
#include "core/providers/coreml/builders/op_builder_factory.h"
#include "core/providers/shared/utils/utils.h"
#ifdef __APPLE__
#include "core/framework/tensorprotoutils.h"
#include "core/providers/coreml/builders/model_builder.h"
#endif

#include "base_op_builder.h"

namespace onnxruntime {
namespace coreml {

class BinaryOpBuilder : public BaseOpBuilder {
  // Add operator related
 private:
#ifdef __APPLE__
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override;
#endif
  // Operator support related
  int GetMinSupportedOpSet(const Node& node) const override;

  bool HasSupportedInputsImpl(const Node& node, const logging::Logger& logger) const override;
};

#ifdef __APPLE__
static bool CheckIfBothInputShapesMatch(const Node& node, const logging::Logger& logger) {
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

// Add operator related

Status BinaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                              const logging::Logger& logger) const {
  const auto& op_type(node.OpType());
  const auto& input_defs(node.InputDefs());

  std::unique_ptr<COREML_SPEC::NeuralNetworkLayer> layer = CreateNNLayer(model_builder, node);

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
                           "BinaryOpBuilder::AddToModelBuilderImpl, unknown op: ", op_type);
  }

  *layer->mutable_input()->Add() = input_defs[0]->Name();
  *layer->mutable_input()->Add() = input_defs[1]->Name();
  *layer->mutable_output()->Add() = node.OutputDefs()[0]->Name();

  model_builder.AddLayer(std::move(layer));
  return Status::OK();
}
#endif

// Operator support related

int BinaryOpBuilder::GetMinSupportedOpSet(const Node& /* node */) const {
  // Add/Sub/Mul/Div opset 6- has broadcast attributes we do not support now
  return 7;
}

bool BinaryOpBuilder::HasSupportedInputsImpl(const Node& node, const logging::Logger& logger) const {
  bool is_pow = node.OpType() == "Pow";
  if (!is_pow) {
    return BaseOpBuilder::HasSupportedInputsImpl(node, logger);
  }

  const auto& input_1 = *node.InputDefs()[0];
  const auto& input_2 = *node.InputDefs()[1];
  // Pow we only support both inputs as fp32 for now
  int32_t input_type_1;
  if (!GetType(input_1, input_type_1, logger))
    return false;

  int32_t input_type_2;
  if (!GetType(input_2, input_type_2, logger))
    return false;

  if (input_type_1 != ONNX_NAMESPACE::TensorProto_DataType_FLOAT || input_type_1 != input_type_2) {
    LOGS(logger, VERBOSE) << "Pow only supports fp32 inputs, actual input type"
                          << ", Input type 1: " << input_type_1
                          << ", Input type 2: " << input_type_2;
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
