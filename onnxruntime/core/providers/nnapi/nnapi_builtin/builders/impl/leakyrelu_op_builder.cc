// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/onnx_pb.h>

#include "core/common/logging/logging.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_factory.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_helpers.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_builder.h"

using namespace android::nn::wrapper;

namespace onnxruntime {
namespace nnapi {

class LeakyReluOpBuilder : public BaseOpBuilder {
  // Add operator related
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  // LeakyRelu opset 6- has unsupported attributes
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 6; }
};

// Add operator related

Status LeakyReluOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                 const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& operand_indices(model_builder.GetOperandIndices());

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  auto input_shape = shaper[input];

  NodeAttrHelper helper(node_unit);
  const auto alpha = helper.Get("alpha", 0.01f);

  // We will use multiple operations to simulate LeakyRelu here, including NNAPI ANEURALNETWORKS_SELECT/
  // ANEURALNETWORKS_LESS/ANEURALNETWORKS_MUL.
  // input X = [-1, 0, 1]
  // multiply by the alpha value: Z = alpha * X = [-0.1, 0, 0.1]
  // then construct the boolean input X less than zero: C = [false, true, true], true means selecting element from the first input,
  // and false vice versa.
  // output Y = [-0.1, 0, 1]
  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));

  // Add Less operation - Less(X, Zero)
  float zero_value = 0.0f;
  InlinedVector<uint32_t> value_shape{1};

  std::string zero_vec_name = model_builder.GetUniqueName(node_unit.Name() + input + "_zero");
  std::string less_output_name = model_builder.GetUniqueName(node_unit.Name() + input + "_less_than_zero");

  const OperandType zero_operand_type(Type::TENSOR_FLOAT32, value_shape);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(zero_vec_name, &zero_value,
                                                                      zero_operand_type));
  input_indices.push_back(operand_indices.at(zero_vec_name));

  OperandType less_output_operand_type(Type::TENSOR_BOOL8, input_shape);
  shaper.AddShape(less_output_name, input_shape);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_LESS,
                                                 input_indices, {less_output_name}, {less_output_operand_type}));

  // Add Mul operation - Mul(Alpha, X)
  input_indices.clear();
  input_indices.push_back(operand_indices.at(input));

  std::string alpha_vec_name = model_builder.GetUniqueName(node_unit.Name() + input + "_alpha");
  const OperandType alpha_operand_type(Type::TENSOR_FLOAT32, value_shape);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(alpha_vec_name, &alpha,
                                                                      alpha_operand_type));
  input_indices.push_back(operand_indices.at(alpha_vec_name));

  ADD_SCALAR_OPERAND(model_builder, input_indices, ANEURALNETWORKS_FUSED_NONE);

  std::string mul_output_name = model_builder.GetUniqueName(node_unit.Name() + input + "_multiply_alpha");
  const OperandType mul_output_operand_type(operand_types.at(input).type, input_shape);
  shaper.AddShape(mul_output_name, input_shape);

  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_MUL, input_indices,
                                                 {mul_output_name}, {mul_output_operand_type}));

  // Add Select Operation - Select(XLessThanZero, AlphaMulX, X)
  input_indices.clear();
  input_indices.push_back(operand_indices.at(less_output_name));
  input_indices.push_back(operand_indices.at(mul_output_name));
  input_indices.push_back(operand_indices.at(input));

  const OperandType select_output_operand_type(operand_types.at(input).type, input_shape);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_SELECT, input_indices,
                                                 {output}, {select_output_operand_type}));

  return Status::OK();
}

// Operator support related

bool LeakyReluOpBuilder::IsOpSupportedImpl(const GraphViewer& /*graph_viewer*/, const NodeUnit& node_unit,
                                           const OpSupportCheckParams& /* params */) const {
  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  // Note: We will use ANEURALNETWORKS_SELECT/LESS to simulate LeakyRelu op, ANEURALNETWORKS_SELECT/LESS has supported
  // tensor rank from 1:
  // https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a49b2dc37ea9219789a6d82f281499dbb
  // And ANEURALNETWORKS_MUL has supported tensor rank up to 4
  if (input_shape.empty() || input_shape.size() > 4) {
    LOGS_DEFAULT(VERBOSE) << "NNAPI LeakyRelu supports tensor rank from 1-4d. Empty shape is not supported. Input is "
                          << input_shape.size() << "d shape";
    return false;
  }

  return true;
}

void CreateLeakyReluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<LeakyReluOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace nnapi
}  // namespace onnxruntime
