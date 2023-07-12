// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/onnx_pb.h>

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
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

using namespace op_builder_helpers;

class BatchNormalizationOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  // BatchNormalization opset 6- has unsupported attributes
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 7; }
};

// Add operator related

void BatchNormalizationOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  // skip everything except input0 for BatchNormalization
  model_builder.AddInitializerToSkip(node_unit.Inputs()[1].node_arg.Name());  // scale
  model_builder.AddInitializerToSkip(node_unit.Inputs()[2].node_arg.Name());  // B
  model_builder.AddInitializerToSkip(node_unit.Inputs()[3].node_arg.Name());  // mean
  model_builder.AddInitializerToSkip(node_unit.Inputs()[4].node_arg.Name());  // var
}

Status BatchNormalizationOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                          const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());
  NodeAttrHelper helper(node_unit);
  const auto& inputs = node_unit.Inputs();

  // For reshape we are not really doing anything but
  // register a new operand with new shape
  const auto& input = inputs[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  const auto input_shape = shaper[input];

  const auto& scale_tensor = *initializers.at(inputs[1].node_arg.Name());
  const auto& bias_tensor = *initializers.at(inputs[2].node_arg.Name());
  const auto& mean_tensor = *initializers.at(inputs[3].node_arg.Name());
  const auto& var_tensor = *initializers.at(inputs[4].node_arg.Name());
  const auto eps = helper.Get("epsilon", 1e-5f);

  const auto size = SafeInt<uint32_t>(scale_tensor.dims()[0]);
  std::vector<float> a, b;
  a.reserve(size);
  b.reserve(size);

  Initializer unpacked_scale_tensor(scale_tensor);
  Initializer unpacked_bias_tensor(bias_tensor);
  Initializer unpacked_mean_tensor(mean_tensor);
  Initializer unpacked_var_tensor(var_tensor);

  auto scale_data = unpacked_scale_tensor.DataAsSpan<float>();
  auto bias_data = unpacked_bias_tensor.DataAsSpan<float>();
  auto mean_data = unpacked_mean_tensor.DataAsSpan<float>();
  auto var_data = unpacked_var_tensor.DataAsSpan<float>();

  for (uint32_t i = 0; i < size; i++) {
    a.push_back(scale_data[i] / sqrt(var_data[i] + eps));
    b.push_back((scale_data[i] * -mean_data[i]) / sqrt(var_data[i] + eps) +
                bias_data[i]);
  }

  const auto tensor_a_name = model_builder.GetUniqueName(node_unit.Name() + input + "_imm_a");
  const auto tensor_b_name = model_builder.GetUniqueName(node_unit.Name() + input + "_imm_b");
  const auto tensor_imm_product_name = model_builder.GetUniqueName(node_unit.Name() + input + "_imm_mul");
  Shape tensor_a_dimen = {size};

  bool use_nchw = model_builder.UseNCHW();
  ORT_RETURN_IF_ERROR(IsOpInRequiredLayout(use_nchw, node_unit));

  if (use_nchw) {
    // the batch normalization is applied on C channel,
    // if the input is NC[HW], will need correct shape for tensor_a/b
    // to make sure we are broadcasting on the correct channel,
    // input shape {N, C}       ==> tensor_a/b's shape {size}
    // input shape {N, C, H}    ==> tensor_a/b's shape {size, 1}
    // input shape {N, C, H, W} ==> tensor_a/b's shape {size, 1, 1}
    const auto input_rank = input_shape.size();
    for (size_t i = 2; i < input_rank; i++)
      tensor_a_dimen.push_back(1);
  }

  shaper.AddShape(input, input_shape);
  shaper.AddShape(tensor_a_name, tensor_a_dimen);
  shaper.AddShape(tensor_b_name, tensor_a_dimen);
  const OperandType a_operand_type(operand_types.at(input).type, tensor_a_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(tensor_a_name, a.data(), a_operand_type));
  const OperandType b_operand_type(operand_types.at(input).type, tensor_a_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(tensor_b_name, b.data(), b_operand_type));

  int32_t fuse_code = model_builder.FindActivation(node_unit);
  ORT_RETURN_IF_ERROR(AddNnapiBatchNormalization(model_builder, input, tensor_a_name, tensor_b_name,
                                                 tensor_imm_product_name, output, fuse_code));
  return Status::OK();
}

// Operator support related

bool BatchNormalizationOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                                    const OpSupportCheckParams& /* params */) const {
  if (node_unit.Outputs().size() != 1) {
    LOGS_DEFAULT(VERBOSE) << "Your onnx model may be in training mode, please export "
                             "it in test mode.";
    return false;
  }

  const auto& inputs = node_unit.Inputs();
  Shape input_shape;
  if (!GetShape(inputs[0].node_arg, input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size > 4) {
    LOGS_DEFAULT(VERBOSE) << "BN only support up to 4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  NodeAttrHelper helper(node_unit);
  const auto spatial = helper.Get("spatial", 1);
  if (spatial != 1) {
    LOGS_DEFAULT(VERBOSE) << "Non-spatial BN is not supported";
    return false;
  }

  const auto& scale_name = inputs[1].node_arg.Name();
  const auto& b_name = inputs[2].node_arg.Name();
  const auto& mean_name = inputs[3].node_arg.Name();
  const auto& var_name = inputs[4].node_arg.Name();
  if (!Contains(initializers, scale_name)) {
    LOGS_DEFAULT(VERBOSE) << "Scale of BN must be known";
    return false;
  }
  if (!Contains(initializers, b_name)) {
    LOGS_DEFAULT(VERBOSE) << "B of BN must be known";
    return false;
  }
  if (!Contains(initializers, mean_name)) {
    LOGS_DEFAULT(VERBOSE) << "Mean of BN must be known";
    return false;
  }
  if (!Contains(initializers, var_name)) {
    LOGS_DEFAULT(VERBOSE) << "Var of BN must be known";
    return false;
  }

  return true;
}

void CreateBatchNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<BatchNormalizationOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace nnapi
}  // namespace onnxruntime
