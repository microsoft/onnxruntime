// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/onnx_pb.h>

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_factory.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_helpers.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_builder.h"

using namespace android::nn::wrapper;

namespace onnxruntime {
namespace nnapi {

using namespace op_builder_helpers;

class LRNOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void CreateLRNOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<LRNOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

Status LRNOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  NodeAttrHelper helper(node_unit);
  const auto android_feature_level = model_builder.GetNNAPIFeatureLevel();

  auto input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  auto use_nchw = model_builder.UseNCHW();
  ORT_RETURN_IF_ERROR(IsOpInRequiredLayout(use_nchw, node_unit));

  if (android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
    // on android api level 28, we need to transpose the nchw input to nhwc
    // it is very rare that users set nchw format when using nnapi. Therefore, instead of
    // adding the ability to support conversion we fail and stop.
    ORT_ENFORCE(!use_nchw, "NCHW format is not supported on android api level 28");
  }

  auto alpha = helper.Get("alpha", 0.0001f);
  const auto beta = helper.Get("beta", 0.75f);
  const auto bias = helper.Get("bias", 1.0f);
  const auto size = helper.Get("size", 1);

  const auto radius = (size - 1) / 2;
  alpha /= size;  // NNAPI's alpha is different than ONNX's alpha

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, radius);
  ADD_SCALAR_OPERAND(model_builder, input_indices, bias);
  ADD_SCALAR_OPERAND(model_builder, input_indices, alpha);
  ADD_SCALAR_OPERAND(model_builder, input_indices, beta);

  // specify axis is only available on api level >= 29
  if (android_feature_level > ANEURALNETWORKS_FEATURE_LEVEL_2) {
    // ONNX LRN is always performed on C dimension
    int32_t axis = use_nchw
                       ? 1   // nchw
                       : 3;  // nhwc
    ADD_SCALAR_OPERAND(model_builder, input_indices, axis);
  }

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

}  // namespace nnapi
}  // namespace onnxruntime
