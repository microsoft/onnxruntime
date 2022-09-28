// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/onnx_pb.h>

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
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

class EluOpBuilder : public BaseOpBuilder {
  // Add operator related
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_4;
  }

  // Elu opset 5- uses consumed_inputs attribute which is not supported for now
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 6; }
};

Status EluOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  NodeAttrHelper helper(node_unit);
  const auto alpha = helper.Get("alpha", 1.0f);
  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, alpha);
  return model_builder.AddOperation(ANEURALNETWORKS_ELU, input_indices,
                                    {output}, {output_operand_type});
}

void CreateEluOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<EluOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace nnapi
}  // namespace onnxruntime
