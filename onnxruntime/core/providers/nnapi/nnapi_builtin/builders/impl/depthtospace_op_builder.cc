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

class DepthToSpaceOpBuilder : public BaseOpBuilder {
  // Add operator related
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

// Add operator related

Status DepthToSpaceOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto android_feature_level = model_builder.GetEffectiveFeatureLevel();
  NodeAttrHelper helper(node_unit);

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  bool use_nchw = model_builder.UseNCHW();
  ORT_RETURN_IF_ERROR(IsOpInRequiredLayout(use_nchw, node_unit));

  int32_t blocksize = SafeInt<int32_t>(node_unit.GetNode().GetAttributes().at("blocksize").i());

  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, blocksize);

  if (use_nchw && android_feature_level > ANEURALNETWORKS_FEATURE_LEVEL_2) {
    // optional input to use nchw is available starting NNAPI feature level 3
    ADD_SCALAR_OPERAND(model_builder, input_indices, use_nchw);
  }
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_DEPTH_TO_SPACE, input_indices, {output},
                                                 {output_operand_type}));
  return Status::OK();
}

// Operator support related

bool DepthToSpaceOpBuilder::IsOpSupportedImpl(const GraphViewer& /* graph_viewer */, const NodeUnit& node_unit,
                                              const OpSupportCheckParams& params) const {
  NodeAttrHelper helper(node_unit);

  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS_DEFAULT(VERBOSE) << "DepthToSpace only supports 4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  if (params.use_nchw && params.android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
    LOGS_DEFAULT(VERBOSE) << "NCHW layout is not supported for android feature level: " << params.android_feature_level;
    return false;
  }

  auto since_version = node_unit.SinceVersion();
  if (since_version >= 11) {
    // For now, only DCR mode is accepted as NNAPI only supports DCR format data rearrangement
    const auto mode = helper.Get("mode", "DCR");
    if (mode != "DCR") {
      LOGS_DEFAULT(VERBOSE) << "ANEURALNETWORKS_DEPTH_TO_SPACE only supports DCR rearrangement mode. Current mode:"
                            << mode;
      return false;
    }
  }

  return true;
}

void CreateDepthToSpaceOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<DepthToSpaceOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace nnapi
}  // namespace onnxruntime
