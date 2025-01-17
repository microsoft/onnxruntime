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

class ClipOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

// Add operator related

void ClipOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
  if (inputs.size() > 1)
    model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());  // min

  if (inputs.size() > 2)
    model_builder.AddInitializerToSkip(inputs[2].node_arg.Name());  // max
}

Status ClipOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);

  if (Contains(model_builder.GetFusedActivations(), input)) {
    LOGS_DEFAULT(VERBOSE) << "Clip Node [" << node_unit.Name() << "] fused";
    model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type);
    return Status::OK();
  }

  float min, max;
  GetClipMinMax(model_builder.GetGraphViewer(), node_unit.GetNode(), min, max,
                logging::LoggingManager::DefaultLogger());

  int32_t op_code;
  if (min == 0.0f && max == 6.0f)
    op_code = ANEURALNETWORKS_RELU6;
  else if (min == -1.0f && max == 1.0f)
    op_code = ANEURALNETWORKS_RELU1;
  else
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ClipOpBuilder, unsupported input [", min, ", ", max, "].",
                           "We should not reach here, ClipOpBuilder::IsOpSupportedImpl should have caught this.");

  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

// Operator support related

bool ClipOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                      const OpSupportCheckParams& /* params */) const {
  float min, max;
  if (!GetClipMinMax(graph_viewer, node_unit.GetNode(), min, max, logging::LoggingManager::DefaultLogger()))
    return false;

  // We only supoort relu6 or relu1
  // TODO, support clip between 2 arbitrary numbers
  if ((min == 0.0f && max == 6.0f) || (min == -1.0f && max == 1.0f)) {
    return true;
  }

  LOGS_DEFAULT(VERBOSE) << "Clip only supports [min, max] = [0, 6] or [-1, 1], the input is ["
                        << min << ", " << max << "]";
  return false;
}

void CreateClipOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ClipOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace nnapi
}  // namespace onnxruntime
