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

class TransposeOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_2;
  }

  bool HasSupportedInputOutputsImpl(
      const GraphViewer& graph_viewer, const NodeUnit& node_unit,
      const OpSupportCheckParams& params) const override;
  bool IsNodeUnitTypeSupported(const NodeUnit& /* node_unit */) const override { return true; }
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

// Add operator related

void TransposeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (!IsQuantizedOp(node_unit))
    return;

  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Inputs()[0].quant_param);   // x_scale, x_zp
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
}

Status TransposeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  NodeAttrHelper helper(node_unit);
  std::vector<int32_t> perm = helper.Get("perm", std::vector<int32_t>{});
  auto input_dims = static_cast<int32_t>(shaper[input].size());
  if (perm.empty()) {
    for (int32_t i = input_dims - 1; i >= 0; i--)
      perm.push_back(i);
  } else {
    ORT_RETURN_IF_NOT(static_cast<int32_t>(perm.size()) == input_dims, "Perm and input should have same dimension");
  }

  // Check if the quantization scale and ZP are correct
  if (IsQuantizedOp(node_unit)) {
    float x_scale = 0.0f;
    int32_t x_zero_point = 0;
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        model_builder.GetGraphViewer(), node_unit.Inputs()[0], node_unit.ModelPath(), x_scale, x_zero_point));
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));
  }

  std::string perm_name = model_builder.GetUniqueName(node_unit.Name() + input + "perm");

  ORT_RETURN_IF_ERROR(op_builder_helpers::AddNnapiTranspose(model_builder, input, perm_name, perm, output));

  return Status::OK();
}

// Operator support related

bool TransposeOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQTranspose;
}

bool TransposeOpBuilder::IsOpSupportedImpl(const GraphViewer& /* graph_viewer */, const NodeUnit& node_unit,
                                           const OpSupportCheckParams& /* params */) const {
  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size > 4 || input_size == 0) {
    LOGS_DEFAULT(VERBOSE) << "Transpose only supports 1-4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  return true;
}

bool TransposeOpBuilder::HasSupportedInputOutputsImpl(
    const GraphViewer& graph_viewer, const NodeUnit& node_unit,
    const OpSupportCheckParams& params) const {
  int32_t input_type;
  if (!GetType(node_unit.Inputs()[0].node_arg, input_type))
    return false;

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      input_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    LOGS_DEFAULT(VERBOSE) << "[" << node_unit.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
    return false;
  }

  if (IsQuantizedOp(node_unit)) {
    if (!IsQuantizedIOSupported(graph_viewer, node_unit, {0}, params, ArgType::kInput))
      return false;

    if (!IsQuantizedIOSupported(graph_viewer, node_unit, {0}, params, ArgType::kOutput))
      return false;
  }

  return true;
}

void CreateTransposeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<TransposeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace nnapi
}  // namespace onnxruntime
