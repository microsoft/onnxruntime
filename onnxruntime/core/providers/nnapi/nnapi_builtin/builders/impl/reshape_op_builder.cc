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

#include "base_op_builder.h"

using namespace android::nn::wrapper;

namespace onnxruntime {
namespace nnapi {

using namespace op_builder_helpers;

class ReshapeOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  // Reshape opset 4- uses attributes for new shape which we do not support for now
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 5; }
  bool HasSupportedInputOutputsImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                    const OpSupportCheckParams& params) const override;
  bool IsNodeUnitTypeSupported(const NodeUnit& /* node_unit */) const override { return true; }
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

// Add operator related

void ReshapeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (IsQuantizedOp(node_unit)) {
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Inputs()[0].quant_param);   // x_scale, x_zp
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
  }
  model_builder.AddInitializerToSkip(node_unit.Inputs()[1].node_arg.Name());
}

Status ReshapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& graph_viewer(model_builder.GetGraphViewer());
  auto input = node_unit.Inputs()[0].node_arg.Name();

  const auto& shape_tensor = *graph_viewer.GetConstantInitializer(node_unit.Inputs()[1].node_arg.Name());
  Initializer unpacked_tensor(shape_tensor);
  auto raw_shape = unpacked_tensor.DataAsSpan<int64_t>();
  const auto size = SafeInt<uint32_t>(shape_tensor.dims()[0]);

  const auto input_shape = shaper[input];
  std::vector<int32_t> shape(size);
  for (uint32_t i = 0; i < size; i++) {
    int32_t dim = SafeInt<int32_t>(raw_shape[i]);
    // NNAPI reshape does not support 0 as dimension
    shape[i] = dim == 0 ? input_shape[i] : dim;
  }

  // Check if the quantization scale and ZP are correct
  float x_scale = 0.0f;
  int32_t x_zero_point = 0;
  if (IsQuantizedOp(node_unit)) {
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        graph_viewer, node_unit.Inputs()[0], node_unit.ModelPath(), x_scale, x_zero_point));
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));
  }

  return AddReshapeOperator(model_builder, node_unit, input, shape);
}

// Operator support related

bool ReshapeOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQReshape;
}

bool ReshapeOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                         const OpSupportCheckParams& /* params */) const {
  const auto& inputs = node_unit.Inputs();
  const auto& perm_name = inputs[1].node_arg.Name();
  const auto* perm = graph_viewer.GetConstantInitializer(perm_name);
  if (!perm) {
    LOGS_DEFAULT(VERBOSE) << "New shape of reshape must be a constant initializer";
    return false;
  }

  Shape input_shape;
  if (!GetShape(inputs[0].node_arg, input_shape))
    return false;

  if (input_shape.size() > 4 || input_shape.empty()) {
    LOGS_DEFAULT(VERBOSE) << "Reshape only supports up to 1-4d shape, input is "
                          << input_shape.size() << "d shape";
    return false;
  }

  const auto& perm_tensor = *perm;
  Initializer unpacked_tensor(perm_tensor);
  auto raw_perm = unpacked_tensor.DataAsSpan<int64_t>();
  const auto perm_size = SafeInt<uint32_t>(perm_tensor.dims()[0]);

  NodeAttrHelper helper(node_unit);
  const bool allow_zero = helper.Get("allowzero", 0) == 1;
  for (uint32_t i = 0; i < perm_size; i++) {
    // NNAPI reshape does not support 0 as dimension
    if (raw_perm[i] == 0) {
      if (i < input_shape.size() && input_shape[i] == 0) {
        LOGS_DEFAULT(VERBOSE) << "Reshape doesn't support 0 reshape dimension on a dynamic dimension";
        return false;
      }

      if (allow_zero) {
        LOGS_DEFAULT(VERBOSE) << "Reshape doesn't support 0 reshape dimension when allowzero is enabled";
        return false;
      }
    }
  }

  return true;
}

bool ReshapeOpBuilder::HasSupportedInputOutputsImpl(
    const GraphViewer& graph_viewer, const NodeUnit& node_unit,
    const OpSupportCheckParams& params) const {
  if (!IsQuantizedOp(node_unit)) {
    return BaseOpBuilder::HasSupportedInputOutputsImpl(graph_viewer, node_unit, params);
  }

  if (!IsQuantizedIOSupported(graph_viewer, node_unit, {0}, params, ArgType::kInput)) {
    return false;
  }

  if (!IsQuantizedIOSupported(graph_viewer, node_unit, {0}, params, ArgType::kOutput)) {
    return false;
  }

  return true;
}

void CreateReshapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ReshapeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace nnapi
}  // namespace onnxruntime
