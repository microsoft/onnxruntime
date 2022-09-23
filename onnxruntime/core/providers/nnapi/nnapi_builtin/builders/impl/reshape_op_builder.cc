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

#include "base_op_builder.h"

using namespace android::nn::wrapper;

namespace onnxruntime {
namespace nnapi {

using namespace op_builder_helpers;

class ReshapeOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

void CreateReshapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ReshapeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

void ReshapeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (IsQuantizedOp(node_unit)) {
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Inputs()[0].quant_param);   // x_scale, x_zp
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
  }
  model_builder.AddInitializerToSkip(node_unit.Inputs()[1].node_arg.Name());
}

bool ReshapeOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQReshape;
}

Status ReshapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& initializers(model_builder.GetInitializerTensors());

  auto input = node_unit.Inputs()[0].node_arg.Name();

  const auto& shape_tensor = *initializers.at(node_unit.Inputs()[1].node_arg.Name());
  std::vector<uint8_t> unpacked_tensor;
  ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(shape_tensor, unpacked_tensor));
  const int64_t* raw_shape = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
  const auto size = SafeInt<uint32_t>(shape_tensor.dims()[0]);

  Shape input_shape = shaper[input];
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
        initializers, node_unit.Inputs()[0], node_unit.ModelPath(), x_scale, x_zero_point));
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));
  }

  return AddReshapeOperator(model_builder, node_unit, input, shape);
}

}  // namespace nnapi
}  // namespace onnxruntime
