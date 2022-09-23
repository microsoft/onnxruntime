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

class ResizeOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

void CreateResizeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ResizeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

bool ResizeOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQResize;
}

void ResizeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
  if (IsQuantizedOp(node_unit)) {
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[0].quant_param);               // x_scale, x_zp
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
  }

  // We don't really use ROI here, so add them to skipped list
  model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());  // ROI

  // We will still add scales to the skipped list even sizes are present
  // since there is no use of it, we will not process it later
  model_builder.AddInitializerToSkip(inputs[2].node_arg.Name());  // scales

  if (inputs.size() > 3)
    model_builder.AddInitializerToSkip(inputs[3].node_arg.Name());  // sizes
}

Status ResizeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());
  NodeAttrHelper helper(node_unit);
  const auto& inputs = node_unit.Inputs();
  const auto android_feature_level = model_builder.GetNNAPIFeatureLevel();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  auto input = inputs[0].node_arg.Name();
  bool use_nchw = model_builder.UseNCHW();
  ORT_RETURN_IF_ERROR(IsOpInRequiredLayout(use_nchw, node_unit));

  // Check if the quantization scale and ZP is correct
  if (IsQuantizedOp(node_unit)) {
    float x_scale = 0.0f;
    int32_t x_zero_point = 0;
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        initializers, node_unit.Inputs()[0], node_unit.ModelPath(), x_scale, x_zero_point));
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));
  }

  bool is_linear_resize = helper.Get("mode", "nearest") == "linear";

  int32_t operationCode = is_linear_resize ? ANEURALNETWORKS_RESIZE_BILINEAR
                                           : ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR;

  const auto coord_trans_mode = helper.Get("coordinate_transformation_mode", "half_pixel");
  bool using_half_pixel = coord_trans_mode == "half_pixel";
  bool using_align_corners = coord_trans_mode == "align_corners";

  // if the node domain is NHWC it means all the node inputs are converted to NHWC format by the layout transformer.
  // pick the index for height and width based on the format.
  int h_idx = use_nchw ? 2 : 1;
  int w_idx = use_nchw ? 3 : 2;

  if (inputs.size() == 3) {  // we are using scales
    const auto& scales_name = inputs[2].node_arg.Name();
    const auto& scales_tensor = *initializers.at(scales_name);
    std::vector<uint8_t> unpacked_tensor;
    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(scales_tensor, unpacked_tensor));
    const float* scales_data = reinterpret_cast<const float*>(unpacked_tensor.data());
    ORT_RETURN_IF_ERROR(
        shaper.ResizeUsingScales(input, scales_data[h_idx], scales_data[w_idx], use_nchw, output));
  } else {  // we are using sizes
    const auto& sizes_name = inputs[3].node_arg.Name();
    const auto& sizes_tensor = *initializers.at(sizes_name);
    std::vector<uint8_t> unpacked_tensor;
    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(sizes_tensor, unpacked_tensor));
    const int64_t* sizes_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
    ORT_RETURN_IF_ERROR(
        shaper.ResizeUsingOutputSizes(input, SafeInt<uint32_t>(sizes_data[h_idx]), SafeInt<uint32_t>(sizes_data[w_idx]), use_nchw, output));
  }

  const auto& output_shape = shaper[output];
  int32_t output_h = output_shape[h_idx];
  int32_t output_w = output_shape[w_idx];

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, output_w);
  ADD_SCALAR_OPERAND(model_builder, input_indices, output_h);

  if (android_feature_level > ANEURALNETWORKS_FEATURE_LEVEL_2) {
    // using nchw is only available on API level 29
    ADD_SCALAR_OPERAND(model_builder, input_indices, use_nchw);
  }

  // Currently we only support align_corners and half_pixel on bilinear resize
  // TODO, investigate nearest neighbor resize difference between NNAPI(based on TF) and ONNX
  if (is_linear_resize) {
    if (android_feature_level > ANEURALNETWORKS_FEATURE_LEVEL_3 && (using_align_corners || using_half_pixel)) {
      ADD_SCALAR_OPERAND(model_builder, input_indices, using_align_corners);
      if (using_half_pixel)
        ADD_SCALAR_OPERAND(model_builder, input_indices, using_half_pixel);
    }
  }

  OperandType output_operand_type = operand_types.at(input);
  output_operand_type.SetDimensions(output_shape);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(operationCode, input_indices,
                                                 {output}, {output_operand_type}));

  return Status::OK();
}

}  // namespace nnapi
}  // namespace onnxruntime
