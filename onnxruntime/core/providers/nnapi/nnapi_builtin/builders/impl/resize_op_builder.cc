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

class ResizeOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override;

  // Resize opset 10- is very different than Resize opset 11+, with many key attributes missing
  // We only support Resize opset 11+ here
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 11; }

  bool HasSupportedInputOutputsImpl(
      const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
      const OpSupportCheckParams& /* params */) const override;
  bool IsNodeUnitTypeSupported(const NodeUnit& /* node_unit */) const override { return true; }
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

// Add operator related

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
  const auto android_feature_level = model_builder.GetEffectiveFeatureLevel();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  auto input = inputs[0].node_arg.Name();

  const auto& output_shape = shaper[output];
  const auto& input_shape = shaper[input];

  const bool input_is_nchw = output_shape[1] == input_shape[1];  // not Channel last

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
  int h_idx = input_is_nchw ? 2 : 1;
  int w_idx = input_is_nchw ? 3 : 2;

  int32_t output_h = output_shape[h_idx];
  int32_t output_w = output_shape[w_idx];

  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, output_w);
  ADD_SCALAR_OPERAND(model_builder, input_indices, output_h);

  if (android_feature_level > ANEURALNETWORKS_FEATURE_LEVEL_2) {
    // using nchw is only available on API level 29+
    ADD_SCALAR_OPERAND(model_builder, input_indices, input_is_nchw);
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

// Operator support related

bool ResizeOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQResize;
}

bool ResizeOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                        const OpSupportCheckParams& params) const {
  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  const auto input_rank = input_shape.size();
  if (input_rank != 4) {
    LOGS_DEFAULT(VERBOSE) << "Resize only support 4d shape, input is "
                          << input_rank << "d shape";
    return false;
  }

  {  // check attributes
    NodeAttrHelper helper(node_unit);
    const auto mode = helper.Get("mode", "nearest");
    bool is_linear_resize = mode == "linear";
    bool is_nearest_resize = mode == "nearest";
    if (!is_linear_resize && !is_nearest_resize) {
      LOGS_DEFAULT(VERBOSE) << "Resize unsupported input mode, " << mode;
      return false;
    }

    const auto exclude_outside = helper.Get("exclude_outside", 0);
    if (exclude_outside != 0) {
      LOGS_DEFAULT(VERBOSE) << "Resize does not support exclude_outside for now";
      return false;
    }

    const auto coord_trans_mode = helper.Get("coordinate_transformation_mode", "half_pixel");
    bool using_half_pixel = coord_trans_mode == "half_pixel";
    bool using_align_corners = coord_trans_mode == "align_corners";
    bool using_asymmetric = coord_trans_mode == "asymmetric";
    if (is_linear_resize) {
      if (!using_half_pixel && !using_align_corners && !using_asymmetric) {
        LOGS_DEFAULT(VERBOSE) << "Resize bilinear, unsupported coord_trans_mode, " << coord_trans_mode;
        return false;
      }

      if (params.android_feature_level < 30 && (using_half_pixel || using_align_corners)) {
        LOGS_DEFAULT(VERBOSE)
            << "Resize bilinear only support half_pixel/align_corners on API level 30+, current API level is "
            << params.android_feature_level;
        return false;
      }
    } else {
      // nearest neighbor resizing
      // For resize using nearest neighbor, we only support coord_trans_mode == "asymmetric" && nearest_mode == "floor"
      if (!using_asymmetric) {
        LOGS_DEFAULT(VERBOSE) << "Resize nearest neighbor, unsupported coord_trans_mode, " << coord_trans_mode;
        return false;
      }

      const auto nearest_mode = helper.Get("nearest_mode", "round_prefer_floor");
      if (nearest_mode != "floor") {
        LOGS_DEFAULT(VERBOSE) << "Resize nearest neighbor, unsupported nearest_mode, " << nearest_mode;
        return false;
      }
    }

    // The new feature - antialiasing introduced since opset 18 doesn't have a NNAPI mapping support yet.
    // And a few other new attributes are currently not handled by NNAPI EP, can add support in the future if needed.
    if (node_unit.SinceVersion() >= 18) {
      const auto antialias = helper.Get("antialias", 0);
      const auto axes = helper.Get("axes", std::vector<int64_t>{});
      const auto keep_aspect_ratio_policy = helper.Get("keep_aspect_ratio_policy", "stretch");
      if (antialias != 0) {
        LOGS_DEFAULT(VERBOSE) << "Resize 18+ antialias feature is not currently supported by NNAPI.";
        return false;
      }
      if (!axes.empty()) {
        LOGS_DEFAULT(VERBOSE) << "Resize 18+ axes attribute is not currently supported by NNAPI EP.";
        return false;
      }
      if (keep_aspect_ratio_policy != "stretch") {
        LOGS_DEFAULT(VERBOSE) << "Resize 18+ keep_aspect_ratio_policy attribute is not currently supported by NNAPI EP.";
        return false;
      }
    }
  }

  {  // scales and sizes (if present) must be initializers
    const auto inputs = node_unit.Inputs();
    if (inputs.size() < 3) {
      LOGS_DEFAULT(VERBOSE) << "Input scales or sizes of Resize must be known";
      return false;
    }

    // scales
    bool using_scales = (inputs.size() > 2 && inputs[2].node_arg.Exists());
    if (using_scales && !Contains(initializers, inputs[2].node_arg.Name())) {
      LOGS_DEFAULT(VERBOSE) << "Input scales of Resize must be known";
      return false;
    }

    // sizes
    bool using_sizes = inputs.size() > 3 && inputs[3].node_arg.Exists();
    if (using_sizes && !Contains(initializers, inputs[3].node_arg.Name())) {
      LOGS_DEFAULT(VERBOSE) << "Input sizes of Resize must be known";
      return false;
    }
    bool input_is_nchw = false;
    // haven't a good solution to check layout when scale is 1.0F
    // We want to check if the scales or sizes are not trying to resize on N/C channels here
    if (using_scales) {  // we are using scales
      const auto& scales_tensor = *initializers.at(inputs[2].node_arg.Name());
      Initializer const unpacked_tensor(scales_tensor);
      auto scales_data = unpacked_tensor.DataAsSpan<float>();
      input_is_nchw = scales_data[1] == 1.0F;
      float const scale_n = scales_data[0];
      float const scale_c = input_is_nchw ? scales_data[1] : scales_data[3];
      if (scale_n != 1.0f || scale_c != 1.0f) {
        LOGS_DEFAULT(VERBOSE) << "Scales of N/C channel should be 1"
                              << "Resize of N/C channels are not supported"
                              << ", scale_n, " << scale_n << ", scale_c, " << scale_c;
        return false;
      }
    } else {
      // we are using sizes
      const auto& sizes_name = inputs[3].node_arg.Name();
      const auto& sizes_tensor = *initializers.at(sizes_name);
      Initializer unpacked_tensor(sizes_tensor);
      auto sizes_data = unpacked_tensor.DataAsSpan<int64_t>();

      input_is_nchw = sizes_data[1] == input_shape[1];
      int channel_idx = input_is_nchw ? 1 : 3;
      uint32_t size_n = SafeInt<uint32_t>(sizes_data[0]);
      uint32_t size_c = SafeInt<uint32_t>(sizes_data[channel_idx]);
      if (size_n != input_shape[0] || size_c != input_shape[channel_idx]) {
        LOGS_DEFAULT(VERBOSE) << "Output sizes of N/C channel should match the input sizes, "
                              << "Resize of N/C channels are not supported"
                              << ", input_size_n, " << input_shape[0] << ", output_size_n, " << size_n
                              << ". input_size_c, " << input_shape[channel_idx] << ", output_size_c, " << size_c;
        return false;
      }
    }

    if (input_is_nchw && params.android_feature_level <= ANEURALNETWORKS_FEATURE_LEVEL_2) {
      LOGS_DEFAULT(VERBOSE) << "android_feature_level below 29 does not support nchw Resize.";
      return false;
    }
  }

  return true;
}

int32_t ResizeOpBuilder::GetMinSupportedNNAPIFeatureLevel(const NodeUnit& node_unit,
                                                          const OpSupportCheckParams& /* params */) const {
  int32_t input_type;

  // This should not happen, but if it happens make sure this will require an impossible version
  if (!GetType(node_unit.Inputs()[0].node_arg, input_type))
    return std::numeric_limits<int32_t>::max();

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8)
    return ANEURALNETWORKS_FEATURE_LEVEL_3;

  return ANEURALNETWORKS_FEATURE_LEVEL_2;
}

bool ResizeOpBuilder::HasSupportedInputOutputsImpl(
    const InitializedTensorSet& initializers, const NodeUnit& node_unit,
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
    if (!IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kInput))
      return false;

    if (!IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput))
      return false;
  }

  return true;
}

void CreateResizeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ResizeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace nnapi
}  // namespace onnxruntime
