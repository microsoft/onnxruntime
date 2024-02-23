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

class ConvOpBuilder : public BaseOpBuilder {
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
                                           const OpSupportCheckParams& params) const override {
    return params.use_nchw ? ANEURALNETWORKS_FEATURE_LEVEL_3 : ANEURALNETWORKS_FEATURE_LEVEL_2;
  }

  bool HasSupportedInputOutputsImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                    const OpSupportCheckParams& params) const override;
  bool IsNodeUnitTypeSupported(const NodeUnit& /* node_unit */) const override { return true; }
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

// Add operator related

void ConvOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
  // skip the weight for conv as we need to transpose
  if (IsQuantizedOp(node_unit)) {
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[0].quant_param);               // x_scale, x_zp
    AddInputToSkip(model_builder, inputs[1]);                                                    // w, w_scale, w_zp
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
    if (inputs.size() > 2)
      AddInputToSkip(model_builder, inputs[2]);  // B, B_scale, B_zp
  } else {
    model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());  // w
  }
}

Status ConvOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());
  NodeAttrHelper helper(node_unit);
  const auto inputs = node_unit.Inputs();
  bool is_quant_conv = IsQuantizedOp(node_unit);

  // onnx strides are in the order height, width
  // while nnapi strides are in the order width, height
  const auto onnx_strides = helper.Get("strides", std::vector<int>{1, 1});

  // onnx pads are in the order top, left, bottom, right
  // while nnapi pads is in the order left, right, top, bottom
  auto onnx_pads = helper.Get("pads", std::vector<int>{0, 0, 0, 0});

  // onnx dilations is in the order height, width
  // while nnapi dilations are in the order width, height
  const auto onnx_dilations = helper.Get("dilations", std::vector<int>{1, 1});
  const auto group = helper.Get("group", 1);

  auto input = inputs[0].node_arg.Name();
  bool use_nchw = model_builder.UseNCHW();
  ORT_RETURN_IF_ERROR(IsOpInRequiredLayout(use_nchw, node_unit));

  const auto& weight = inputs[1].node_arg.Name();
  const auto& weight_tensor = *initializers.at(weight);
  auto conv_type = GetConvType(node_unit, model_builder.GetInitializerTensors());
  bool conv_2d = (conv_type == ConvType::Regular),
       depthwise_conv_2d = (conv_type == ConvType::Depthwise),
       grouped_conv_2d = (conv_type == ConvType::Grouped);

  float x_scale = 0.0f,
        w_scale = 0.0f,
        y_scale = 0.0f;
  int32_t x_zero_point = 0,
          w_zero_point = 0,
          y_zero_point = 0;

  // this is for per-channel quantization weights
  optional<std::vector<float>> w_scales;
  bool is_per_tensor_u8s8 = false;
  if (is_quant_conv) {
    ORT_RETURN_IF_ERROR(GetConvMatMulOpQuantizationScaleAndZeroPoint(model_builder, node_unit,
                                                                     x_scale, w_scale, y_scale,
                                                                     x_zero_point, w_zero_point, y_zero_point,
                                                                     w_scales, is_per_tensor_u8s8));
  }

  Shape onnx_weight_shape;
  for (auto dim : weight_tensor.dims())
    onnx_weight_shape.push_back(SafeInt<uint32_t>(dim));

  Type onnx_weight_type;
  switch (weight_tensor.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      onnx_weight_type = Type::TENSOR_FLOAT32;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      onnx_weight_type = Type::TENSOR_QUANT8_ASYMM;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      // We support both per-tensor and per-channel u8s8
      // For per-tensor u8s8 we will convert the int8 weight to uint8
      if (is_per_tensor_u8s8) {
        // Per-Tensor u8s8
        onnx_weight_type = Type::TENSOR_QUANT8_ASYMM;
      } else {
        // Per-Channel u8s8
        onnx_weight_type = Type::TENSOR_QUANT8_SYMM_PER_CHANNEL;
      }
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The initializer of graph ", weight, " doesn't have valid type: ",
                             weight_tensor.data_type());
  }

  // Get weight operand type
  // Per-channel quantized weight is handled differently
  OperandType onnx_weight_operand_type =
      (is_quant_conv && w_scales.has_value())
          ? OperandType{onnx_weight_type, onnx_weight_shape,
                        SymmPerChannelQuantParams{w_scales.value(),
                                                  depthwise_conv_2d ? 3u : 0u}}  // channelDim is 3 for depthwise-conv
          : OperandType{onnx_weight_type, onnx_weight_shape, w_scale, w_zero_point};

  // Pre-process weights
  if (conv_2d || grouped_conv_2d) {
    ORT_RETURN_IF_ERROR(AddInitializerInNewLayout(
        model_builder, weight, onnx_weight_operand_type, L_0231, is_per_tensor_u8s8));
  } else {  // depthwise_conv_2d
    ORT_RETURN_IF_ERROR(AddInitializerInNewLayout(
        model_builder, weight, onnx_weight_operand_type, L_1230, is_per_tensor_u8s8));
  }

  if (is_quant_conv) {
    // Verify if the scale and zero point matchs from onnx input/weight and nnapi input/weight
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));
    ORT_RETURN_IF_ERROR(IsValidConvWeightQuantizedType(model_builder, weight, w_scale, w_zero_point, w_scales));
  }

  bool hasBias = (inputs.size() > 2);
  std::string bias = hasBias ? inputs[2].node_arg.Name() : weight + "_bias";
  if (!hasBias) {
    const auto weight_dimen = shaper[weight];
    Shape bias_dimen;
    if (conv_2d || grouped_conv_2d)
      bias_dimen = {weight_dimen[0]};
    else
      bias_dimen = {weight_dimen[3]};

    const auto& weight_type = operand_types.at(weight).type;
    if (weight_type == Type::TENSOR_FLOAT32) {
      std::vector<float> buffer(bias_dimen[0], 0.0f);
      OperandType bias_operand_type(Type::TENSOR_FLOAT32, bias_dimen, x_scale * w_scale);
      ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
    } else if (weight_type == Type::TENSOR_QUANT8_ASYMM || weight_type == Type::TENSOR_QUANT8_SYMM_PER_CHANNEL) {
      std::vector<int32_t> buffer(bias_dimen[0], 0);
      OperandType bias_operand_type(Type::TENSOR_INT32, bias_dimen, x_scale * w_scale);
      ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown weight type ", TypeToStr(weight_type));
    }
  } else if (is_quant_conv) {
    // QLinearConv's bias type need special handling to add scale for quantization input
    const auto& bias_tensor = *model_builder.GetInitializerTensors().at(bias);
    ORT_RETURN_IF_NOT(bias_tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32,
                      "bias of QLinearConv should be int32, actual type: ", bias_tensor.data_type());
    Shape bias_dimen;
    for (auto dim : bias_tensor.dims())
      bias_dimen.push_back(SafeInt<uint32_t>(dim));
    Initializer unpacked_tensor(bias_tensor);
    OperandType bias_operand_type(Type::TENSOR_INT32, bias_dimen, x_scale * w_scale);
    ORT_RETURN_IF_ERROR(
        model_builder.AddOperandFromPersistMemoryBuffer(bias, unpacked_tensor.data<int32_t>(), bias_operand_type));
  }

  const auto auto_pad_type = StringToAutoPadType(helper.Get("auto_pad", "NOTSET"));
  bool use_auto_pad = false;
  int32_t nnapi_padding_code = ANEURALNETWORKS_PADDING_SAME;
  const auto input_shape = shaper[input];
  const auto& kernel_shape = shaper[weight];
  const auto weight_size_y = kernel_shape[1];
  const auto weight_size_x = kernel_shape[2];
  ORT_RETURN_IF_ERROR(
      HandleAutoPad(input_shape, weight_size_y, weight_size_x,
                    onnx_strides, onnx_dilations,
                    auto_pad_type, use_nchw,
                    onnx_pads, nnapi_padding_code, use_auto_pad));

  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  input_indices.push_back(operand_indices.at(weight));
  input_indices.push_back(operand_indices.at(bias));

  if (use_auto_pad) {
    ADD_SCALAR_OPERAND(model_builder, input_indices, nnapi_padding_code);
  } else {
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[1]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[3]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[0]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[2]);
  }

  ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_strides[1]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_strides[0]);

  if (!conv_2d) {
    if (depthwise_conv_2d) {
      int32_t depthwiseMultiplier = shaper[weight][3] / group;
      ADD_SCALAR_OPERAND(model_builder, input_indices, depthwiseMultiplier);
    } else {  // grouped_conv_2d
      ADD_SCALAR_OPERAND(model_builder, input_indices, group);
    }
  }

  int32_t fuse_code = model_builder.FindActivation(node_unit);
  ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);

  if (model_builder.GetEffectiveFeatureLevel() > ANEURALNETWORKS_FEATURE_LEVEL_2) {
    ADD_SCALAR_OPERAND(model_builder, input_indices, use_nchw);

    // 1. NNAPI Grouped Conv does not support dilations
    // 2. There is a bug in NNAPI (not sure NNAPI itself or Qualcomm Hexagon driver),
    //    setting dilation (even it is the default (1,1)) will make the execution fall back to CPU
    //    so if dilations == (1,1) we simply ignore it
    if (!grouped_conv_2d &&
        (onnx_dilations[1] != 1 || onnx_dilations[0] != 1)) {
      ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_dilations[1]);
      ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_dilations[0]);
    }
  }

  int32_t operationCode;
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  if (conv_2d || grouped_conv_2d) {
    operationCode = conv_2d ? ANEURALNETWORKS_CONV_2D
                            : ANEURALNETWORKS_GROUPED_CONV_2D;
  } else {  // depthwise_conv_2d
    operationCode = ANEURALNETWORKS_DEPTHWISE_CONV_2D;
  }

  const OperandType output_operand_type(operand_types.at(input).type, shaper[output], y_scale, y_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(operationCode, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

bool ConvOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  return IsQuantizedConv(GetQuantizedOpType(node_unit));
}

bool ConvOpBuilder::HasSupportedInputOutputsImpl(
    const GraphViewer& graph_viewer, const NodeUnit& node_unit,
    const OpSupportCheckParams& params) const {
  if (!IsQuantizedOp(node_unit))
    return BaseOpBuilder::HasSupportedInputOutputsImpl(graph_viewer, node_unit, params);

  // QLinearConv only supports input of uint8 for now
  if (!HasValidBinaryOpQuantizedInputTypes(node_unit))
    return false;

  if (!IsQuantizedIOSupported(graph_viewer, node_unit, {0, 1}, params, ArgType::kInput))
    return false;

  if (!IsQuantizedIOSupported(graph_viewer, node_unit, {0}, params, ArgType::kOutput))
    return false;

  return true;
}

// Operator support related

bool ConvOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                      const OpSupportCheckParams& params) const {
  const auto& op_type = node_unit.OpType();
  bool is_quant_conv = IsQuantizedOp(node_unit);

  // We don't support nhwc com.microsoft.QLinearConv for now
  if (is_quant_conv && node_unit.Domain() == kMSDomain) {
    LOGS_DEFAULT(VERBOSE) << "com.microsoft.QLinearConv is not supported";
    return false;
  }

  const auto& inputs = node_unit.Inputs();
  NodeAttrHelper helper(node_unit);
  const auto group = helper.Get("group", 1);
  const auto weight_name = inputs[1].node_arg.Name();
  const auto* weight = graph_viewer.GetConstantInitializer(weight_name);
  if (weight) {
    const auto& tensor = *weight;
    if (tensor.dims().size() != 4) {
      LOGS_DEFAULT(VERBOSE) << "Only conv 2d is supported.";
      return false;
    }

    const auto onnx_dilations = helper.Get("dilations", std::vector<int>{1, 1});
    if (onnx_dilations != std::vector<int>{1, 1}) {
      if (group != 1 && tensor.dims()[1] != 1) {
        LOGS_DEFAULT(VERBOSE) << "dilation is not supported on grouped conv";
        return false;
      }

      if (params.android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
        LOGS_DEFAULT(VERBOSE) << op_type << " dilations is only supported on Android API level 29+, "
                              << "actual API level: " << params.android_feature_level;
        return false;
      }
    }
  } else {
    LOGS_DEFAULT(VERBOSE) << "The weight of convolution must be a constant initializer";
    return false;
  }

  if (is_quant_conv) {
    if (inputs.size() > 2 && !graph_viewer.GetConstantInitializer(inputs[2].node_arg.Name())) {
      LOGS_DEFAULT(VERBOSE) << "Bias of QLinearConv must be a constant initializer";
      return false;
    }
  }

  return true;
}

void CreateConvOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<ConvOpBuilder>(
      op_type, op_registrations,
      {
          "Conv",
          "QLinearConv",
      });
}

}  // namespace nnapi
}  // namespace onnxruntime
