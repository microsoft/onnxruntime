// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "op_builder.h"

#include <onnx/onnx_pb.h>

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/providers/cpu/tensor/slice_helper.h"
#include "helper.h"
#include "model_builder.h"
#include "op_support_checker.h"

using namespace android::nn::wrapper;

namespace onnxruntime {
namespace nnapi {

#pragma region helpers

struct OpBuilderRegistrations {
  std::vector<std::unique_ptr<IOpBuilder>> builders;
  std::unordered_map<std::string, const IOpBuilder*> op_builder_map;
};

#define ADD_SCALAR_OPERAND(model_builder, input_indices, scalar_value)             \
  {                                                                                \
    uint32_t _index = 0;                                                           \
    ORT_RETURN_IF_ERROR(model_builder.AddOperandFromScalar(scalar_value, _index)); \
    input_indices.push_back(_index);                                               \
  }

Status AddTransposeOperator(ModelBuilder& model_builder,
                            const std::string& input,
                            const std::string& perm_name,
                            std::vector<int32_t> perm,
                            const std::string& output,
                            bool output_is_nhwc) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));  // input

  Shape perm_dimen = {SafeInt<uint32_t>(perm.size())};
  OperandType perm_operand_type(Type::TENSOR_INT32, perm_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(perm_name, perm.data(), perm_operand_type));
  uint32_t perm_idx = operand_indices.at(perm_name);

  input_indices.push_back(perm_idx);  // permutation
  ORT_RETURN_IF_ERROR(shaper.Transpose(input, perm, output));
  OperandType output_operand_type = operand_types.at(input);
  output_operand_type.SetDimensions(shaper[output]);
  return model_builder.AddOperation(ANEURALNETWORKS_TRANSPOSE, input_indices, {output},
                                    {output_operand_type}, {output_is_nhwc});
}

Status TransposeBetweenNCHWAndNHWC(ModelBuilder& model_builder,
                                   const std::string& input,
                                   const std::string& output,
                                   bool nchw_to_nhwc) {
  ORT_RETURN_IF_NOT(!model_builder.UseNCHW(), "model_builder.UseNCHW() is on");
  const auto& shaper(model_builder.GetShaper());
  ORT_RETURN_IF_NOT(4 == shaper[input].size(),
                    "TransposeBetweenNCHWAndNHWC input has to be a 4d tensor, actual dimensions: ", shaper[input].size());

  std::string perm_name;
  std::vector<int32_t> perm;
  if (nchw_to_nhwc) {
    perm_name = model_builder.GetUniqueName(input + "nchw_to_nhwc_perm");
    perm = {0, 2, 3, 1};
  } else {  // nhwc_to_nchw
    perm_name = model_builder.GetUniqueName(input + "nhwc_to_nchw_perm");
    perm = {0, 3, 1, 2};
  }

  ORT_RETURN_IF_ERROR(AddTransposeOperator(model_builder, input, perm_name, perm, output, nchw_to_nhwc));

  if (nchw_to_nhwc) {
    ORT_RETURN_IF_ERROR(model_builder.SetNCHWToNHWCOperandMap(input, output));
  } else {  // nhwc_to_nchw
    ORT_RETURN_IF_ERROR(model_builder.SetNHWCToNCHWOperandMap(input, output));
  }

  LOGS_DEFAULT(VERBOSE) << "Operand [" << input << "] with shape "
                        << Shape2String(shaper[input])
                        << " is transposed "
                        << (nchw_to_nhwc ? "nchw_to_nhwc" : "nhwc_to_nchw")
                        << " to [" << output << "] with shape "
                        << Shape2String(shaper[output]);

  return Status::OK();
}

Status TransposeNHWCToNCHW(ModelBuilder& model_builder,
                           const std::string& input,
                           const std::string& output) {
  return TransposeBetweenNCHWAndNHWC(model_builder, input, output, false /* nchw_to_nhwc */);
}

Status TransposeNCHWToNHWC(ModelBuilder& model_builder,
                           const std::string& input,
                           const std::string& output) {
  return TransposeBetweenNCHWAndNHWC(model_builder, input, output, true /* nchw_to_nhwc */);
}

// Convert the input from nchw to nhwc
// Caller should ensure input is currently in nchw format using ModelBuilder::IsOperandNHWC
Status GetNHWCInput(ModelBuilder& model_builder, const NodeUnit& node_unit, size_t input_index, std::string& nhwc_input) {
  const auto& nchw_input = node_unit.Inputs()[input_index].node_arg.Name();
  if (!model_builder.GetNHWCOperand(nchw_input, nhwc_input)) {
    nhwc_input = model_builder.GetUniqueName(nchw_input + "_nchw_to_nhwc");
    ORT_RETURN_IF_ERROR(TransposeNCHWToNHWC(model_builder, nchw_input, nhwc_input));
  }
  return Status::OK();
}

// Convert the input from nhwc to nchw
// Caller should ensure input is currently in nhwc format using ModelBuilder::IsOperandNHWC
Status GetNCHWInput(ModelBuilder& model_builder, const NodeUnit& node_unit, size_t input_index, std::string& nchw_input) {
  const auto& nhwc_input = node_unit.Inputs()[input_index].node_arg.Name();
  if (!model_builder.GetNCHWOperand(nhwc_input, nchw_input)) {
    nchw_input = model_builder.GetUniqueName(nhwc_input + "_nhwc_to_nchw");
    ORT_RETURN_IF_ERROR(TransposeNHWCToNCHW(model_builder, nhwc_input, nchw_input));
  }
  return Status::OK();
}

// Transpose layouts if necessary for element wise operators with 2 inputs
// and return the layout type of output tensor
// If both inputs have same layout, the output will have the same layout
// Otherwise we will need transpose the nhwc input back to nchw, and output will be nchw
Status TransposeBinaryOpInputLayout(ModelBuilder& model_builder, const NodeUnit& node_unit,
                                    std::string& input1, std::string& input2,
                                    bool& output_is_nhwc) {
  bool input1_is_nhwc = model_builder.IsOperandNHWC(input1);
  bool input2_is_nhwc = model_builder.IsOperandNHWC(input2);
  output_is_nhwc = false;

  if (input1_is_nhwc == input2_is_nhwc) {
    output_is_nhwc = input1_is_nhwc;
  } else if (input1_is_nhwc) {
    // need transpose input1 back to nchw
    ORT_RETURN_IF_ERROR(GetNCHWInput(model_builder, node_unit, 0, input1));
  } else {  // input2_is_nhwc
    // need transpose input2 back to nchw
    ORT_RETURN_IF_ERROR(GetNCHWInput(model_builder, node_unit, 1, input2));
  }

  return Status::OK();
}

static Status AddBinaryOperator(int32_t op_type,
                                ModelBuilder& model_builder,
                                const std::string& input1,
                                const std::string& input2,
                                bool add_activation,
                                int32_t fuse_code,
                                const std::string& output,
                                bool output_is_nhwc,
                                float output_scale = 0.0f,
                                int32_t output_zero_point = 0) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));  // input 1
  input_indices.push_back(operand_indices.at(input2));  // input 2

  if (add_activation) {
    ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);
  }

  ORT_RETURN_IF_ERROR(shaper.Eltwise(input1, input2, output));
  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output],
                                        output_scale, output_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_type, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

static Status AddSqueezeOp(ModelBuilder& model_builder,
                           const std::string& node_name,
                           const std::string& input, const std::string& output,
                           std::vector<int32_t> axes) {
  if (model_builder.GetNNAPIFeatureLevel() < ANEURALNETWORKS_FEATURE_LEVEL_2) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, FAIL, "Squeeze is not supported on API level ", model_builder.GetNNAPIFeatureLevel());
  }

  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input_shape(shaper[input]);
  auto input_dims = input_shape.size();
  for (auto& axis : axes) {
    axis = static_cast<int32_t>(HandleNegativeAxis(axis, input_dims));
  }

  // Despite the spec of ANEURALNETWORKS_SQUEEZE at
  // https://developer.android.com/ndk/reference/group/neural-networks
  // states, that the axes (input 1 of ANEURALNETWORKS_SQUEEZE) is optional.
  //
  // The actual code of NNAPI requires the axes to be provided
  // https://android.googlesource.com/platform/frameworks/ml/+/master/nn/common/operations/Squeeze.cpp#31
  if (axes.empty()) {  // Squeeze all
    for (size_t i = 0; i < input_dims; i++) {
      if (input_shape[i] == 1)
        axes.push_back(i);
    }
  }

  const auto axes_name = model_builder.GetUniqueName(node_name + input + "_axes");
  Shape axes_dimen = {static_cast<uint32_t>(axes.size())};
  const OperandType axes_operand_type(Type::TENSOR_INT32, axes_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(axes_name, axes.data(), axes_operand_type));

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));      // input
  input_indices.push_back(operand_indices.at(axes_name));  // axes

  ORT_RETURN_IF_ERROR(shaper.Squeeze(input, axes, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_SQUEEZE, input_indices,
                                                 {output}, {output_operand_type}, {false}));
  return Status::OK();
}

enum DataLayout {
  L_0231 = 0,
  L_1230 = 1,
};

// This is primarily used for adding the weight (an initializer) of Conv/QlinearConv
// And perform layout change from ONNX -> NNAPI
// If is_per_tensor_u8s8 is true, the QlinearConv is per-tensor u8s8 (input X is unsigned int8
// and weight W is signed int8 and it is per-tensor (NOT per-channel) quantized), in this case,
// since NNAPI requires X and W to be same type for per-tensor quantization,
// the initializer tensor W will be converted from int8 to uint8 by flip each byte by XOR 0x80
// byte ^ 0x80 == byte + 128
static Status AddInitializerInNewLayout(ModelBuilder& model_builder,
                                        const std::string& name,
                                        const OperandType& source_operand_type,
                                        DataLayout new_layout,
                                        bool is_per_tensor_u8s8) {
  const auto& tensor = *model_builder.GetInitializerTensors().at(name);
  const Shape& shape = source_operand_type.dimensions;
  ORT_RETURN_IF_NOT(shape.size() == 4,
                    "The initializer is not 4D: ", name, " actual dim ", shape.size());

  // TODO support other data types
  const uint8_t* src = nullptr;
  std::vector<uint8_t> unpacked_tensor;

  switch (tensor.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      ORT_RETURN_IF_ERROR(
          onnxruntime::utils::UnpackInitializerData(tensor, model_builder.GetGraphViewer().ModelPath(),
                                                    unpacked_tensor));
      src = unpacked_tensor.data();
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The initializer of graph ", name,
                             " doesn't have valid type: ", tensor.data_type());
  }

  const auto out_t = shape[0], in_t = shape[1],
             h_t = shape[2], w_t = shape[3];
  Shape dest_shape;
  if (new_layout == L_0231)
    dest_shape = {out_t, h_t, w_t, in_t};  // L_0231
  else
    dest_shape = {in_t, h_t, w_t, out_t};  // L_1230 for depthwise conv weight

  OperandType operand_type = source_operand_type;
  operand_type.SetDimensions(dest_shape);
  std::unique_ptr<uint8_t[]> buffer_holder(new uint8_t[operand_type.GetOperandBlobByteSize()]);
  uint8_t* buffer = buffer_holder.get();
  size_t element_size = operand_type.GetElementByteSize();

  uint8_t bit_flip_val = is_per_tensor_u8s8 ? 0x80 : 0;
  for (uint32_t out = 0; out < out_t; out++) {
    for (uint32_t in = 0; in < in_t; in++) {
      for (uint32_t h = 0; h < h_t; h++) {
        for (uint32_t w = 0; w < w_t; w++) {
          auto onnx_idx = out * in_t * h_t * w_t +
                          in * h_t * w_t +
                          h * w_t +
                          w;

          uint32_t nnapi_idx;
          if (new_layout == L_0231) {  // L_0231
            nnapi_idx = out * h_t * w_t * in_t +
                        h * w_t * in_t +
                        w * in_t +
                        in;
          } else {  // L_1230 for depthwise conv weight
            nnapi_idx = in * h_t * w_t * out_t +
                        h * w_t * out_t +
                        w * out_t +
                        out;
          }

          for (size_t i = 0; i < element_size; i++) {
            buffer[element_size * nnapi_idx + i] = src[element_size * onnx_idx + i] ^ bit_flip_val;
          }
        }
      }
    }
  }

  return model_builder.AddOperandFromPersistMemoryBuffer(name, &buffer[0], operand_type);
}

// This is primarily used for adding the input B (an initializer) of MatMul/QlinearMatMul/Gemm (not transposed)
// and transpose it, since for NNAPI only supports A*B'
//
// If is_per_tensor_u8s8 is true, the QlinearMatMul is per-tensor u8s8 (input A is unsigned int8
// and input B is signed int8), in this case, since NNAPI requires A and B to be same type,
// the initializer tensor B will be converted from int8 to uint8 by flip each byte by XOR 0x80
// byte ^ 0x80 == byte + 128
static Status AddInitializerTransposed(ModelBuilder& model_builder,
                                       const OperandType& source_operand_type,
                                       const std::string& name,
                                       bool is_per_tensor_u8s8) {
  const auto& tensor = *model_builder.GetInitializerTensors().at(name);
  const Shape& shape = source_operand_type.dimensions;

  ORT_RETURN_IF_NOT(shape.size() == 2,
                    "The initializer is not 2D: ", name, " actual dim ", shape.size());

  // TODO support other data types
  const uint8_t* src = nullptr;
  std::vector<uint8_t> unpacked_tensor;
  switch (tensor.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      ORT_RETURN_IF_ERROR(
          onnxruntime::utils::UnpackInitializerData(tensor, model_builder.GetGraphViewer().ModelPath(),
                                                    unpacked_tensor));
      src = unpacked_tensor.data();
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The initializer of graph ", name,
                             " doesn't have valid type: ", tensor.data_type());
  }

  const auto x_t = shape[0], y_t = shape[1];
  Shape dest_shape = {y_t, x_t};
  OperandType operand_type = source_operand_type;
  operand_type.SetDimensions(dest_shape);
  std::unique_ptr<uint8_t[]> buffer_holder(new uint8_t[operand_type.GetOperandBlobByteSize()]);
  uint8_t* buffer = buffer_holder.get();
  size_t element_size = operand_type.GetElementByteSize();
  uint8_t bit_flip_val = is_per_tensor_u8s8 ? 0x80 : 0;
  for (uint32_t x = 0; x < x_t; x++) {
    for (uint32_t y = 0; y < y_t; y++) {
      for (size_t i = 0; i < element_size; i++) {
        buffer[element_size * (y * x_t + x) + i] = src[element_size * (x * y_t + y) + i] ^ bit_flip_val;
      }
    }
  }

  return model_builder.AddOperandFromPersistMemoryBuffer(name, &buffer[0], operand_type);
}

static Status ComputeConvPads(
    const Shape& input_dimen,
    const uint32_t weight_size_y, const uint32_t weight_size_x,
    const std::vector<int32_t>& onnx_pads, const std::vector<int32_t>& onnx_strides, const std::vector<int32_t>& onnx_dilations,
    AutoPadType auto_pad_type, bool nchw,
    std::vector<int32_t>& pads_out) {
  const int32_t input_size_y = nchw ? input_dimen[2] : input_dimen[1];
  const int32_t input_size_x = nchw ? input_dimen[3] : input_dimen[2];
  const int32_t stride_y = onnx_strides[0];
  const int32_t stride_x = onnx_strides[1];
  const int32_t dilation_y = onnx_dilations[0];
  const int32_t dilation_x = onnx_dilations[1];

  int64_t padding_top = onnx_pads[0];
  int64_t padding_bottom = onnx_pads[2];
  int64_t padding_left = onnx_pads[1];
  int64_t padding_right = onnx_pads[3];

  ORT_RETURN_IF_ERROR(ComputePad(input_size_y,
                                 stride_y, weight_size_y, dilation_y,
                                 auto_pad_type,
                                 padding_top, padding_bottom));
  ORT_RETURN_IF_ERROR(ComputePad(input_size_x,
                                 stride_x, weight_size_x, dilation_x,
                                 auto_pad_type,
                                 padding_left, padding_right));

  pads_out = {static_cast<int32_t>(padding_top), static_cast<int32_t>(padding_left),
              static_cast<int32_t>(padding_bottom), static_cast<int32_t>(padding_right)};

  return Status::OK();
}

static Status HandleAutoPad(const Shape& input_shape,
                            const uint32_t weight_size_y,
                            const uint32_t weight_size_x,
                            const std::vector<int32_t>& onnx_strides,
                            const std::vector<int32_t>& onnx_dilations,
                            AutoPadType auto_pad_type,
                            bool use_nchw,
                            std::vector<int32_t>& onnx_pads,
                            int32_t& nnapi_padding_code,
                            bool& use_auto_pad) {
  use_auto_pad = false;
  if (auto_pad_type != AutoPadType::NOTSET) {
    ORT_RETURN_IF_ERROR(ComputeConvPads(input_shape, weight_size_y, weight_size_x,
                                        onnx_pads, onnx_strides, onnx_dilations,
                                        auto_pad_type, use_nchw,
                                        onnx_pads));

    if (AutoPadType::VALID == auto_pad_type || AutoPadType::SAME_UPPER == auto_pad_type) {
      use_auto_pad = true;
      nnapi_padding_code = (AutoPadType::VALID == auto_pad_type) ? ANEURALNETWORKS_PADDING_VALID
                                                                 : ANEURALNETWORKS_PADDING_SAME;
    }
  } else if (onnx_dilations == std::vector<int32_t>{1, 1}) {
    // Since NNAPI runs more efficiently using auto_pad, we try to map the NOTSET padding to auto_pad
    std::vector<int32_t> same_upper_pads;
    ORT_RETURN_IF_ERROR(ComputeConvPads(input_shape, weight_size_y, weight_size_x,
                                        onnx_pads, onnx_strides, onnx_dilations,
                                        AutoPadType::SAME_UPPER, use_nchw,
                                        same_upper_pads));
    if (onnx_pads == same_upper_pads) {
      use_auto_pad = true;
      nnapi_padding_code = ANEURALNETWORKS_PADDING_SAME;
    }
  }

  return Status::OK();
}

// Get scales and zero points for the qlinear binary ops (which has 2 input and 1 output)
// QLinearConv, QLinearMatmul, QLinearAdd
// a, b are inputs, and y is output
static Status GetBinaryOpQuantizationScaleAndZeroPoint(
    const InitializedTensorSet& initializers, const NodeUnit& node_unit,
    float& a_scale, float& b_scale, float& y_scale,
    int32_t& a_zero_point, int32_t& b_zero_point, int32_t& y_zero_point) {
  ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
      initializers, node_unit.Inputs()[0], node_unit.ModelPath(), a_scale, a_zero_point));
  ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
      initializers, node_unit.Inputs()[1], node_unit.ModelPath(), b_scale, b_zero_point));
  ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
      initializers, node_unit.Outputs()[0], node_unit.ModelPath(), y_scale, y_zero_point));

  return Status::OK();
}

// Get scale and zero point for
// [QlinearConv] input, weight, output
// [QlinearMatMul] A, B, Y
//
// In case of u8s8 (input/A is uint8 and weight/B is int8)
// If the QlinearConv is using per-channel u8s8, return the scales vector
// If the Qlinear[Conv/MatMul] is using per-tensor u8s8, the weight/B tensor
// will be convert to uint8 later, will return the same scale and 128 as zero point
// Also will set is_per_tensor_u8s8 to true to be used later
static Status GetConvMatMulOpQuantizationScaleAndZeroPoint(
    const ModelBuilder& model_builder, const NodeUnit& node_unit,
    float& a_scale, float& w_scale, float& y_scale,
    int32_t& a_zero_point, int32_t& w_zero_point, int32_t& y_zero_point,
    optional<std::vector<float>>& w_scales, bool& is_per_tensor_u8s8) {
  is_per_tensor_u8s8 = false;
  const auto& initializers(model_builder.GetInitializerTensors());
  // Get scale and zero points
  // We will handle per-channel weight scale and zero point later
  ORT_RETURN_IF_ERROR(
      GetBinaryOpQuantizationScaleAndZeroPoint(initializers, node_unit,
                                               a_scale, w_scale, y_scale,
                                               a_zero_point, w_zero_point, y_zero_point));

  const auto& inputs = node_unit.Inputs();
  const auto& weight_tensor = *initializers.at(inputs[1].node_arg.Name());

  // We are done here is this is u8u8 QLinearConv
  if (weight_tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT8)
    return Status::OK();

  // This is per-tensor u8s8
  // NNAPI does not support per-tensor u8s8
  // For this case we will need to convert the int8 weight tensor to uint8
  // And have same scale and 128 as zero point
  // The conversion of the weight tensor itself will be done in the OpBuilder
  const auto& scale_tensor = *initializers.at(inputs[1].quant_param->scale.Name());
  int64_t scale_dim = scale_tensor.dims().empty() ? 1 : scale_tensor.dims()[0];
  if (scale_dim == 1) {
    w_zero_point = 128;
    is_per_tensor_u8s8 = true;
    return Status::OK();
  }

  // Now we have u8s8 per-channel QlinearConv
  // u8s8 QlinearConv always have 0 as zero point so we are not getting it here
  // and we do not use w_scale here, so we reset them back to 0
  w_scale = 0.0f;
  w_zero_point = 0;

  // We need to copy the 1d scales array for per-channel quantization
  std::vector<uint8_t> unpacked_tensor;
  ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(scale_tensor, unpacked_tensor));
  const float* scales = reinterpret_cast<const float*>(unpacked_tensor.data());
  const size_t scales_size = scale_tensor.dims().empty() ? 1 : scale_tensor.dims()[0];
  std::vector<float> scales_vec(scales, scales + scales_size);
  w_scales = onnxruntime::make_optional(std::move(scales_vec));
  return Status::OK();
}

// NNAPI has the quantization scale and zero point embedded in the ANeuralNetworksOperandType
// ONNX has the quantization scale and zero point as the inputs of the qlinear operators
// We want to verify the scale and zeropoint of the ONNX inputs matches the values embedded in the NNAPI inputs
static Status IsValidInputQuantizedType(const ModelBuilder& model_builder,
                                        const std::string& input_name,
                                        float scale,
                                        int32_t zero_point) {
  const OperandType& input_operand_type = model_builder.GetOperandTypes().at(input_name);
  if (input_operand_type.operandType.scale != scale) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input [", input_name,
                           "] NNAPI input scale: ", input_operand_type.operandType.scale,
                           ", ONNX input scale: ", scale);
  }

  if (input_operand_type.operandType.zeroPoint != zero_point) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input [", input_name,
                           "] NNNAPI input zero point: ", input_operand_type.operandType.zeroPoint,
                           ", ONNX input zero point: ", zero_point);
  }

  return Status::OK();
}

static Status IsValidConvWeightQuantizedType(const ModelBuilder& model_builder,
                                             const std::string& input_name,
                                             float scale,
                                             int32_t zero_point,
                                             const optional<std::vector<float>>& scales) {
  // first verify as the weight has no per-channel quantization
  ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input_name, scale, zero_point));

  if (scales) {
    const OperandType& input_operand_type = model_builder.GetOperandTypes().at(input_name);
    if (!input_operand_type.channelQuant) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input [", input_name, "] has no channelQuant");
    }

    if (input_operand_type.channelQuant.value().scales != scales.value()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input [", input_name, "] has mismatch scales between onnx and NNAPI");
    }
  }

  return Status::OK();
}

static void AddQuantizationScaleAndZeroPointToSkip(ModelBuilder& model_builder,
                                                   const NodeUnitIODef::QuantParam& quant_param) {
  // If we reach here, we assume the io_def has quant_param
  model_builder.AddInitializerToSkip(quant_param.scale.Name());  // scale
  LOGS_DEFAULT(VERBOSE) << quant_param.scale.Name() << "is skipped";
  if (quant_param.zero_point) {
    model_builder.AddInitializerToSkip(quant_param.zero_point->Name());  // zero_point
    LOGS_DEFAULT(VERBOSE) << quant_param.zero_point->Name() << "is skipped";
  }
}

// Ignore the input (with quantization scale and ZP if available)
// The input (usually weight) is already embedded in the NNAPI model
static void AddInputToSkip(ModelBuilder& model_builder, const NodeUnitIODef& io_def) {
  model_builder.AddInitializerToSkip(io_def.node_arg.Name());  // main input
  if (io_def.quant_param)
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *io_def.quant_param);
}

template <class T>
void CreateSharedOpBuilderImpl(const std::string& op_type,
                               OpBuilderRegistrations& op_registrations,
                               const std::vector<std::string>& op_types) {
  // The shared OpSupportChecker is already in the OpSupportCheckerRegistrations
  if (op_registrations.op_builder_map.find(op_type) != op_registrations.op_builder_map.cend())
    return;

  op_registrations.builders.push_back(std::make_unique<T>());
  for (const auto& op : op_types) {
    op_registrations.op_builder_map.emplace(op, op_registrations.builders.back().get());
  }
}

#pragma endregion helpers

#pragma region op_base

class BaseOpBuilder : public IOpBuilder {
 public:
  virtual ~BaseOpBuilder() = default;
  virtual void AddInitializersToSkip(ModelBuilder& /* model_builder */, const NodeUnit& /* node_unit */) const override {}
  Status AddToModelBuilder(ModelBuilder& model_builder, const NodeUnit& node_unit) const override final;

 protected:
  virtual Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const = 0;
  static bool IsOpSupported(const ModelBuilder& model_builder, const NodeUnit& node_unit) ORT_MUST_USE_RESULT;
};

/* static */ bool BaseOpBuilder::IsOpSupported(const ModelBuilder& model_builder, const NodeUnit& node_unit) {
  OpSupportCheckParams params{
      model_builder.GetNNAPIFeatureLevel(),
      model_builder.UseNCHW(),
  };

  return IsNodeSupported(node_unit, model_builder.GetGraphViewer(), params);
}

Status BaseOpBuilder::AddToModelBuilder(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  ORT_RETURN_IF_NOT(IsOpSupported(model_builder, node_unit), "Unsupported operator ", node_unit.OpType());
  ORT_RETURN_IF_ERROR(AddToModelBuilderImpl(model_builder, node_unit));
  LOGS_DEFAULT(VERBOSE) << "Operator name: [" << node_unit.Name()
                        << "] type: [" << node_unit.OpType() << "] was added";
  return Status::OK();
}

#pragma endregion op_base

#pragma region op_binary

class BinaryOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  static void CreateSharedOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);

 private:
  static bool IsQuantizedOp(const NodeUnit& node_unit) ORT_MUST_USE_RESULT;  // TODO, see if we want to move this to BaseOpBuilder
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

/* static */ bool BinaryOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) {
  // TODO, add support for QDQ NodeUnit
  return node_unit.OpType() == "QLinearAdd";
}

void BinaryOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (!IsQuantizedOp(node_unit))
    return;

  const auto& inputs = node_unit.Inputs();
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[0].quant_param);               // a_scale, a_zp
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[1].quant_param);               // b_scale, b_zp
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
}

/* static */ void BinaryOpBuilder::CreateSharedOpBuilder(
    const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<BinaryOpBuilder>(
      op_type, op_registrations,
      {
          "Add",
          "Sub",
          "Mul",
          "Div",
          "QLinearAdd",
          "Pow",
      });
}

Status BinaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& op_type(node_unit.OpType());
  const auto& inputs = node_unit.Inputs();

  int32_t op_code;
  bool add_activation = true;
  bool op_is_qlinear = op_type == "QLinearAdd";
  if (op_type == "Add" || op_is_qlinear) {
    op_code = ANEURALNETWORKS_ADD;
  } else if (op_type == "Sub") {
    op_code = ANEURALNETWORKS_SUB;
  } else if (op_type == "Mul") {
    op_code = ANEURALNETWORKS_MUL;
  } else if (op_type == "Div") {
    op_code = ANEURALNETWORKS_DIV;
  } else if (op_type == "Pow") {
    add_activation = false;  // ANEURALNETWORKS_POW does not have activation
    op_code = ANEURALNETWORKS_POW;
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "UnaryOpBuilder, unknown op: ", op_type);
  }

  std::string input1 = inputs[0].node_arg.Name();
  std::string input2 = inputs[1].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  bool output_is_nhwc = false;
  ORT_RETURN_IF_ERROR(
      TransposeBinaryOpInputLayout(model_builder, node_unit, input1, input2, output_is_nhwc));

  float a_scale = 0.0f,
        b_scale = 0.0f,
        y_scale = 0.0f;
  int32_t a_zero_point = 0,
          b_zero_point = 0,
          y_zero_point = 0;

  if (op_is_qlinear) {
    ORT_RETURN_IF_ERROR(GetBinaryOpQuantizationScaleAndZeroPoint(
        model_builder.GetInitializerTensors(), node_unit,
        a_scale, b_scale, y_scale,
        a_zero_point, b_zero_point, y_zero_point));
  }

  // Verify if the scale and zero point matchs from onnx input and nnapi input match
  if (op_is_qlinear) {
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input1, a_scale, a_zero_point));
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input2, b_scale, b_zero_point));
  }

  int32_t fuse_code = ANEURALNETWORKS_FUSED_NONE;
  if (add_activation) {
    fuse_code = model_builder.FindActivation(node_unit);
  }

  return AddBinaryOperator(op_code, model_builder,
                           input1, input2,
                           add_activation, fuse_code,
                           output, output_is_nhwc, y_scale, y_zero_point);
}

#pragma endregion

#pragma region op_relu

class ReluOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

Status ReluOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  bool output_is_nhwc = model_builder.IsOperandNHWC(input);
  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);

  // skip this relu if it is some op's fuse output
  if (Contains(model_builder.GetFusedActivations(), input)) {
    LOGS_DEFAULT(VERBOSE) << "Relu Node [" << node_unit.Name() << "] fused";
    model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type, output_is_nhwc);
  } else {
    std::vector<uint32_t> input_indices;
    input_indices.push_back(operand_indices.at(input));
    ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_RELU, input_indices,
                                                   {output}, {output_operand_type}, {output_is_nhwc}));
  }

  return Status::OK();
}

#pragma endregion op_relu

#pragma region op_transpose

class TransposeOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

Status TransposeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  NodeAttrHelper helper(node_unit);
  std::vector<int32_t> perm = helper.Get("perm", std::vector<int32_t>());
  auto input_dims = shaper[input].size();
  if (perm.empty()) {
    for (int32_t i = input_dims - 1; i >= 0; i--)
      perm.push_back(i);
  } else {
    ORT_RETURN_IF_NOT(perm.size() == input_dims, "Perm and input should have same dimension");
  }

  if (model_builder.IsOperandNHWC(input)) {
    ORT_RETURN_IF_NOT(input_dims == 4, "Only 4D shape can be nhwc");

    // we are using nhwc here, but the axis is in nchw, need to transpose axis from nchw to nhwc
    const int32_t axis_nchw_to_nhwc[4]{0, 3, 1, 2};
    for (size_t i = 0; i < perm.size(); i++)
      perm[i] = axis_nchw_to_nhwc[perm[i]];
  }

  std::string perm_name = model_builder.GetUniqueName(node_unit.Name() + input + "perm");

  // It is possible this onnx transpose operator can be nchw->nhwc, but so far I don't see
  // any scenario will do this since onnx is nchw only, assume the output is always not nhwc
  // even it is, there will be extra transpose in the onnx model to convert it back to nchw
  // before conv/pool/... operators
  ORT_RETURN_IF_ERROR(AddTransposeOperator(model_builder, input, perm_name, perm, output, false /* is_nhwc */));

  return Status::OK();
}

#pragma endregion op_transpose

#pragma region op_reshape

class ReshapeOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  static Status AddReshapeOperator(ModelBuilder& model_builder, const NodeUnit& node_unit,
                                   const std::string& input, const std::vector<int32_t>& shape);

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  static bool CanSkipReshape(const ModelBuilder& model_builder, const NodeUnit& node_unit,
                             size_t input_rank, size_t output_rank);
};

void ReshapeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  model_builder.AddInitializerToSkip(node_unit.Inputs()[1].node_arg.Name());
}

// We can skip the Reshape if all the output edges satisfies both the following conditions
// 1. The output the reshape/flatten is not an output of the graph
// 2. The output of the reshape/flatten is the input 0 of one or more GEMM/Matmul operators,
//    and not any other types of operator,
//    and the input rank >= 2 and output_rank == 2
//    This is because Gemm/Matmul will map to ANEURALNETWORKS_FULLY_CONNECTED in NNAPI,
//    ANEURALNETWORKS_FULLY_CONNECTED will flatten the 2+ dim input 0 to 2d
// The reason we want to skip Reshape is that Reshape is not running on Hardware (NPU,...) in NNAPI for
// some CPU (e.g. Qualcomm SD for now), skipping unnecessary Reshape will prevent context switching
// between NNAPI CPU impl and Hardware Accelerator impl and will speed up the execution
// If we are going to skip the reshape, we will still add correct shape and operand type for the output in
// onnxruntime::nnapi::Model.
/* static */ bool ReshapeOpBuilder::CanSkipReshape(const ModelBuilder& model_builder, const NodeUnit& node_unit,
                                                   size_t input_rank, size_t output_rank) {
  const auto& output_node_arg = node_unit.Outputs()[0].node_arg;
  const auto& output_name = output_node_arg.Name();
  const auto& output_node = *node_unit.GetOutputNodes()[0];

  // We will go through all the output edges
  for (auto it = output_node.OutputEdgesBegin(), end = output_node.OutputEdgesEnd(); it != end; ++it) {
    const auto& dest_node_unit = model_builder.GetNodeUnit(&it->GetNode());
    const auto& op_type = dest_node_unit.OpType();
    // TODO add quantized matmul when reshape support quantized input
    if (op_type != "Gemm" && op_type != "MatMul") {
      LOGS_DEFAULT(VERBOSE) << "Reshape/Flatten can only be skipped when the output is Gemm/Matmul"
                            << " or no op is using the output (output is graph output)"
                            << ", output name, " << output_name
                            << " is used by " << op_type;
      return false;
    }

    // Now the dest node is Gemm/Matmul, we want to make sure it is supported
    if (!BaseOpBuilder::IsOpSupported(model_builder, node_unit)) {
      return false;
    }

    // NNAPI ANEURALNETWORKS_FULLY_CONNECTED will only flatten the input 0
    if (&output_node_arg != &dest_node_unit.Inputs()[0].node_arg) {
      LOGS_DEFAULT(VERBOSE) << "Reshape/Flatten can only be skipped when the output is input 0 of Gemm/Matmul"
                            << ", output name, " << output_name;
      return false;
    }

    // We only support 2d matmul/gemm here
    // And NNAPI ANEURALNETWORKS_FULLY_CONNECTED will only flatten input rank >= 2
    if (input_rank < 2 || output_rank != 2) {
      LOGS_DEFAULT(VERBOSE) << "Reshape/Flatten can only be skipped when input_rank >= 2 and output_rank == 2"
                            << ", output name, " << output_name
                            << ", the actual input_rank, " << input_rank
                            << ", the actual output_rank, " << output_rank;
      return false;
    }
  }

  // If we reach here, we have all the Reshape outputs are used by gemm/matmul, or Reshape has no output edge
  // Check if the Reshape output is a graph output, if so we cannot skip the Reshape
  // We do not care the case where the Reshape output is a dead end
  for (const auto* node_arg : model_builder.GetGraphViewer().GetOutputs()) {
    if (node_arg == &output_node_arg) {
      LOGS_DEFAULT(VERBOSE) << "Reshape/Flatten can not be skipped when the output is a graph output"
                            << ", output name, " << output_name;
      return false;
    }
  }

  LOGS_DEFAULT(VERBOSE) << "Skipping Reshape/Flatten node ["
                        << node_unit.Name() << "] with output, " << output_name;
  return true;
}

/* static */ Status ReshapeOpBuilder::AddReshapeOperator(ModelBuilder& model_builder,
                                                         const NodeUnit& node_unit,
                                                         const std::string& input,
                                                         const std::vector<int32_t>& shape) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  ORT_RETURN_IF_ERROR(shaper.Reshape(input, shape, output));
  auto input_rank = shaper[input].size();
  auto output_rank = shaper[output].size();

  // Since Reshape is not running using hardware in NNAPI for some CPU (e.g. Qualcomm SD for now)
  // We will try to see if we the skip the Reshape to prevent context switching between
  // NNAPI CPU impl and NNAPI hardware accelerator impl
  if (CanSkipReshape(model_builder, node_unit, input_rank, output_rank)) {
    // Since reshape can be skipped, only register the dimension and type, with same index and new name
    const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
    model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type, false);
  } else {
    // We still need to perform a reshape here
    // Add input
    std::vector<uint32_t> input_indices;
    input_indices.push_back(operand_indices.at(input));
    // Add new shape
    Shape shape_dimen = {static_cast<uint32_t>(shape.size())};
    std::string shape_name = model_builder.GetUniqueName(node_unit.Name() + input + "newshape");
    OperandType shape_operand_type(Type::TENSOR_INT32, shape_dimen);
    ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(shape_name, shape.data(), shape_operand_type));
    input_indices.push_back(operand_indices.at(shape_name));

    const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
    ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_RESHAPE, input_indices, {output}, {output_operand_type}, {false}));
  }

  return Status::OK();
}

Status ReshapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& initializers(model_builder.GetInitializerTensors());

  auto input = node_unit.Inputs()[0].node_arg.Name();
  if (model_builder.IsOperandNHWC(input)) {
    // We want to transpose nhwc operand back to nchw before reshape
    ORT_RETURN_IF_ERROR(GetNCHWInput(model_builder, node_unit, 0, input));
  }

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

  return AddReshapeOperator(model_builder, node_unit, input, shape);
}

#pragma endregion op_reshape

#pragma region op_batchnormalization

class BatchNormalizationOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void BatchNormalizationOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  // skip everything except input0 for BatchNormalization
  model_builder.AddInitializerToSkip(node_unit.Inputs()[1].node_arg.Name());  // scale
  model_builder.AddInitializerToSkip(node_unit.Inputs()[2].node_arg.Name());  // B
  model_builder.AddInitializerToSkip(node_unit.Inputs()[3].node_arg.Name());  // mean
  model_builder.AddInitializerToSkip(node_unit.Inputs()[4].node_arg.Name());  //var
}

Status BatchNormalizationOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());
  NodeAttrHelper helper(node_unit);
  const auto& inputs = node_unit.Inputs();

  // For reshape we are not really doing anything but
  // register a new operand with new shape
  const auto& input = inputs[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  const auto& scale_tensor = *initializers.at(inputs[1].node_arg.Name());
  const auto& bias_tensor = *initializers.at(inputs[2].node_arg.Name());
  const auto& mean_tensor = *initializers.at(inputs[3].node_arg.Name());
  const auto& var_tensor = *initializers.at(inputs[4].node_arg.Name());
  const auto eps = helper.Get("epsilon", 1e-5f);

  const auto size = SafeInt<uint32_t>(scale_tensor.dims()[0]);
  std::vector<float> a, b;
  a.reserve(size);
  b.reserve(size);

  std::vector<uint8_t> unpacked_scale_tensor;
  ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(scale_tensor, unpacked_scale_tensor));
  const float* scale_data = reinterpret_cast<const float*>(unpacked_scale_tensor.data());

  std::vector<uint8_t> unpacked_bias_tensor;
  ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(bias_tensor, unpacked_bias_tensor));
  const float* bias_data = reinterpret_cast<const float*>(unpacked_bias_tensor.data());

  std::vector<uint8_t> unpacked_mean_tensor;
  ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(mean_tensor, unpacked_mean_tensor));
  const float* mean_data = reinterpret_cast<const float*>(unpacked_mean_tensor.data());

  std::vector<uint8_t> unpacked_var_tensor;
  ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(var_tensor, unpacked_var_tensor));
  const float* var_data = reinterpret_cast<const float*>(unpacked_var_tensor.data());

  for (int64_t i = 0; i < size; i++) {
    a.push_back(scale_data[i] / sqrt(var_data[i] + eps));
    b.push_back((scale_data[i] * -mean_data[i]) / sqrt(var_data[i] + eps) +
                bias_data[i]);
  }

  const auto tensor_a_name = model_builder.GetUniqueName(node_unit.Name() + input + "_imm_a");
  const auto tensor_b_name = model_builder.GetUniqueName(node_unit.Name() + input + "_imm_b");
  const auto tensor_imm_product_name = model_builder.GetUniqueName(node_unit.Name() + input + "_imm_mul");
  Shape tensor_a_dimen = {size};

  bool input_is_nhwc = model_builder.IsOperandNHWC(input);
  bool output_is_nhwc = input_is_nhwc;

  if (!input_is_nhwc) {
    // the batch normalization is applied on C channel,
    // if the input is NC[HW], will need correct shape for tensor_a/b
    // to make sure we are broadcasting on the correct channel,
    // input shape {N, C}       ==> tensor_a/b's shape {size}
    // input shape {N, C, H}    ==> tensor_a/b's shape {size, 1}
    // input shape {N, C, H, W} ==> tensor_a/b's shape {size, 1, 1}
    const auto input_rank = shaper[input].size();
    for (size_t i = 2; i < input_rank; i++)
      tensor_a_dimen.push_back(1);
  }

  shaper.AddShape(tensor_a_name, tensor_a_dimen);
  shaper.AddShape(tensor_b_name, tensor_a_dimen);
  const OperandType a_operand_type(operand_types.at(input).type, tensor_a_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(tensor_a_name, a.data(), a_operand_type));
  const OperandType b_operand_type(operand_types.at(input).type, tensor_a_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(tensor_b_name, b.data(), b_operand_type));

  // Mul
  ORT_RETURN_IF_ERROR(AddBinaryOperator(ANEURALNETWORKS_MUL,
                                        model_builder,
                                        input, tensor_a_name,
                                        true /* add_activation */, ANEURALNETWORKS_FUSED_NONE,
                                        tensor_imm_product_name,
                                        output_is_nhwc));

  // Add
  int32_t fuse_code = model_builder.FindActivation(node_unit);
  ORT_RETURN_IF_ERROR(AddBinaryOperator(ANEURALNETWORKS_ADD,
                                        model_builder,
                                        tensor_imm_product_name, tensor_b_name,
                                        true /* add_activation */, fuse_code,
                                        output,
                                        output_is_nhwc));

  return Status::OK();
}

#pragma endregion op_batchnormalization

#pragma region op_pool

class PoolOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  static void CreateSharedOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);

 private:
  static bool IsQuantizedOp(const NodeUnit& node_unit) ORT_MUST_USE_RESULT;  // TODO, see if we want to move this to BaseOpBuilder
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

/* static */ bool PoolOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) {
  // TODO, add support for QDQ NodeUnit
  return node_unit.OpType() == "QLinearAveragePool";
}

void PoolOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (!IsQuantizedOp(node_unit))
    return;

  // skip input/output scales and zeropoints
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Inputs()[0].quant_param);   // x_scale, x_zp
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
}

/* static */ void PoolOpBuilder::CreateSharedOpBuilder(
    const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<PoolOpBuilder>(
      op_type, op_registrations,
      {
          "GlobalAveragePool",
          "GlobalMaxPool",
          "AveragePool",
          "MaxPool",
          "QLinearAveragePool",
      });
}

Status PoolOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  NodeAttrHelper helper(node_unit);

  auto input = node_unit.Inputs()[0].node_arg.Name();
  bool use_nchw = model_builder.UseNCHW();
  bool input_is_nhwc = model_builder.IsOperandNHWC(input);
  bool output_is_nhwc = false;
  if (use_nchw) {
    ORT_RETURN_IF_NOT(!input_is_nhwc, "model_builder.UseNCHW() but input is NHWC");
  } else {
    output_is_nhwc = true;
    if (!input_is_nhwc) {
      ORT_RETURN_IF_ERROR(GetNHWCInput(model_builder, node_unit, 0, input));
    }
  }

  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  const auto& op_type = node_unit.OpType();

  int32_t op_code;
  bool is_qlinear_average_pool = op_type == "QLinearAveragePool";
  bool is_average_pool = op_type == "AveragePool" || is_qlinear_average_pool;
  if (is_average_pool || op_type == "GlobalAveragePool")
    op_code = ANEURALNETWORKS_AVERAGE_POOL_2D;
  else  // (op_type == "MaxPool" || op_type == "GlobalMaxPool")
    op_code = ANEURALNETWORKS_MAX_POOL_2D;

  std::vector<int32_t> onnx_pads, onnx_strides, kernel_shape;
  bool use_auto_pad = false;
  int32_t nnapi_padding_code = ANEURALNETWORKS_PADDING_VALID;
  const auto& input_shape = shaper[input];
  if (is_average_pool || op_type == "MaxPool") {
    const auto auto_pad_type = StringToAutoPadType(helper.Get("auto_pad", "NOTSET"));
    kernel_shape = helper.Get("kernel_shape", std::vector<int32_t>{0, 0});
    onnx_strides = helper.Get("strides", std::vector<int>{1, 1});
    onnx_pads = helper.Get("pads", std::vector<int>{0, 0, 0, 0});
    const auto weight_size_y = static_cast<uint32_t>(kernel_shape[0]);
    const auto weight_size_x = static_cast<uint32_t>(kernel_shape[1]);
    ORT_RETURN_IF_ERROR(
        HandleAutoPad(input_shape, weight_size_y, weight_size_x,
                      onnx_strides, {1, 1} /* onnx_dilations */,
                      auto_pad_type, use_nchw,
                      onnx_pads, nnapi_padding_code, use_auto_pad));
  } else {  // (op_type == "GlobalAveragePool" || op_type == "GlobalMaxPool")
    use_auto_pad = true;
    nnapi_padding_code = ANEURALNETWORKS_PADDING_VALID;
    onnx_strides = std::vector<int32_t>{1, 1};
    onnx_pads = std::vector<int32_t>{0, 0, 0, 0};
    if (use_nchw) {
      kernel_shape = std::vector<int32_t>{static_cast<int32_t>(input_shape[2]),
                                          static_cast<int32_t>(input_shape[3])};
    } else {
      kernel_shape = std::vector<int32_t>{static_cast<int32_t>(input_shape[1]),
                                          static_cast<int32_t>(input_shape[2])};
    }
  }

  int32_t fuse_code = model_builder.FindActivation(node_unit);

  // Get output scale and zero point if this is QLinearAveragePool
  // Otherwise we will use the scale and zero point of the input
  const OperandType& input_operand_type = operand_types.at(input);
  float y_scale = input_operand_type.operandType.scale;
  int32_t y_zero_point = input_operand_type.operandType.zeroPoint;
  if (is_qlinear_average_pool) {
    const auto& initializers = model_builder.GetInitializerTensors();
    float x_scale = 0.0f;
    int32_t x_zero_point = 0;
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        initializers, node_unit.Inputs()[0], node_unit.ModelPath(), x_scale, x_zero_point));

    // Verify if the scale and zero point values from onnx input and nnapi input match
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        initializers, node_unit.Outputs()[0], node_unit.ModelPath(), y_scale, y_zero_point));
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));

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
  ADD_SCALAR_OPERAND(model_builder, input_indices, kernel_shape[1]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, kernel_shape[0]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);

  if (model_builder.GetNNAPIFeatureLevel() > ANEURALNETWORKS_FEATURE_LEVEL_2) {  // nchw only supported on api 29+
    ADD_SCALAR_OPERAND(model_builder, input_indices, use_nchw);
  }

  ORT_RETURN_IF_ERROR(shaper.Pool(input,
                                  onnx_pads, onnx_strides, kernel_shape,
                                  use_nchw,
                                  output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output], y_scale, y_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion op_pool

#pragma region op_conv

class ConvOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  static void CreateSharedOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);

 private:
  static bool IsQuantizedOp(const NodeUnit& node_unit) ORT_MUST_USE_RESULT;  // TODO, see if we want to move this to BaseOpBuilder
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

/* static */ bool ConvOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) {
  // TODO, add support for QDQ NodeUnit
  return node_unit.OpType() == "QLinearConv";
}

/* static */ void
ConvOpBuilder::CreateSharedOpBuilder(
    const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<ConvOpBuilder>(
      op_type, op_registrations,
      {
          "Conv",
          "QLinearConv",
      });
}

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
  bool is_qlinear_conv = IsQuantizedOp(node_unit);

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
  bool input_is_nhwc = model_builder.IsOperandNHWC(input);
  bool output_is_nhwc = false;
  if (use_nchw) {
    ORT_RETURN_IF_NOT(!input_is_nhwc, "model_builder.UseNCHW() but input is NHWC");
  } else {
    output_is_nhwc = true;
    if (!input_is_nhwc) {
      ORT_RETURN_IF_ERROR(GetNHWCInput(model_builder, node_unit, 0, input));
    }
  }

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
  if (is_qlinear_conv) {
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
                             "The initializer of graph ", weight, " doesn't have valid type: ", weight_tensor.data_type());
  }

  // Get weight operand type
  // Per-channel quantized weight is handled differently
  OperandType onnx_weight_operand_type =
      (is_qlinear_conv && w_scales.has_value())
          ? OperandType{onnx_weight_type, onnx_weight_shape,
                        SymmPerChannelQuantParams{w_scales.value(),
                                                  depthwise_conv_2d ? 3u : 0u}}  // channelDim is 3 for depthwise-conv
          : OperandType{onnx_weight_type, onnx_weight_shape, w_scale, w_zero_point};

  // Pre-process weights
  if (conv_2d || grouped_conv_2d) {
    ORT_RETURN_IF_ERROR(AddInitializerInNewLayout(model_builder, weight, onnx_weight_operand_type, L_0231, is_per_tensor_u8s8));
  } else {  // depthwise_conv_2d
    ORT_RETURN_IF_ERROR(AddInitializerInNewLayout(model_builder, weight, onnx_weight_operand_type, L_1230, is_per_tensor_u8s8));
  }

  if (is_qlinear_conv) {
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
  } else if (is_qlinear_conv) {
    // QLinearConv's bias type need special handling to add scale for quantization input
    const auto& bias_tensor = *model_builder.GetInitializerTensors().at(bias);
    ORT_RETURN_IF_NOT(bias_tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32,
                      "bias of QLinearConv should be int32, actual type: ", bias_tensor.data_type());
    Shape bias_dimen;
    for (auto dim : bias_tensor.dims())
      bias_dimen.push_back(SafeInt<uint32_t>(dim));

    std::vector<uint8_t> unpacked_tensor;
    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(bias_tensor, unpacked_tensor));
    OperandType bias_operand_type(Type::TENSOR_INT32, bias_dimen, x_scale * w_scale);
    ORT_RETURN_IF_ERROR(
        model_builder.AddOperandFromPersistMemoryBuffer(bias, unpacked_tensor.data(), bias_operand_type));
  }

  const auto auto_pad_type = StringToAutoPadType(helper.Get("auto_pad", "NOTSET"));
  bool use_auto_pad = false;
  int32_t nnapi_padding_code = ANEURALNETWORKS_PADDING_SAME;
  const auto& input_shape = shaper[input];
  const auto& kernel_shape = shaper[weight];
  const auto weight_size_y = kernel_shape[1];
  const auto weight_size_x = kernel_shape[2];
  ORT_RETURN_IF_ERROR(
      HandleAutoPad(input_shape, weight_size_y, weight_size_x,
                    onnx_strides, onnx_dilations,
                    auto_pad_type, use_nchw,
                    onnx_pads, nnapi_padding_code, use_auto_pad));

  std::vector<uint32_t> input_indices;
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

  if (model_builder.GetNNAPIFeatureLevel() > ANEURALNETWORKS_FEATURE_LEVEL_2) {
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
    ORT_RETURN_IF_ERROR(shaper.Conv(input, weight,
                                    onnx_pads, onnx_strides, onnx_dilations,
                                    use_nchw,
                                    output));
  } else {  // depthwise_conv_2d
    operationCode = ANEURALNETWORKS_DEPTHWISE_CONV_2D;
    ORT_RETURN_IF_ERROR(shaper.DepthwiseConv(input, weight,
                                             onnx_pads, onnx_strides, onnx_dilations,
                                             use_nchw,
                                             output));
  }

  const OperandType output_operand_type(operand_types.at(input).type, shaper[output], y_scale, y_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(operationCode, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion op_conv

#pragma region op_cast

class CastOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

Status CastOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  NodeAttrHelper helper(node_unit);

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  bool output_is_nhwc = model_builder.IsOperandNHWC(input);

  auto to = helper.Get("to", 0);
  Type type;
  switch (to) {
    case ONNX_NAMESPACE::TensorProto::FLOAT:
      type = Type::TENSOR_FLOAT32;
      break;
    case ONNX_NAMESPACE::TensorProto::INT32:
      type = Type::TENSOR_INT32;
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid cast to type: ", to);
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_CAST, input_indices, {output},
                                                 {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion

#pragma region op_softmax

class SoftMaxOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

Status SoftMaxOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto android_feature_level = model_builder.GetNNAPIFeatureLevel();
  NodeAttrHelper helper(node_unit);

  auto input = node_unit.Inputs()[0].node_arg.Name();
  bool input_is_nhwc = model_builder.IsOperandNHWC(input);
  bool output_is_nhwc = input_is_nhwc;
  if (android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
    if (model_builder.IsOperandNHWC(input)) {
      output_is_nhwc = false;
      // We want to transpose nhwc operand back to nchw before softmax
      ORT_RETURN_IF_ERROR(GetNCHWInput(model_builder, node_unit, 0, input));
    }
  }

  int32_t axis = helper.Get("axis", 1);
  if (output_is_nhwc) {
    const int32_t axis_nchw_to_nhwc[4]{0, 3, 1, 2};
    axis = axis_nchw_to_nhwc[axis];
  }

  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  float beta = 1.f;
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, beta);

  if (android_feature_level > ANEURALNETWORKS_FEATURE_LEVEL_2) {
    // you can only specify axis for android api level 29+
    ADD_SCALAR_OPERAND(model_builder, input_indices, axis);
  }

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_SOFTMAX, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion

#pragma region op_identity

class IdentityOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

Status IdentityOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  // Identity is not really going to do anything
  // Just register the dimension and type, with same index and new name
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  bool output_is_nhwc = model_builder.IsOperandNHWC(input);

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));  // input

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type, output_is_nhwc);
  return Status::OK();
}

#pragma endregion

#pragma region op_gemm

class GemmOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  static void CreateSharedOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);

 private:
  static bool IsQuantizedOp(const NodeUnit& node_unit) ORT_MUST_USE_RESULT;  // TODO, see if we want to move this to BaseOpBuilder
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

/* static */ bool GemmOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) {
  // TODO, add support for QDQ NodeUnit
  return node_unit.OpType() == "QLinearMatMul";
}

/* static */ void GemmOpBuilder::CreateSharedOpBuilder(
    const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<GemmOpBuilder>(
      op_type, op_registrations,
      {
          "Gemm",
          "MatMul",
          "QLinearMatMul",
      });
}

void GemmOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
  if (IsQuantizedOp(node_unit)) {
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[0].quant_param);               // b_scale, b_zp
    AddInputToSkip(model_builder, inputs[1]);                                                    // b, b_scale, b_zp
    AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
  } else {
    const auto& op = node_unit.OpType();
    if (op == "MatMul") {
      model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());
    } else if (op == "Gemm") {
      NodeAttrHelper helper(node_unit);
      const auto transB = helper.Get("transB", 0);
      if (transB == 0)
        model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());
    }
  }
}

Status GemmOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());

  const auto& op = node_unit.OpType();
  const auto& inputs = node_unit.Inputs();
  NodeAttrHelper helper(node_unit);
  bool is_qlinear_matmul = op == "QLinearMatMul";

  const auto& input1 = inputs[0].node_arg.Name();
  const auto& input2 = inputs[1].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  const auto transB = helper.Get("transB", 0);

  float a_scale = 0.0f,
        b_scale = 0.0f,
        y_scale = 0.0f;
  int32_t a_zero_point = 0,
          b_zero_point = 0,
          y_zero_point = 0;

  bool is_per_tensor_u8s8 = false;
  if (is_qlinear_matmul) {
    optional<std::vector<float>> w_scales;
    ORT_RETURN_IF_ERROR(
        GetConvMatMulOpQuantizationScaleAndZeroPoint(model_builder, node_unit,
                                                     a_scale, b_scale, y_scale,
                                                     a_zero_point, b_zero_point, y_zero_point,
                                                     w_scales, is_per_tensor_u8s8));
  }

  uint32_t input_2_idx;
  if (transB == 0) {
    Type onnx_mat_b_type;
    if (!is_qlinear_matmul)
      onnx_mat_b_type = Type::TENSOR_FLOAT32;
    else
      onnx_mat_b_type = Type::TENSOR_QUANT8_ASYMM;

    const auto& mat_b_tensor = *initializers.at(input2);
    Shape onnx_mat_b_shape;
    for (auto dim : mat_b_tensor.dims())
      onnx_mat_b_shape.push_back(SafeInt<uint32_t>(dim));

    const OperandType onnx_mat_b_operand_type(onnx_mat_b_type, onnx_mat_b_shape, b_scale, b_zero_point);
    ORT_RETURN_IF_ERROR(AddInitializerTransposed(model_builder, onnx_mat_b_operand_type, input2, is_per_tensor_u8s8));
  }

  input_2_idx = operand_indices.at(input2);

  // Verify if the scale and zero point matchs from onnx input and nnapi input
  if (is_qlinear_matmul) {
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input1, a_scale, a_zero_point));
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input2, b_scale, b_zero_point));
  }

  uint32_t bias_idx;
  bool has_bias = inputs.size() > 2;
  if (has_bias) {
    const auto& bias = inputs[2].node_arg.Name();
    // We need squeeze the input tensor to 1d if necessary
    if (shaper[bias].size() > 1) {
      std::string bias_squeezed = model_builder.GetUniqueName(node_unit.Name() + op + "_bias_squeezed");
      // We will use squeeze all here
      ORT_RETURN_IF_ERROR(AddSqueezeOp(model_builder, node_unit.Name(),
                                       bias, bias_squeezed,
                                       {} /* axes */));
      bias_idx = operand_indices.at(bias_squeezed);
      LOGS_DEFAULT(VERBOSE) << "GemmOpBuilder - Operand [" << bias << "] squeezed from "
                            << Shape2String(shaper[bias])
                            << " to "
                            << Shape2String(shaper[bias_squeezed]);
    } else {
      bias_idx = operand_indices.at(bias);
    }
  } else {
    // No C supplied, we need a vector of 0
    std::string bias = model_builder.GetUniqueName(node_unit.Name() + op + "_bias");
    const auto& bias_type = operand_types.at(input2).type;
    const Shape& bias_dimen = {shaper[input2][0]};
    if (bias_type == Type::TENSOR_FLOAT32) {
      std::vector<float> buffer(bias_dimen[0], 0.f);
      OperandType bias_operand_type(Type::TENSOR_FLOAT32, bias_dimen);
      ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
    } else if (bias_type == Type::TENSOR_QUANT8_ASYMM) {
      std::vector<int32_t> buffer(bias_dimen[0], 0);
      OperandType bias_operand_type(Type::TENSOR_INT32, bias_dimen, a_scale * b_scale, 0);
      ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown weight type ", TypeToStr(bias_type));
    }

    bias_idx = operand_indices.at(bias);
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));  // A
  input_indices.push_back(input_2_idx);                 // B
  input_indices.push_back(bias_idx);                    // C
  int32_t fuse_code = model_builder.FindActivation(node_unit);
  ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);

  ORT_RETURN_IF_ERROR(shaper.FC(input1, input2, output));
  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output], y_scale, y_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_FULLY_CONNECTED, input_indices,
                                                 {output}, {output_operand_type}, {false}));
  return Status::OK();
}

#pragma endregion

#pragma region op_unary

class UnaryOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  static void CreateSharedOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);

 private:
  static bool IsQuantizedOp(const NodeUnit& node_unit) ORT_MUST_USE_RESULT;  // TODO, see if we want to move this to BaseOpBuilder
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

/* static */ bool UnaryOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) {
  // TODO, add support for QDQ NodeUnit
  return node_unit.OpType() == "QLinearSigmoid";
}

void UnaryOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (!IsQuantizedOp(node_unit))
    return;

  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Inputs()[0].quant_param);   // x_scale, x_zp
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
}

/* static */ void UnaryOpBuilder::CreateSharedOpBuilder(
    const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<UnaryOpBuilder>(
      op_type, op_registrations,
      {
          "Abs",
          "Exp",
          "Floor",
          "Log",
          "Sigmoid",
          "Neg",
          "Sin",
          "Sqrt",
          "Tanh",
          "QLinearSigmoid",
      });
}

Status UnaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& op_type(node_unit.OpType());

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  bool output_is_nhwc = model_builder.IsOperandNHWC(input);

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  bool is_qlinear_sigmoid = op_type == "QLinearSigmoid";

  int32_t op_code;
  if (op_type == "Abs")
    op_code = ANEURALNETWORKS_ABS;
  else if (op_type == "Exp")
    op_code = ANEURALNETWORKS_EXP;
  else if (op_type == "Floor")
    op_code = ANEURALNETWORKS_FLOOR;
  else if (op_type == "Log")
    op_code = ANEURALNETWORKS_LOG;
  else if (op_type == "Sigmoid" || is_qlinear_sigmoid)
    op_code = ANEURALNETWORKS_LOGISTIC;
  else if (op_type == "Neg")
    op_code = ANEURALNETWORKS_NEG;
  else if (op_type == "Sin")
    op_code = ANEURALNETWORKS_SIN;
  else if (op_type == "Sqrt")
    op_code = ANEURALNETWORKS_SQRT;
  else if (op_type == "Tanh")
    op_code = ANEURALNETWORKS_TANH;
  else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "UnaryOpBuilder, unknown op: ", op_type);
  }

  float y_scale = 0.0f;
  int32_t y_zero_point = 0;
  if (is_qlinear_sigmoid) {
    const auto& initializers = model_builder.GetInitializerTensors();
    float x_scale = 0.0f;
    int32_t x_zero_point = 0;
    ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
        initializers, node_unit.Inputs()[0], node_unit.ModelPath(), x_scale, x_zero_point));

    // Verify if the scale and zero point values from onnx input and nnapi input match
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));

    // We already verified this in  UnaryOpSupportChecker::IsOpSupportedImpl
    y_scale = 1.f / 256;
    y_zero_point = 0;
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output], y_scale, y_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion

#pragma region op_concat

class ConcatOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

Status ConcatOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  NodeAttrHelper helper(node_unit);
  const auto& inputs = node_unit.Inputs();

  std::vector<uint32_t> input_indices;
  const auto& input0 = inputs[0].node_arg.Name();
  bool all_input_have_same_layout = true;
  bool output_is_nhwc = false;
  const auto node_input_size = inputs.size();

  // First if the inputs are uint8, we need verify all the inputs have same scale and zero points
  if (operand_types.at(input0).type == android::nn::wrapper::Type::TENSOR_QUANT8_ASYMM) {
    auto scale = operand_types.at(input0).operandType.scale;
    auto zero_point = operand_types.at(input0).operandType.zeroPoint;

    // Compare scale and zp of input0 to input1~n
    for (size_t i = 1; i < node_input_size; i++) {
      const auto& type = operand_types.at(inputs[i].node_arg.Name());
      ORT_RETURN_IF_NOT(scale == type.operandType.scale,
                        "Input[", i, "]'s scale: ", type.operandType.scale,
                        " is different than input[0]'s scale: ", scale);

      ORT_RETURN_IF_NOT(zero_point == type.operandType.zeroPoint,
                        "Input[", i, "]'s zero_point: ", type.operandType.zeroPoint,
                        " is different than input[0]'s zero_point: ", zero_point);
    }
  }

  // First we want to see if all the input are same layout
  for (size_t i = 0; i < node_input_size - 1; i++) {
    all_input_have_same_layout =
        all_input_have_same_layout &&
        model_builder.IsOperandNHWC(inputs[i].node_arg.Name()) ==
            model_builder.IsOperandNHWC(inputs[i + 1].node_arg.Name());
  }

  std::vector<std::string> input_names;
  input_names.reserve(node_input_size);
  if (all_input_have_same_layout) {
    // if all the inputs are of same layout, output will be the same layout
    output_is_nhwc = model_builder.IsOperandNHWC(input0);

    for (size_t i = 0; i < node_input_size; i++) {
      const auto& input = inputs[i].node_arg.Name();
      input_indices.push_back(operand_indices.at(input));
      input_names.push_back(input);
    }
  } else {
    // if all the inputs are not same layout,
    // will need transpos those nhwc tensors back to nchw
    for (size_t i = 0; i < node_input_size; i++) {
      auto input = inputs[i].node_arg.Name();
      if (model_builder.IsOperandNHWC(input)) {
        ORT_RETURN_IF_ERROR(GetNCHWInput(model_builder, node_unit, i, input));
      }
      input_indices.push_back(operand_indices.at(input));
      input_names.push_back(input);
    }
  }

  int rank = shaper[input0].size();
  int32_t axis = static_cast<int32_t>(HandleNegativeAxis(helper.Get("axis", 1), rank));

  if (output_is_nhwc) {
    ORT_RETURN_IF_NOT(rank == 4,
                      "nhwc is only on 4d shape, input ", input0, " has rank: ", rank);
    // we are using nhwc here, but the axis is in nchw, need to transpose axis from nchw to nhwc
    const uint32_t axis_nchw_to_nhwc[4]{0, 3, 1, 2};
    axis = axis_nchw_to_nhwc[axis];
  }
  ADD_SCALAR_OPERAND(model_builder, input_indices, axis);

  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  ORT_RETURN_IF_ERROR(shaper.Concat(input_names, axis, output));
  OperandType output_operand_type = operand_types.at(input0);
  output_operand_type.SetDimensions(shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_CONCATENATION, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion

#pragma region op_squeeze

class SqueezeOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  static Status GetAxes(ModelBuilder& model_builder, const NodeUnit& node_unit, std::vector<int32_t>& axes);
};

void SqueezeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (node_unit.SinceVersion() > 12 && node_unit.Inputs().size() > 1) {
    model_builder.AddInitializerToSkip(node_unit.Inputs()[1].node_arg.Name());
  }
}

/* static */ Status SqueezeOpBuilder::GetAxes(ModelBuilder& model_builder,
                                              const NodeUnit& node_unit, std::vector<int32_t>& axes) {
  // Squeeze opset 13 use input as axes
  if (node_unit.SinceVersion() > 12) {
    // If axes is not supplied, return an empty axes as default to squeeze all
    if (node_unit.Inputs().size() > 1) {
      const auto& initializers(model_builder.GetInitializerTensors());
      const auto& axes_tensor = *initializers.at(node_unit.Inputs()[1].node_arg.Name());
      std::vector<uint8_t> unpacked_tensor;
      ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(axes_tensor, unpacked_tensor));
      const int64_t* raw_axes = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
      const auto size = SafeInt<uint32_t>(axes_tensor.dims()[0]);
      axes.resize(size);
      for (uint32_t i = 0; i < size; i++) {
        // it is unlikely we have a axis value overflow for int32
        axes[i] = static_cast<int32_t>(raw_axes[i]);
      }
    }
  } else {
    NodeAttrHelper helper(node_unit);
    axes = helper.Get("axes", std::vector<int32_t>());
  }

  return Status::OK();
}

Status SqueezeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto input = node_unit.Inputs()[0].node_arg.Name();
  if (model_builder.IsOperandNHWC(input)) {
    // We want to transpose nhwc operand back to nchw before squeeze
    ORT_RETURN_IF_ERROR(GetNCHWInput(model_builder, node_unit, 0, input));
  }

  std::vector<int32_t> axes;
  ORT_RETURN_IF_ERROR(GetAxes(model_builder, node_unit, axes));
  return AddSqueezeOp(model_builder, node_unit.Name(), input, node_unit.Outputs()[0].node_arg.Name(), axes);
}

#pragma endregion

#pragma region op_quantizelinear

class QuantizeLinearOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void QuantizeLinearOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
}

Status QuantizeLinearOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());

  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  bool output_is_nhwc = model_builder.IsOperandNHWC(input);

  float scale = 0.0f;
  int32_t zero_point = 0;
  ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
      model_builder.GetInitializerTensors(), node_unit.Outputs()[0], node_unit.ModelPath(), scale, zero_point));

  Type output_type = Type::TENSOR_QUANT8_ASYMM;
  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(output_type, shaper[output], scale, zero_point);
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_QUANTIZE, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion

#pragma region op_dequantizelinear

class DequantizeLinearOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void DequantizeLinearOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Inputs()[0].quant_param);  // x_scale, x_zp
}

Status DequantizeLinearOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& inputs = node_unit.Inputs();

  const auto& input = inputs[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  bool output_is_nhwc = model_builder.IsOperandNHWC(input);

  float scale = 0.0;
  int32_t zero_point = 0;
  ORT_RETURN_IF_ERROR(GetQuantizationScaleAndZeroPoint(
      model_builder.GetInitializerTensors(), node_unit.Inputs()[0], node_unit.ModelPath(), scale, zero_point));

  ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, scale, zero_point));

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(Type::TENSOR_FLOAT32, shaper[output]);

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_DEQUANTIZE, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion

#pragma region op_LRN

class LRNOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

Status LRNOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  NodeAttrHelper helper(node_unit);
  const auto android_feature_level = model_builder.GetNNAPIFeatureLevel();

  auto input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  bool output_is_nhwc = model_builder.IsOperandNHWC(input);
  if (android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
    // on android api level 28, we need to transpose the nchw input to nhwc
    output_is_nhwc = true;
    if (!model_builder.IsOperandNHWC(input)) {
      ORT_RETURN_IF_ERROR(GetNHWCInput(model_builder, node_unit, 0, input));
    }
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
    int32_t axis = output_is_nhwc
                       ? 3   // nhwc
                       : 1;  // nchw
    ADD_SCALAR_OPERAND(model_builder, input_indices, axis);
  }

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion

#pragma region op_clip

class ClipOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

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
  bool output_is_nhwc = model_builder.IsOperandNHWC(input);

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);

  if (Contains(model_builder.GetFusedActivations(), input)) {
    LOGS_DEFAULT(VERBOSE) << "Clip Node [" << node_unit.Name() << "] fused";
    model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type, output_is_nhwc);
    return Status::OK();
  }

  float min, max;
  GetClipMinMax(model_builder.GetInitializerTensors(), node_unit.GetNode(), min, max,
                logging::LoggingManager::DefaultLogger());

  int32_t op_code;
  if (min == 0.0f && max == 6.0f)
    op_code = ANEURALNETWORKS_RELU6;
  else if (min == -1.0f && max == 1.0f)
    op_code = ANEURALNETWORKS_RELU1;
  else
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ClipOpBuilder, unsupported input [", min, ", ", max, "].",
                           "We should not reach here, ClipOpBuilder::IsOpSupportedImpl should have caught this.");

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion

#pragma region op_Resize

class ResizeOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void ResizeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
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
  bool input_is_nhwc = model_builder.IsOperandNHWC(input);
  bool output_is_nhwc = false;
  if (use_nchw) {
    ORT_RETURN_IF_NOT(!input_is_nhwc, "model_builder.UseNCHW() but input is NHWC");
  } else {
    output_is_nhwc = true;
    if (!input_is_nhwc) {
      ORT_RETURN_IF_ERROR(GetNHWCInput(model_builder, node_unit, 0, input));
    }
  }

  bool is_linear_resize = helper.Get("mode", "nearest") == "linear";

  int32_t operationCode = is_linear_resize ? ANEURALNETWORKS_RESIZE_BILINEAR
                                           : ANEURALNETWORKS_RESIZE_NEAREST_NEIGHBOR;

  const auto coord_trans_mode = helper.Get("coordinate_transformation_mode", "half_pixel");
  bool using_half_pixel = coord_trans_mode == "half_pixel";
  bool using_align_corners = coord_trans_mode == "align_corners";

  if (inputs.size() == 3) {  // we are using scales
    const auto& scales_name = inputs[2].node_arg.Name();
    const auto& scales_tensor = *initializers.at(scales_name);
    std::vector<uint8_t> unpacked_tensor;
    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(scales_tensor, unpacked_tensor));
    const float* scales_data = reinterpret_cast<const float*>(unpacked_tensor.data());
    float scale_h = scales_data[2];
    float scale_w = scales_data[3];
    ORT_RETURN_IF_ERROR(
        shaper.ResizeUsingScales(input, scale_h, scale_w, use_nchw, output));
  } else {  // we are using sizes
    const auto& sizes_name = inputs[3].node_arg.Name();
    const auto& sizes_tensor = *initializers.at(sizes_name);
    std::vector<uint8_t> unpacked_tensor;
    ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(sizes_tensor, unpacked_tensor));
    const int64_t* sizes_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
    ORT_RETURN_IF_ERROR(
        shaper.ResizeUsingOutputSizes(input, SafeInt<uint32_t>(sizes_data[2]), SafeInt<uint32_t>(sizes_data[3]), use_nchw, output));
  }

  const auto& output_shape = shaper[output];
  int32_t output_h = use_nchw ? output_shape[2] : output_shape[1];
  int32_t output_w = use_nchw ? output_shape[3] : output_shape[2];

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
                                                 {output}, {output_operand_type}, {output_is_nhwc}));

  return Status::OK();
}

#pragma endregion

#pragma region op_flatten

class FlattenOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

Status FlattenOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto input = node_unit.Inputs()[0].node_arg.Name();
  if (model_builder.IsOperandNHWC(input)) {
    // We want to transpose nhwc operand back to nchw before reshape
    ORT_RETURN_IF_ERROR(GetNCHWInput(model_builder, node_unit, 0, input));
  }

  // Flatten is basically a reshape to 2d tensor
  // Get the shape for Reshape here
  Shape input_shape;
  GetShape(node_unit.Inputs()[0].node_arg, input_shape);
  int32_t dim_1 = 1;
  int32_t dim_2 = 1;
  GetFlattenOutputShape(node_unit, input_shape, dim_1, dim_2);
  // If the input is of dynamic shape, replace 0 (dynamic) dimension with -1
  // We cannot have dim_1 and dim_2 both be 0 here, it was checked in IsOpSupportedImpl
  dim_1 = dim_1 == 0 ? -1 : dim_1;
  dim_2 = dim_2 == 0 ? -1 : dim_2;
  std::vector<int32_t> shape{dim_1, dim_2};
  return ReshapeOpBuilder::AddReshapeOperator(model_builder, node_unit, input, shape);
}

#pragma endregion

#pragma region op_minmax

class MinMaxOpBuilder : public BaseOpBuilder {
 public:
  static void CreateSharedOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations);

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
  static Status AddMinMaxOperator(ModelBuilder& model_builder, const NodeUnit& node_unit,
                                  const std::string& input1, const std::string& input2,
                                  bool output_is_nhwc);
};

/* static */ void MinMaxOpBuilder::CreateSharedOpBuilder(
    const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<MinMaxOpBuilder>(
      op_type, op_registrations,
      {
          "Min",
          "Max",
      });
}

/* static */ Status MinMaxOpBuilder::AddMinMaxOperator(ModelBuilder& model_builder, const NodeUnit& node_unit,
                                                       const std::string& input1, const std::string& input2,
                                                       bool output_is_nhwc) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& output = node_unit.Outputs()[0].node_arg.Name();

  const auto& op_type(node_unit.OpType());
  int32_t op_code;
  if (op_type == "Min")
    op_code = ANEURALNETWORKS_MINIMUM;
  else if (op_type == "Max")
    op_code = ANEURALNETWORKS_MAXIMUM;
  else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "MinMaxOpBuilder, unknown op: ", op_type);
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));  // input 1
  input_indices.push_back(operand_indices.at(input2));  // input 2
  ORT_RETURN_IF_ERROR(shaper.Eltwise(input1, input2, output));
  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));

  return Status::OK();
}

Status MinMaxOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  const auto& inputs = node_unit.Inputs();
  std::string input1 = inputs[0].node_arg.Name();
  std::string input2 = inputs[1].node_arg.Name();
  bool output_is_nhwc = false;
  ORT_RETURN_IF_ERROR(TransposeBinaryOpInputLayout(model_builder, node_unit,
                                                   input1, input2, output_is_nhwc));

  return AddMinMaxOperator(model_builder, node_unit, input1, input2, output_is_nhwc);
}

#pragma endregion

#pragma region op_elu

class EluOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

Status EluOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& input = node_unit.Inputs()[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  bool output_is_nhwc = model_builder.IsOperandNHWC(input);
  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  NodeAttrHelper helper(node_unit);
  const auto alpha = helper.Get("alpha", 1.0f);
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, alpha);
  return model_builder.AddOperation(ANEURALNETWORKS_ELU, input_indices,
                                    {output}, {output_operand_type}, {output_is_nhwc});
}

#pragma endregion

#pragma region op_slice

class SliceOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;
};

void SliceOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  // Skip everything except input0 for Slice
  const auto& inputs = node_unit.Inputs();
  model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());  // starts
  model_builder.AddInitializerToSkip(inputs[2].node_arg.Name());  // ends
  if (inputs.size() > 3) {
    model_builder.AddInitializerToSkip(inputs[3].node_arg.Name());  // axes
    if (inputs.size() > 4) {
      model_builder.AddInitializerToSkip(inputs[4].node_arg.Name());  // steps
    }
  }
}

Status SliceOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& inputs = node_unit.Inputs();
  const auto& input_shape = shaper[inputs[0].node_arg.Name()];
  std::vector<int64_t> input_shape_64(input_shape.cbegin(), input_shape.cend());
  SliceOp::PrepareForComputeMetadata compute_metadata(input_shape_64);

  {
    // We need to copy the data from the starts/ends/axes/steps initializers to int64 vectors
    // to be used in shared PrepareForCompute function to calculate the output shape
    // and normalize inputs, for example, input can be starts/ends/steps for certain axes,
    // PrepareForCompute can generate standard starts/ends/steps/axes for each axes
    std::vector<int64_t> input_starts;
    std::vector<int64_t> input_ends;
    std::vector<int64_t> input_axes;
    std::vector<int64_t> input_steps;

    const auto CopyInputData = [&inputs, &model_builder](size_t input_idx, std::vector<int64_t>& data) {
      data.clear();

      // This is an optional input, return empty vector
      if (inputs.size() <= input_idx)
        return Status::OK();

      const auto& input_name = inputs[input_idx].node_arg.Name();
      const auto& initializers(model_builder.GetInitializerTensors());

      const auto& tensor = *initializers.at(input_name);
      std::vector<uint8_t> unpacked_tensor;
      ORT_RETURN_IF_ERROR(
          onnxruntime::utils::UnpackInitializerData(tensor, model_builder.GetGraphViewer().ModelPath(),
                                                    unpacked_tensor));
      size_t tensor_byte_size = unpacked_tensor.size();
      const auto data_type = tensor.data_type();
      if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
        const int64_t* tensor_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
        size_t size = tensor_byte_size / sizeof(int64_t);
        data.insert(data.end(), tensor_data, tensor_data + size);
      } else if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
        const int32_t* tensor_data = reinterpret_cast<const int32_t*>(unpacked_tensor.data());
        size_t size = tensor_byte_size / sizeof(int32_t);
        data.insert(data.end(), tensor_data, tensor_data + size);
      } else {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Data type for starts and ends inputs' is not supported in this build. Got ",
                               data_type);
      }

      return Status::OK();
    };

    ORT_RETURN_IF_ERROR(CopyInputData(1, input_starts));
    ORT_RETURN_IF_ERROR(CopyInputData(2, input_ends));
    ORT_RETURN_IF_ERROR(CopyInputData(3, input_axes));
    ORT_RETURN_IF_ERROR(CopyInputData(4, input_steps));
    ORT_RETURN_IF_ERROR(
        SliceOp::PrepareForComputeHelper(input_starts, input_ends, input_axes, input_steps, compute_metadata));
  }

  // output shape is of type uint32_t, convert from int64 compute_metadata.output_dims_
  Shape nnapi_output_shape;
  nnapi_output_shape.reserve(compute_metadata.output_dims_.size());
  std::transform(compute_metadata.output_dims_.cbegin(), compute_metadata.output_dims_.cend(),
                 std::back_inserter(nnapi_output_shape),
                 [](int64_t i) { return SafeInt<uint32_t>(i); });

  const auto& input = inputs[0].node_arg.Name();
  const auto& output = node_unit.Outputs()[0].node_arg.Name();
  bool output_is_nhwc = model_builder.IsOperandNHWC(input);

  // No shape inference for Slice, everything is calculated here, we only need to add the output shape
  // to the shaper
  shaper.AddShape(output, nnapi_output_shape);
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));

  // begin/end/strides of ANEURALNETWORKS_STRIDED_SLICE have the same shape
  Shape param_dimen = {static_cast<uint32_t>(input_shape.size())};

  // helper function to add begin/end/strides of ANEURALNETWORKS_STRIDED_SLICE
  const auto AddOperand = [&model_builder, &node_unit, &input_indices, &operand_indices](
                              const char* name, const Shape& shape, const std::vector<int64_t>& param_raw_data) {
    std::vector<int32_t> param_data;
    param_data.reserve(param_raw_data.size());
    std::transform(param_raw_data.cbegin(), param_raw_data.cend(),
                   std::back_inserter(param_data),
                   [](int64_t i) { return SafeInt<int32_t>(i); });
    std::string param_name = model_builder.GetUniqueName(node_unit.Name() + name);
    OperandType param_operand_type(Type::TENSOR_INT32, shape);
    ORT_RETURN_IF_ERROR(
        model_builder.AddOperandFromPersistMemoryBuffer(param_name, param_data.data(), param_operand_type));
    input_indices.push_back(operand_indices.at(param_name));
    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(AddOperand("starts", param_dimen, compute_metadata.starts_));  //nnapi_begin

  // NNAPI has 2 slice operations
  // - ANEURALNETWORKS_SLICE
  //    Simpler and faster version of slice without steps, available from ANEURALNETWORKS_FEATURE_LEVEL_3
  //    Use this one if no step other than 1 is used in ONNX slice
  // - ANEURALNETWORKS_STRIDED_SLICE
  //    More comprehensive version, available from ANEURALNETWORKS_FEATURE_LEVEL_2
  int op_code = ANEURALNETWORKS_STRIDED_SLICE;
  if (std::all_of(compute_metadata.steps_.cbegin(),
                  compute_metadata.steps_.cend(),
                  [](int64_t i) { return i == 1; }) &&
      model_builder.GetNNAPIFeatureLevel() > ANEURALNETWORKS_FEATURE_LEVEL_2) {
    op_code = ANEURALNETWORKS_SLICE;
    // the nnapi size of the slice in this case is the output shape
    ORT_RETURN_IF_ERROR(AddOperand("sizes", param_dimen, compute_metadata.output_dims_));  //nnapi_sizes
  } else {
    // ** The special treatment of ends **
    // The nnapi_end need some special handling, based on the current undocumented design of
    // ANEURALNETWORKS_STRIDED_SLICE
    // For ORT, for a single axis, after SliceOp::PrepareForCompute, and the step is negative,
    // and the last element for slice is at the beginning of the axis (we are slicing backwards)
    // The end for this axis will be -1
    // For NNAPI, it is not documented that end can be negative,
    // see https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1ggaabbe492c60331b13038e39d4207940e0a89695302f8b1e7ae7ce8f4d8c0b8a752
    // However, the actual NNAPI StridedSlice has some odd implementations,
    // See https://android.googlesource.com/platform/frameworks/ml/+/5b525d4d9100819d87447bd2c2a0bcfdd62899ee/nn/common/operations/StridedSlice.cpp#177
    // and, https://android.googlesource.com/platform/frameworks/ml/+/5b525d4d9100819d87447bd2c2a0bcfdd62899ee/nn/common/include/OperationsUtils.h#262
    // If a negative end is no less than -dim (dimension of the axis), it will be treated as an index counting from
    // the end, for example, dim = 5, and end = -1, the end will be normalized to 4, which will cause
    // incorrect result, so here we have to make the end = -dim - 1 such that it will not be treated as
    // an index counting from the end.
    std::vector<int64_t> ends = compute_metadata.ends_;
    for (size_t i = 0; i < ends.size(); ++i) {
      if (ends[i] == -1) {
        ends[i] = -static_cast<int32_t>(input_shape[i] + 1);
      }
    }
    ORT_RETURN_IF_ERROR(AddOperand("ends", param_dimen, ends));                      //nnapi_end
    ORT_RETURN_IF_ERROR(AddOperand("steps", param_dimen, compute_metadata.steps_));  //nnapi_strides
    // We do not use the following inputs in ANEURALNETWORKS_STRIDED_SLICE, set them all to 0
    ADD_SCALAR_OPERAND(model_builder, input_indices, 0);  // begin_mask
    ADD_SCALAR_OPERAND(model_builder, input_indices, 0);  // end_mask
    ADD_SCALAR_OPERAND(model_builder, input_indices, 0);  // shrink_axis_mask
  }
  return model_builder.AddOperation(op_code, input_indices, {output}, {output_operand_type}, {output_is_nhwc});
}

#pragma endregion

#pragma region CreateGetOpBuilders

// The reason we use macros to create OpBuilders is for easy exclusion in build if certain op(s) are not used
// such that we can reduce binary size.
// This is for multiple ops share the same OpBuilder, we only need create one for all of them
#define NNAPI_EP_ADD_SHARED_OP_BUILDER(OP_TYPE, BUILDER_NAME) \
  BUILDER_NAME::CreateSharedOpBuilder(OP_TYPE, op_registrations);

// This is for ops with dedicated OpBuilder
#define NNAPI_EP_ADD_SINGLE_OP_BUILDER(OP_TYPE, BUILDER_NAME)                                 \
  {                                                                                           \
    op_registrations.builders.push_back(std::make_unique<BUILDER_NAME>());                    \
    op_registrations.op_builder_map.emplace(OP_TYPE, op_registrations.builders.back().get()); \
  }

static OpBuilderRegistrations CreateOpBuilderRegistrations() {
  OpBuilderRegistrations op_registrations;

  // Builders handle a single op
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("BatchNormalization", BatchNormalizationOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Cast", CastOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Clip", ClipOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Concat", ConcatOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("DequantizeLinear", DequantizeLinearOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Elu", EluOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Flatten", FlattenOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Identity", IdentityOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("LRN", LRNOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("QuantizeLinear", QuantizeLinearOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Relu", ReluOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Reshape", ReshapeOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Resize", ResizeOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Slice", SliceOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Softmax", SoftMaxOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Squeeze", SqueezeOpBuilder);
  NNAPI_EP_ADD_SINGLE_OP_BUILDER("Transpose", TransposeOpBuilder);

  // Builders shared among similar ops
  {
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Add", BinaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Div", BinaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Mul", BinaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Pow", BinaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("QLinearAdd", BinaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Sub", BinaryOpBuilder);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_BUILDER("AveragePool", PoolOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("GlobalAveragePool", PoolOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("GlobalMaxPool", PoolOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("MaxPool", PoolOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("QLinearAveragePool", PoolOpBuilder);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Conv", ConvOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("QLinearConv", ConvOpBuilder);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Gemm", GemmOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("MatMul", GemmOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("QLinearMatMul", GemmOpBuilder);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Abs", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Exp", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Floor", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Log", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Neg", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("QLinearSigmoid", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Sigmoid", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Sin", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Sqrt", UnaryOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Tanh", UnaryOpBuilder);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Max", MinMaxOpBuilder);
    NNAPI_EP_ADD_SHARED_OP_BUILDER("Min", MinMaxOpBuilder);
  }

  return op_registrations;
}

const std::unordered_map<std::string, const IOpBuilder*>& GetOpBuilders() {
  static const OpBuilderRegistrations op_registrations = CreateOpBuilderRegistrations();
  return op_registrations.op_builder_map;
}

#pragma endregion

}  // namespace nnapi
}  // namespace onnxruntime