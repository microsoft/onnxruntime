// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <vector>

#include "core/common/common.h"
#include "core/providers/common.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksWrapper.h"
#include "core/providers/shared/node_unit/node_unit.h"

namespace onnxruntime::nnapi::op_builder_helpers {

using android::nn::wrapper::OperandType, android::nn::wrapper::Type;

enum DataLayout {
  L_0231 = 0,
  L_1230 = 1,
};

// Add operator related helpers

// adds a scalar operand to the NNAPI model and appends its index to `input_indices`
template <typename T>
Status AddScalarOperand(ModelBuilder& model_builder, InlinedVector<uint32_t>& input_indices, T scalar_value) {
  uint32_t index = 0;
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromScalar(std::move(scalar_value), index));
  input_indices.push_back(index);
  return Status::OK();
}

#define ADD_SCALAR_OPERAND(model_builder, input_indices, scalar_value)          \
  ORT_RETURN_IF_ERROR(onnxruntime::nnapi::op_builder_helpers::AddScalarOperand( \
      model_builder, input_indices, scalar_value))

// adds ANEURALNETWORKS_TRANSPOSE operation
Status AddNnapiTranspose(ModelBuilder& model_builder,
                         const std::string& data_input,
                         const std::string& perm_input, const gsl::span<const int32_t> perm,
                         const std::string& output);

// adds ANEURALNETWORKS_RESHAPE operation
Status AddNnapiReshape(ModelBuilder& model_builder,
                       const std::string& data_input,
                       const std::string& shape_input, const std::vector<int32_t>& shape_value,
                       const std::string& output);

// adds ANEURALNETWORKS_SPLIT operation
Status AddNnapiSplit(ModelBuilder& model_builder,
                     const std::string& input,
                     int32_t axis,
                     const std::vector<std::string>& outputs);

Status AddNnapiBatchNormalization(ModelBuilder& model_builder,
                                  const std::string& input1,
                                  const std::string& input2,
                                  const std::string& input3,
                                  const std::string& output1,
                                  const std::string& output2,
                                  int32_t fuse_code,
                                  float output_scale = 0.0f,
                                  int32_t output_zero_point = 0);

// checks whether batch MatMul in the given NodeUnit is supported by NNAPI EP
bool IsSupportedBatchMatMul(const NodeUnit& node_unit, int32_t nnapi_feature_level);

// builds a batch MatMul in the NNAPI model from the given NodeUnit
// note: the pre-conditions of this function are checked in IsSupportedBatchMatMul()
Status BuildBatchMatMul(ModelBuilder& model_builder, const NodeUnit& node_unit);

// This is primarily used for adding the weight (an initializer) of Conv/QlinearConv
// And perform layout change from ONNX -> NNAPI
// If is_per_tensor_u8s8 is true, the QlinearConv is per-tensor u8s8 (input X is unsigned int8
// and weight W is signed int8 and it is per-tensor (NOT per-channel) quantized), in this case,
// since NNAPI requires X and W to be same type for per-tensor quantization,
// the initializer tensor W will be converted from int8 to uint8 by flip each byte by XOR 0x80
// byte ^ 0x80 == byte + 128
Status AddInitializerInNewLayout(ModelBuilder& model_builder,
                                 const std::string& name,
                                 const OperandType& source_operand_type,
                                 DataLayout new_layout,
                                 bool is_per_tensor_u8s8);

// This is primarily used for adding the input B (an initializer) of MatMul/QLinearMatMul/Gemm (not transposed)
// and transpose it, since for NNAPI only supports A*B'
//
// If is_per_tensor_u8s8 is true, the QLinearMatMul is per-tensor u8s8 (input A is unsigned int8
// and input B is signed int8), in this case, since NNAPI requires A and B to be same type,
// the initializer tensor B will be converted from int8 to uint8 by flip each byte by XOR 0x80
// byte ^ 0x80 == byte + 128
Status AddInitializerTransposed(ModelBuilder& model_builder,
                                const OperandType& source_operand_type,
                                const std::string& name,
                                bool is_per_tensor_u8s8);

Status ComputeConvPads(const Shape& input_dimen,
                       const uint32_t weight_size_y, const uint32_t weight_size_x,
                       const std::vector<int32_t>& onnx_pads, const std::vector<int32_t>& onnx_strides, const std::vector<int32_t>& onnx_dilations,
                       AutoPadType auto_pad_type, bool nchw,
                       std::vector<int32_t>& pads_out);

Status HandleAutoPad(const Shape& input_shape,
                     const uint32_t weight_size_y,
                     const uint32_t weight_size_x,
                     const std::vector<int32_t>& onnx_strides,
                     const std::vector<int32_t>& onnx_dilations,
                     AutoPadType auto_pad_type,
                     bool use_nchw,
                     std::vector<int32_t>& onnx_pads,
                     int32_t& nnapi_padding_code,
                     bool& use_auto_pad);

// Get scales and zero points for the qlinear binary ops (which has 2 input and 1 output)
// QLinearConv, QLinearMatmul, QLinearAdd, QLinearMul
// a, b are inputs, and y is output
Status GetBinaryOpQuantizationScaleAndZeroPoint(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                                float& a_scale, float& b_scale, float& y_scale,
                                                int32_t& a_zero_point, int32_t& b_zero_point, int32_t& y_zero_point);

// Get scale and zero point for
// [QLinearConv] input, weight, output
// [QLinearMatMul] A, B, Y
//
// In case of u8s8 (input/A is uint8 and weight/B is int8)
// If the QlinearConv is using per-channel u8s8, return the scales vector
// If the Qlinear[Conv/MatMul] is using per-tensor u8s8, the weight/B tensor
// will be convert to uint8 later, will return the same scale and 128 as zero point
// Also will set is_per_tensor_u8s8 to true to be used later
Status GetConvMatMulOpQuantizationScaleAndZeroPoint(const ModelBuilder& model_builder, const NodeUnit& node_unit,
                                                    float& a_scale, float& w_scale, float& y_scale,
                                                    int32_t& a_zero_point, int32_t& w_zero_point, int32_t& y_zero_point,
                                                    std::optional<std::vector<float>>& w_scales, bool& is_per_tensor_u8s8);

// NNAPI has the quantization scale and zero point embedded in the ANeuralNetworksOperandType
// ONNX has the quantization scale and zero point as the inputs of the qlinear operators
// We want to verify the scale and zeropoint of the ONNX inputs matches the values embedded in the NNAPI inputs
Status IsValidInputQuantizedType(const ModelBuilder& model_builder,
                                 const std::string& input_name,
                                 float scale,
                                 int32_t zero_point);

Status IsValidConvWeightQuantizedType(const ModelBuilder& model_builder,
                                      const std::string& input_name,
                                      float scale,
                                      int32_t zero_point,
                                      const std::optional<std::vector<float>>& scales);

Status IsOpInRequiredLayout(bool use_nchw, const NodeUnit& node_unit);

void AddQuantizationScaleAndZeroPointToSkip(ModelBuilder& model_builder,
                                            const NodeUnitIODef::QuantParam& quant_param);

// Ignore the input (with quantization scale and ZP if available)
// The input (usually weight) is already embedded in the NNAPI model
void AddInputToSkip(ModelBuilder& model_builder, const NodeUnitIODef& io_def);

Status AddBinaryOperator(int32_t op_type,
                         ModelBuilder& model_builder,
                         const std::string& input1,
                         const std::string& input2,
                         bool add_activation,
                         int32_t fuse_code,
                         const std::string& output,
                         float output_scale = 0.0f,
                         int32_t output_zero_point = 0);

Status AddSqueezeOp(ModelBuilder& model_builder,
                    const std::string& node_name,
                    const std::string& input,
                    const std::string& output,
                    std::vector<int32_t> axes);

Status AddMinMaxOperator(ModelBuilder& model_builder, const NodeUnit& node_unit,
                         const std::string& input1, const std::string& input2);

Status AddReshapeOperator(ModelBuilder& model_builder, const NodeUnit& node_unit,
                          const std::string& input, const std::vector<int32_t>& shape);

Status GetAxesForSqueezeAndUnSqueeze(ModelBuilder& model_builder, const NodeUnit& node_unit,
                                     std::vector<int32_t>& axes);

// Operator support related helpers

inline bool IsNodeLayoutNHWC(const NodeUnit& node_unit) {
  return node_unit.Domain() == kMSInternalNHWCDomain;
}

bool IsQuantizationScaleSupported(const GraphViewer& graph_viewer,
                                  const NodeUnitIODef& io_def,
                                  const OpSupportCheckParams& params,
                                  const std::string& op_type,
                                  bool is_quant_matmul,
                                  bool is_conv_matmul_u8s8_weight);

bool IsQuantizationZeroPointSupported(const GraphViewer& graph_viewer,
                                      const NodeUnitIODef& io_def,
                                      const std::string& op_type,
                                      const Path& model_path,
                                      bool is_quant_matmul,
                                      bool is_conv_matmul_u8s8_weight);

// Check if the given quantized input(s) or output(s) is supported
bool IsQuantizedIOSupported(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                            const std::vector<size_t>& indices, const OpSupportCheckParams& params, ArgType arg_type);

// Some Quantized NNAPI operations have required output scale and zero point
// e.g. Softmax (uint8) requires output scale be 1.f/256 and zp be 0
// This helper function checks if the given io_def has required scale and zp
bool HasRequiredScaleAndZeroPoint(const GraphViewer& graph_viewer,
                                  const std::string& op_desc,
                                  const NodeUnitIODef& io_def,
                                  const Path& path,
                                  float required_scale, int32_t required_zp);

// performs broadcasting operation on two shapes to make them compatible
Status PerformBroadcasting(const Shape& shape1, const Shape& shape2, Shape& output_shape);

}  // namespace onnxruntime::nnapi::op_builder_helpers
