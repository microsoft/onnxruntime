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

namespace {
// Get the bias size (C) of Gemm op
// ANEURALNETWORKS_FULLY_CONNECTED only supports 1d bias
// Will test if C of Gemm can be squeezed and return the 1d vector size after squeeze
bool GetGemmBiasSize(const Shape& c_shape, int32_t android_feature_level, uint32_t& size) {
  // TODO add support of scalar C for Gemm
  size_t c_dim = c_shape.size();
  if (c_dim == 0) {
    LOGS_DEFAULT(VERBOSE) << "C of Gemm cannot be a scalar";
    return false;
  }

  if (c_dim != 1 && android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_2) {
    LOGS_DEFAULT(VERBOSE) << "C of Gemm can only be 1d tensor for API level " << android_feature_level
                          << " shape of C, " << Shape2String(c_shape);
    return false;
  }

  if (c_dim != 1) {
    // If C is a (2+)d tensor, it must have the format {1, 1, ..., 1, n}
    // where every except the last dimension should be 1
    for (size_t i = 0; i < c_dim - 1; ++i) {
      if (c_shape[i] != 1) {
        LOGS_DEFAULT(VERBOSE) << "C of Gemm must be a vector or a tensor with only last dimension != 1"
                              << " c_shape: " << Shape2String(c_shape);
        return false;
      }
    }
  }

  size = c_shape[c_dim - 1];
  return true;
}

}  // namespace

class GemmOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const NodeUnit& node_unit) const override;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
  bool HasSupportedInputOutputsImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                    const OpSupportCheckParams& params) const override;
  int GetMinSupportedOpSet(const NodeUnit& node_unit) const override;

  bool IsNodeUnitTypeSupported(const NodeUnit& /* node_unit */) const override { return true; }

  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

// Add operator related

void GemmOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const NodeUnit& node_unit) const {
  if (op_builder_helpers::IsSupportedBatchMatMul(node_unit, model_builder.GetEffectiveFeatureLevel())) {
    // no initializers to skip for batch matmul
    return;
  }

  const auto& inputs = node_unit.Inputs();
  if (IsQuantizedOp(node_unit)) {
    if (node_unit.OpType() == "QLinearMatMul" || node_unit.OpType() == "MatMul") {                 // QLinear/QDQMatMul
      AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[0].quant_param);               // a_scale, a_zp
      AddInputToSkip(model_builder, inputs[1]);                                                    // b, b_scale, b_zp
      AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
    } else if (node_unit.OpType() == "Gemm") {                                                     // QDQGemm
      AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[0].quant_param);               // a_scale, a_zp
      AddQuantizationScaleAndZeroPointToSkip(model_builder, *inputs[1].quant_param);               // b_scale, b_zp

      NodeAttrHelper helper(node_unit);
      const auto transB = helper.Get("transB", 0);
      // For transB == 0, we need to transpose it and add transposed initializer later into nnapi model,
      // not directly using it here, so add to skip list.
      if (transB == 0)
        model_builder.AddInitializerToSkip(inputs[1].node_arg.Name());

      if (inputs.size() > 2) {
        AddInputToSkip(model_builder, inputs[2]);  // c, c_scale, c_zp (bias)
      }
      AddQuantizationScaleAndZeroPointToSkip(model_builder, *node_unit.Outputs()[0].quant_param);  // y_scale, y_zp
    }
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
  if (op_builder_helpers::IsSupportedBatchMatMul(node_unit, model_builder.GetEffectiveFeatureLevel())) {
    return op_builder_helpers::BuildBatchMatMul(model_builder, node_unit);
  }

  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());

  const auto& op = node_unit.OpType();
  const auto& inputs = node_unit.Inputs();
  NodeAttrHelper helper(node_unit);

  const auto quant_type = GetQuantizedOpType(node_unit);
  const bool is_quant_matmul = (quant_type == QuantizedOpType::QDQMatMul ||
                                quant_type == QuantizedOpType::QLinearMatMul);
  const bool is_quant_gemm = quant_type == QuantizedOpType::QDQGemm;

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
  if (is_quant_matmul || is_quant_gemm) {
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
    if (!is_quant_matmul && !is_quant_gemm)
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
  if (is_quant_matmul || is_quant_gemm) {
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input1, a_scale, a_zero_point));
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input2, b_scale, b_zero_point));
  }

  uint32_t bias_idx;
  bool has_bias = inputs.size() > 2;
  if (has_bias) {
    const auto& bias = inputs[2].node_arg.Name();
    if (!is_quant_gemm) {
      // We need to squeeze the input tensor to 1d if necessary
      if (shaper[bias].size() > 1) {
        std::string bias_squeezed = model_builder.GetUniqueName(node_unit.Name() + op + "_bias_squeezed");
        // We will use squeeze all here
        ORT_RETURN_IF_ERROR(AddSqueezeOp(model_builder, node_unit.Name(),
                                         bias, bias_squeezed, {} /* axes */));
        bias_idx = operand_indices.at(bias_squeezed);
        LOGS_DEFAULT(VERBOSE) << "GemmOpBuilder - Operand [" << bias << "] squeezed from "
                              << Shape2String(shaper[bias])
                              << " to "
                              << Shape2String(shaper[bias_squeezed]);
      } else {
        bias_idx = operand_indices.at(bias);
      }
    } else {  // is_quant_gemm
      const auto& bias_tensor = *model_builder.GetInitializerTensors().at(bias);
      // QGemm has a contraint on input C to be int32 type
      ORT_RETURN_IF_NOT(bias_tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32,
                        "bias of QDQGemm should be int32, actual type: ", bias_tensor.data_type());
      Shape bias_dimen;
      for (auto dim : bias_tensor.dims())
        bias_dimen.push_back(SafeInt<uint32_t>(dim));
      Initializer unpacked_tensor(bias_tensor);
      OperandType bias_operand_type(Type::TENSOR_INT32, bias_dimen, a_scale * b_scale);
      ORT_RETURN_IF_ERROR(
          model_builder.AddOperandFromPersistMemoryBuffer(
              bias,
              unpacked_tensor.data<int32_t>(), bias_operand_type));

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

  InlinedVector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));  // A
  input_indices.push_back(input_2_idx);                 // B
  input_indices.push_back(bias_idx);                    // C
  int32_t fuse_code = model_builder.FindActivation(node_unit);
  ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);

  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output], y_scale, y_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_FULLY_CONNECTED, input_indices,
                                                 {output}, {output_operand_type}));
  return Status::OK();
}

// Operator support related

bool GemmOpBuilder::HasSupportedInputOutputsImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                                 const OpSupportCheckParams& params) const {
  if (!IsQuantizedOp(node_unit)) {
    return BaseOpBuilder::HasSupportedInputOutputsImpl(graph_viewer, node_unit, params);
  }

  // QLinearMatMul/QDQGemm/QDQMatMul
  if (!HasValidBinaryOpQuantizedInputTypes(node_unit))
    return false;

  if (!IsQuantizedIOSupported(graph_viewer, node_unit, {0, 1}, params, ArgType::kInput))
    return false;

  if (!IsQuantizedIOSupported(graph_viewer, node_unit, {0}, params, ArgType::kOutput))
    return false;

  return true;
}

int GemmOpBuilder::GetMinSupportedOpSet(const NodeUnit& node_unit) const {
  const auto& op(node_unit.OpType());

  // Gemm opset 6- has broadcast attributes we do not support now
  if (op == "Gemm")
    return 7;

  return 1;
}

bool GemmOpBuilder::IsQuantizedOp(const NodeUnit& node_unit) const {
  return IsQuantizedGemm(GetQuantizedOpType(node_unit));
}

bool GemmOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer, const NodeUnit& node_unit,
                                      const OpSupportCheckParams& params) const {
  // check batch matmul first, then fall back to checking single gemm/matmul
  {
    const bool is_supported_batch_matmul =
        op_builder_helpers::IsSupportedBatchMatMul(node_unit, params.android_feature_level);
    LOGS_DEFAULT(VERBOSE) << "Supported batch matmul: [" << is_supported_batch_matmul << "]";
    if (is_supported_batch_matmul) {
      return true;
    }
  }

  const auto& op_type = node_unit.OpType();
  const auto& inputs = node_unit.Inputs();
  const bool is_qlinear_matmul = op_type == "QLinearMatMul";
  const auto quant_type = GetQuantizedOpType(node_unit);
  const bool is_quant_gemm = quant_type == QuantizedOpType::QDQGemm;

  Shape a_shape;
  {
    if (!GetShape(inputs[0].node_arg, a_shape))
      return false;

    if (a_shape.size() != 2) {
      LOGS_DEFAULT(VERBOSE) << "A must be 2D";
      return false;
    }
  }

  Shape b_shape;
  {
    if (!GetShape(inputs[1].node_arg, b_shape))
      return false;

    if (b_shape.size() != 2) {
      LOGS_DEFAULT(VERBOSE) << "B must be 2D";
      return false;
    }
  }

  if (op_type == "Gemm") {
    // Only support
    // 1. A*B'+C
    // 2. A*B+C and B is an initializer
    NodeAttrHelper helper(node_unit);
    const auto transA = helper.Get("transA", 0);
    const auto transB = helper.Get("transB", 0);
    const auto alpha = helper.Get("alpha", 1.0f);
    const auto beta = helper.Get("beta", 1.0f);

    if (!(transA == 0 && alpha == 1.f && beta == 1.f)) {
      LOGS_DEFAULT(VERBOSE) << "Only transA == 0, alpha == 1.0 "
                            << "and beta == 1.0 is supported."
                            << " transA " << transA
                            << " transB " << transB
                            << " alpha " << alpha
                            << " beta " << beta;
      return false;
    }

    if (transB == 0 && !graph_viewer.GetConstantInitializer(inputs[1].node_arg.Name())) {
      LOGS_DEFAULT(VERBOSE) << "B of Gemm must be a constant initializer if transB != 1";
      return false;
    }

    if (inputs.size() == 3) {
      Shape c_shape;
      if (!GetShape(inputs[2].node_arg, c_shape))
        return false;

      uint32_t c_size;
      if (!GetGemmBiasSize(c_shape, params.android_feature_level, c_size))
        return false;

      if (c_size != (transB == 0 ? b_shape[1] : b_shape[0])) {
        LOGS_DEFAULT(VERBOSE) << "C of Gemm must be a vector of b_shape["
                              << (transB == 0 ? "1" : "0") << "]"
                              << " b_shape: " << Shape2String(b_shape)
                              << " c_shape: " << Shape2String(c_shape);

        return false;
      }
    }
  } else if (op_type == "MatMul" || is_qlinear_matmul) {
    // Only support A*B B is an initializer
    if (!graph_viewer.GetConstantInitializer(inputs[1].node_arg.Name())) {
      LOGS_DEFAULT(VERBOSE) << "B of MatMul must be a constant initializer";
      return false;
    }
  } else {
    LOGS_DEFAULT(VERBOSE) << "GemmOpSupportChecker, unknown op: " << op_type;
  }

  if (is_quant_gemm) {
    if (inputs.size() > 2 && !graph_viewer.GetConstantInitializer(inputs[2].node_arg.Name())) {
      LOGS_DEFAULT(VERBOSE) << "Bias of QDQ Gemm must be a constant initializer";
      return false;
    }
  }

  return true;
}

void CreateGemmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  CreateSharedOpBuilderImpl<GemmOpBuilder>(
      op_type, op_registrations,
      {
          "Gemm",
          "MatMul",
          "QLinearMatMul",
      });
}

}  // namespace nnapi
}  // namespace onnxruntime
