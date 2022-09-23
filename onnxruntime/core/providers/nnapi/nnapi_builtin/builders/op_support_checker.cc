// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "op_support_checker.h"

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph.h"
#include "core/providers/common.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_helpers.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_support_checker.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_checker_factory.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_helpers.h"

namespace onnxruntime {
namespace nnapi {



#pragma region op_batchnormalization

class BatchNormalizationOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  // BatchNormalization opset 6- has unsupported attributes
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 7; }
};

void CreateBatchNormalizationOpSupportChecker(
    const std::string& op_type,
    OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<BatchNormalizationOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool BatchNormalizationOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                                           const OpSupportCheckParams& /* params */) const {
  if (node_unit.Outputs().size() != 1) {
    LOGS_DEFAULT(VERBOSE) << "Your onnx model may be in training mode, please export "
                             "it in test mode.";
    return false;
  }

  const auto& inputs = node_unit.Inputs();
  Shape input_shape;
  if (!GetShape(inputs[0].node_arg, input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size > 4) {
    LOGS_DEFAULT(VERBOSE) << "BN only support up to 4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  NodeAttrHelper helper(node_unit);
  const auto spatial = helper.Get("spatial", 1);
  if (spatial != 1) {
    LOGS_DEFAULT(VERBOSE) << "Non-spatial BN is not supported";
    return false;
  }

  const auto& scale_name = inputs[1].node_arg.Name();
  const auto& b_name = inputs[2].node_arg.Name();
  const auto& mean_name = inputs[3].node_arg.Name();
  const auto& var_name = inputs[4].node_arg.Name();
  if (!Contains(initializers, scale_name)) {
    LOGS_DEFAULT(VERBOSE) << "Scale of BN must be known";
    return false;
  }
  if (!Contains(initializers, b_name)) {
    LOGS_DEFAULT(VERBOSE) << "B of BN must be known";
    return false;
  }
  if (!Contains(initializers, mean_name)) {
    LOGS_DEFAULT(VERBOSE) << "Mean of BN must be known";
    return false;
  }
  if (!Contains(initializers, var_name)) {
    LOGS_DEFAULT(VERBOSE) << "Var of BN must be known";
    return false;
  }

  return true;
}

#pragma endregion

#pragma region op_pool

class PoolOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& params) const override {
    return params.use_nchw ? ANEURALNETWORKS_FEATURE_LEVEL_3 : ANEURALNETWORKS_FEATURE_LEVEL_2;
  }

  bool HasSupportedInputOutputsImpl(
      const InitializedTensorSet& initializers, const NodeUnit& node_unit,
      const OpSupportCheckParams& params) const override;
  bool IsNodeUnitTypeSupported(const NodeUnit& /* node_unit */) const override;
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

void CreatePoolOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  CreateSharedOpSupportCheckerImpl<PoolOpSupportChecker>(
      op_type, op_registrations,
      {
          "GlobalAveragePool",
          "GlobalMaxPool",
          "AveragePool",
          "MaxPool",
          "QLinearAveragePool",
      });
}

bool PoolOpSupportChecker::IsNodeUnitTypeSupported(const NodeUnit& node_unit) const {
  if (node_unit.UnitType() == NodeUnit::Type::QDQGroup) {
    const auto quant_type = GetQuantizedOpType(node_unit);
    return quant_type == QuantizedOpType::QDQAveragePool;
  }

  return true;
}

bool PoolOpSupportChecker::IsQuantizedOp(const NodeUnit& node_unit) const {
  return IsQuantizedPool(GetQuantizedOpType(node_unit));
}

bool PoolOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                             const OpSupportCheckParams& /* params */) const {
  const auto& op_name = node_unit.Name();
  const auto& op_type = node_unit.OpType();
  const auto& inputs = node_unit.Inputs();
  Shape input_shape;
  if (!GetShape(inputs[0].node_arg, input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS_DEFAULT(VERBOSE)
        << op_type << " only supports rank-4 tensor, input ["
        << inputs[0].node_arg.Name() << "] has actual dim count " << input_size;
    return false;
  }

  bool is_quant_pool = IsQuantizedOp(node_unit);
  bool is_average_pool = op_type == "AveragePool" || op_type == "QLinearAveragePool";
  if (is_average_pool || op_type == "MaxPool") {
    NodeAttrHelper helper(node_unit);

    const auto count_include_pad = helper.Get("count_include_pad", 0);
    if (count_include_pad == 1) {
      LOGS_DEFAULT(VERBOSE) << "count_include_pad == 1 is not supported";
      return false;
    }

    const auto storage_order = helper.Get("storage_order", 0);
    if (storage_order == 1) {
      LOGS_DEFAULT(VERBOSE) << "storage_order == 1 is not supported";
      return false;
    }

    if (helper.Get("kernel_shape", std::vector<int32_t>{1, 1}).size() != 2) {
      LOGS_DEFAULT(VERBOSE) << "Only pooling 2d is supported";
      return false;
    }

    if (helper.Get("ceil_mode", 0) == 1) {
      LOGS_DEFAULT(VERBOSE) << "ceil_mode == 1 is not supported for pooling";
      return false;
    }

    if (helper.Get("dilations", std::vector<int32_t>{1, 1}) !=
        std::vector<int32_t>{1, 1}) {
      LOGS_DEFAULT(VERBOSE) << "Dilations of pooling is not supported";
      return false;
    }

    if (node_unit.Outputs().size() != 1) {
      LOGS_DEFAULT(VERBOSE) << "Argmax in maxpooling is not supported";
      return false;
    }
  } else if (op_type != "GlobalAveragePool" && op_type != "GlobalMaxPool") {
    LOGS_DEFAULT(VERBOSE) << "PoolOpBuilder, unknown op: " << op_type;
    return false;
  }

  // We need to check if we have valid scales and zero points for QLinearAveragePool
  if (is_average_pool && is_quant_pool) {
    // NNAPI requires Quantized Average Pool has same scale and zero point for both input and output
    float input_scale = 0.0f;
    int32_t input_zp = 0;
    auto status = GetQuantizationScaleAndZeroPoint(
        initializers, node_unit.Inputs()[0], node_unit.ModelPath(), input_scale, input_zp);
    if (!status.IsOK()) {
      LOGS_DEFAULT(ERROR) << "Op [" << op_type << "] name [" << op_name
                          << "] GetQuantizationScaleAndZeroPoint for input_scale/zp failed, message: "
                          << status.ErrorMessage();
      return false;
    }

    float output_scale = 0.0f;
    int32_t output_zp = 0;
    status = GetQuantizationScaleAndZeroPoint(
        initializers, node_unit.Outputs()[0], node_unit.ModelPath(), output_scale, output_zp);
    if (!status.IsOK()) {
      LOGS_DEFAULT(ERROR) << "Op [" << op_type << "] name [" << op_name
                          << "] GetQuantizationScaleAndZeroPoint for output_scale/zp failed, message: "
                          << status.ErrorMessage();
      return false;
    }

    if (input_scale != output_scale) {
      LOGS_DEFAULT(VERBOSE) << "Op [" << op_type << "] name [" << op_name
                            << "] has different input_scale: " << input_scale
                            << " than the output_scale: " << output_scale;
      return false;
    }

    if (input_zp != output_zp) {
      LOGS_DEFAULT(VERBOSE) << "Op [" << op_type << "] name [" << op_name
                            << "] has different input_zp: " << input_zp
                            << " than the output_zp: " << output_zp;
      return false;
    }
  }

  return true;
}

bool PoolOpSupportChecker::HasSupportedInputOutputsImpl(
    const InitializedTensorSet& initializers, const NodeUnit& node_unit,
    const OpSupportCheckParams& params) const {
  const auto& op_type = node_unit.OpType();
  bool is_quant_pool = IsQuantizedOp(node_unit);
  bool is_max_pool = op_type == "MaxPool";
  bool is_average_pool = op_type == "AveragePool" || op_type == "QLinearAveragePool";
  bool is_quant_average_pool = is_quant_pool && is_average_pool;
  if (!is_max_pool && !is_quant_average_pool)
    return BaseOpSupportChecker::HasSupportedInputOutputsImpl(initializers, node_unit, params);

  if (is_quant_average_pool) {
    if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kInput))
      return false;

    if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput))
      return false;
  }

  // is_max_pool
  // For max pool, we can support both float and uint8 input
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

  return true;
}

#pragma endregion op_pool

#pragma region op_cast

class CastOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;
  }

  // Cast opset 5- uses string attribute for to type, is not supported for now
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 6; }
};

void CreateCastOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<CastOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool CastOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
                                             const OpSupportCheckParams& /* params */) const {
  NodeAttrHelper helper(node_unit);
  const auto to = helper.Get("to", 0);
  if (to != ONNX_NAMESPACE::TensorProto::FLOAT &&
      to != ONNX_NAMESPACE::TensorProto::INT32) {
    LOGS_DEFAULT(VERBOSE) << "[Cast] Only support cast to int32 or float, actual to type, " << to;
    return false;
  }

  return true;
}

#pragma endregion

#pragma region op_depthtospace

class DepthToSpaceOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

void CreateDepthToSpaceOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<DepthToSpaceOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool DepthToSpaceOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
                                                     const OpSupportCheckParams& params) const {
  NodeAttrHelper helper(node_unit);

  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS_DEFAULT(VERBOSE) << "DepthToSpace only supports 4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  if (params.use_nchw && params.android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
    LOGS_DEFAULT(VERBOSE) << "NCHW layout is not supported for android feature level: " << params.android_feature_level;
    return false;
  }

  auto since_version = node_unit.SinceVersion();
  if (since_version >= 11) {
    // For now, only DCR mode is accepted as NNAPI only supports DCR format data rearrangement
    const auto mode = helper.Get("mode", "DCR");
    if (mode != "DCR") {
      LOGS_DEFAULT(VERBOSE) << "ANEURALNETWORKS_DEPTH_TO_SPACE only supports DCR rearrangement mode. Current mode:" << mode;
      return false;
    }
  }

  return true;
}

#pragma endregion

#pragma region op_softmax

class SoftMaxOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_2;
  }
  bool HasSupportedInputOutputsImpl(
      const InitializedTensorSet& initializers, const NodeUnit& node_unit,
      const OpSupportCheckParams& params) const override;

  bool IsNodeUnitTypeSupported(const NodeUnit& /* node_unit */) const override { return true; }

  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

void CreateSoftMaxOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<SoftMaxOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool SoftMaxOpSupportChecker::IsQuantizedOp(const NodeUnit& node_unit) const {
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQSoftmax;
}

bool SoftMaxOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
                                                const OpSupportCheckParams& params) const {
  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 2 && input_size != 4) {
    LOGS_DEFAULT(VERBOSE) << "SoftMax only support 2d/4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  if (params.android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
    NodeAttrHelper helper(node_unit);
    int32_t axis = helper.Get("axis", 1);
    if (axis != 1) {
      LOGS_DEFAULT(VERBOSE)
          << "SoftMax only support axis 1 on Android API level: " << params.android_feature_level
          << " input axis: " << axis;
      return false;
    }
  }

  return true;
}

bool SoftMaxOpSupportChecker::HasSupportedInputOutputsImpl(
    const InitializedTensorSet& initializers, const NodeUnit& node_unit,
    const OpSupportCheckParams& params) const {
  if (!IsQuantizedOp(node_unit)) {
    return BaseOpSupportChecker::HasSupportedInputOutputsImpl(initializers, node_unit, params);
  }

  if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kInput)) {
    return false;
  }

  if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput)) {
    return false;
  }

  // NNAPI requires the scale be 1.f/256 and zero point to be 0
  if (!op_support_helpers::HasRequiredScaleAndZeroPoint(initializers,
                                                        MakeString("Op [", node_unit.OpType(), "] name [", node_unit.Name(), "]'s output 0 "),
                                                        node_unit.Outputs()[0], node_unit.ModelPath(),
                                                        1.f / 256 /* required_scale */, 0 /* required_zp */)) {
    return false;
  }

  return true;
}

#pragma endregion

#pragma region op_gemm

class GemmOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
  bool HasSupportedInputOutputsImpl(
      const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
      const OpSupportCheckParams& /* params */) const override;
  int GetMinSupportedOpSet(const NodeUnit& node_unit) const override;

  bool IsNodeUnitTypeSupported(const NodeUnit& /* node_unit */) const override { return true; }

  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

bool GemmOpSupportChecker::IsQuantizedOp(const NodeUnit& node_unit) const {
  return IsQuantizedGemm(GetQuantizedOpType(node_unit));
}

bool GemmOpSupportChecker::HasSupportedInputOutputsImpl(
    const InitializedTensorSet& initializers, const NodeUnit& node_unit,
    const OpSupportCheckParams& params) const {
  if (!IsQuantizedOp(node_unit)) {
    return BaseOpSupportChecker::HasSupportedInputOutputsImpl(initializers, node_unit, params);
  }

  // QLinearMatMul/QDQGemm/QDQMatMul
  if (!HasValidBinaryOpQuantizedInputTypes(node_unit))
    return false;

  if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0, 1}, params, ArgType::kInput))
    return false;

  if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput))
    return false;

  return true;
}

void CreateGemmOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  CreateSharedOpSupportCheckerImpl<GemmOpSupportChecker>(
      op_type, op_registrations,
      {
          "Gemm",
          "MatMul",
          "QLinearMatMul",
      });
}

// Get the bias size (C) of Gemm op
// ANEURALNETWORKS_FULLY_CONNECTED only supports 1d bias
// Will test if C of Gemm can be squeezed and return the 1d vector size after squeeze
static bool GetBiasSize(const Shape& c_shape, int32_t android_feature_level, uint32_t& size) {
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

int GemmOpSupportChecker::GetMinSupportedOpSet(const NodeUnit& node_unit) const {
  const auto& op(node_unit.OpType());

  // Gemm opset 6- has broadcast attributes we do not support now
  if (op == "Gemm")
    return 7;

  return 1;
}

bool GemmOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
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

    if (transB == 0 && !Contains(initializers, inputs[1].node_arg.Name())) {
      LOGS_DEFAULT(VERBOSE) << "B of Gemm must be known if transB != 1";
      return false;
    }

    if (inputs.size() == 3) {
      Shape c_shape;
      if (!GetShape(inputs[2].node_arg, c_shape))
        return false;

      uint32_t c_size;
      if (!GetBiasSize(c_shape, params.android_feature_level, c_size))
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
    if (!Contains(initializers, inputs[1].node_arg.Name())) {
      LOGS_DEFAULT(VERBOSE) << "B of MatMul must be known";
      return false;
    }
  } else {
    LOGS_DEFAULT(VERBOSE) << "GemmOpSupportChecker, unknown op: " << op_type;
  }

  if (is_quant_gemm) {
    if (inputs.size() > 2 && !Contains(initializers, inputs[2].node_arg.Name())) {
      LOGS_DEFAULT(VERBOSE) << "Bias of QDQ Gemm must be known";
      return false;
    }
  }

  return true;
}

#pragma endregion

#pragma region op_unary

class UnaryOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& params) const override;

  bool HasSupportedInputOutputsImpl(
      const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
      const OpSupportCheckParams& /* params */) const override;

  int GetMinSupportedOpSet(const NodeUnit& node_unit) const override;

  static bool IsQuantizedOpSupported(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                     const OpSupportCheckParams& params);
};

void CreateUnaryOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  CreateSharedOpSupportCheckerImpl<UnaryOpSupportChecker>(
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

bool UnaryOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                              const OpSupportCheckParams& params) const {
  if (node_unit.OpType() == "QLinearSigmoid")
    return IsQuantizedOpSupported(initializers, node_unit, params);
  else  // Everything except "QLinearSigmoid" are by default supported
    return true;
}

int32_t UnaryOpSupportChecker::GetMinSupportedNNAPIFeatureLevel(const NodeUnit& node_unit,
                                                                const OpSupportCheckParams& /* params */) const {
  const auto& op(node_unit.OpType());
  if (op == "Abs" ||
      op == "Exp" ||
      op == "Neg" ||
      op == "Sin" ||
      op == "Sqrt" ||
      op == "Log") {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;
  }

  return ANEURALNETWORKS_FEATURE_LEVEL_1;
}

bool UnaryOpSupportChecker::HasSupportedInputOutputsImpl(
    const InitializedTensorSet& initializers, const NodeUnit& node_unit,
    const OpSupportCheckParams& params) const {
  // We only need to override input check for QLinearSigmoid
  if (node_unit.OpType() != "QLinearSigmoid")
    return BaseOpSupportChecker::HasSupportedInputOutputsImpl(initializers, node_unit, params);

  if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kInput))
    return false;

  if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput))
    return false;

  return true;
}

// All ops except "Sin" opset 5- uses consumed_inputs attribute which is not supported for now
// "Sin" op has support from opset 7, return 6 here for all ops
// "QLinearSigmoid" is a contrib op, OpSet will always be 1
int UnaryOpSupportChecker::GetMinSupportedOpSet(const NodeUnit& node_unit) const {
  if (node_unit.OpType() == "QLinearSigmoid")
    return 1;

  return 6;
}

/* static */ bool UnaryOpSupportChecker::IsQuantizedOpSupported(
    const InitializedTensorSet& initializers, const NodeUnit& node_unit, const OpSupportCheckParams& /* params */) {
  const auto& op_type = node_unit.OpType();
  ORT_ENFORCE(op_type == "QLinearSigmoid");

  // NNAPI requires the scale be 1.f/256 and zero point to be 0
  // See https://android.googlesource.com/platform/frameworks/ml/+/refs/heads/android10-c2f2-release/nn/common/operations/Activation.cpp#180
  if (!op_support_helpers::HasRequiredScaleAndZeroPoint(initializers,
                                                        MakeString("Op [", op_type, "] name [", node_unit.Name(), "]'s output 0 "),
                                                        node_unit.Outputs()[0], node_unit.ModelPath(),
                                                        1.f / 256 /* required_scale */, 0 /* required_zp */)) {
    return false;
  }

  return true;
}

#pragma endregion

#pragma region op_concat

class ConcatOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  bool HasSupportedInputOutputsImpl(
      const InitializedTensorSet& initializers, const NodeUnit& node_unit,
      const OpSupportCheckParams& params) const override;

  bool IsNodeUnitTypeSupported(const NodeUnit& /* node_unit */) const override { return true; }
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

void CreateConcatOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<ConcatOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool ConcatOpSupportChecker::IsQuantizedOp(const NodeUnit& node_unit) const {
  // TODO add support of QLinearConcat
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQConcat;
}

bool ConcatOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
                                               const OpSupportCheckParams& /* params */) const {
  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size > 4 || input_size == 0) {
    LOGS_DEFAULT(VERBOSE) << "Concat only supports up to 1-4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  return true;
}

bool ConcatOpSupportChecker::HasSupportedInputOutputsImpl(
    const InitializedTensorSet& initializers, const NodeUnit& node_unit,
    const OpSupportCheckParams& params) const {
  const auto& op_type = node_unit.OpType();
  const auto& op_name = node_unit.Name();
  const auto input_size = node_unit.Inputs().size();
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
    std::vector<size_t> input_indices(input_size);
    std::iota(input_indices.begin(), input_indices.end(), 0);
    if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, input_indices, params, ArgType::kInput)) {
      return false;
    }

    if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput)) {
      return false;
    }

    // Need to verify all the input and output has the same scale and zp for API 28-
    if (params.android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
      std::vector<float> input_scales(input_size);
      std::vector<int32_t> input_zps(input_size);
      size_t input_idx = 0;

      auto status = GetQuantizationScaleAndZeroPoint(
          initializers, node_unit.Inputs()[input_idx], node_unit.ModelPath(),
          input_scales[input_idx], input_zps[input_idx]);

      if (!status.IsOK()) {
        LOGS_DEFAULT(ERROR) << "Op [" << op_type << "] name [" << op_name
                            << "] GetQuantizationScaleAndZeroPoint for input_scale/zp failed, message: "
                            << status.ErrorMessage();
        return false;
      }

      for (++input_idx; input_idx < input_size; ++input_idx) {
        if (!op_support_helpers::HasRequiredScaleAndZeroPoint(initializers,
                                                              MakeString("Op [", op_type, "] name [", op_name, "] input ", input_idx),
                                                              node_unit.Inputs()[input_idx],
                                                              node_unit.ModelPath(),
                                                              input_scales[0] /* required_scale */,
                                                              input_zps[0] /* required_zp */)) {
          return false;
        }
      }

      // NNAPI (28-) requires the output scale and zp be the same as the input 0
      if (!op_support_helpers::HasRequiredScaleAndZeroPoint(initializers,
                                                            MakeString("Op [", op_type, "] name [", op_name, "]'s output 0"),
                                                            node_unit.Outputs()[0], node_unit.ModelPath(),
                                                            input_scales[0] /* required_scale */,
                                                            input_zps[0] /* required_zp */)) {
        return false;
      }
    }
  }

  return true;
}

#pragma endregion

#pragma region op_squeeze

class SqueezeOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_2;
  }
};

void CreateSqueezeOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<SqueezeOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool SqueezeOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                                const OpSupportCheckParams& /* params */) const {
  const auto& inputs = node_unit.Inputs();
  Shape input_shape;
  if (!GetShape(inputs[0].node_arg, input_shape))
    return false;

  const auto input_rank = input_shape.size();
  if (input_rank > 4 || input_rank == 0) {
    LOGS_DEFAULT(VERBOSE) << "Squeeze only supports 1-4d shape, input is "
                          << input_rank << "d shape";
    return false;
  }

  // Squeeze opset 13 use input 1 as axes, if we have input 1 then it need to be an initializer
  if (node_unit.SinceVersion() > 12 && inputs.size() > 1) {
    const auto& axes_name = inputs[1].node_arg.Name();
    if (!Contains(initializers, axes_name)) {
      LOGS_DEFAULT(VERBOSE) << "Input axes of Squeeze must be known";
      return false;
    }
  }

  return true;
}

#pragma endregion

#pragma region op_quantizelinear

class QuantizeLinearOpSupportChecker : public BaseOpSupportChecker {
 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;
  }

  bool HasSupportedInputOutputsImpl(
      const InitializedTensorSet& initializers, const NodeUnit& node_unit,
      const OpSupportCheckParams& params) const override {
    return op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput);
  }
};

void CreateQuantizeLinearOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<QuantizeLinearOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

#pragma endregion

#pragma region op_dequantizelinear

class DequantizeLinearOpSupportChecker : public BaseOpSupportChecker {
 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_1;
  }

  bool HasSupportedInputOutputsImpl(
      const InitializedTensorSet& initializers, const NodeUnit& node_unit,
      const OpSupportCheckParams& params) const override {
    return op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kInput);
  }
};

void CreateDequantizeLinearOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<DequantizeLinearOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

#pragma endregion

#pragma region op_LRN

class LRNOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_2;
  }
};

void CreateLRNOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<LRNOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool LRNOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
                                            const OpSupportCheckParams& /* params */) const {
  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS_DEFAULT(VERBOSE) << "LRN only support 4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  return true;
}

#pragma endregion

#pragma region op_clip

class ClipOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

void CreateClipOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<ClipOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool ClipOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                             const OpSupportCheckParams& /* params */) const {
  float min, max;
  if (!GetClipMinMax(initializers, node_unit.GetNode(), min, max, logging::LoggingManager::DefaultLogger()))
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

#pragma endregion

#pragma region op_Resize

class ResizeOpSupportChecker : public BaseOpSupportChecker {
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

void CreateResizeOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<ResizeOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool ResizeOpSupportChecker::IsQuantizedOp(const NodeUnit& node_unit) const {
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQResize;
}

bool ResizeOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                               const OpSupportCheckParams& params) const {
  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS_DEFAULT(VERBOSE) << "Resize only support 4d shape, input is "
                          << input_size << "d shape";
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
        LOGS_DEFAULT(VERBOSE) << "Resize bilinear only support half_pixel/align_corners on API level 30+, current API level is "
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
  }

  {  // scales and sizes (if present) must be initializers
    const auto inputs = node_unit.Inputs();
    if (inputs.size() < 3) {
      LOGS_DEFAULT(VERBOSE) << "Input scales or sizes of Resize must be known";
      return false;
    }

    // scales
    if (inputs.size() == 3 && !Contains(initializers, inputs[2].node_arg.Name())) {
      LOGS_DEFAULT(VERBOSE) << "Input scales of Resize must be known";
      return false;
    }

    // sizes
    if (inputs.size() > 3 && !Contains(initializers, inputs[3].node_arg.Name())) {
      LOGS_DEFAULT(VERBOSE) << "Input sizes of Resize must be known";
      return false;
    }

    // We want to check if the scales or sizes are not trying to resize on N/C channels here
    if (inputs.size() == 3) {  // we are using scales
      const auto& scales_tensor = *initializers.at(inputs[2].node_arg.Name());
      std::vector<uint8_t> unpacked_tensor;
      auto status = onnxruntime::utils::UnpackInitializerData(scales_tensor, unpacked_tensor);
      if (!status.IsOK()) {
        LOGS_DEFAULT(ERROR) << "Error while unpacking scales_tensor: " << status.ErrorMessage();
        return false;
      }
      const float* scales_data = reinterpret_cast<const float*>(unpacked_tensor.data());
      float scale_n = scales_data[0];
      float scale_c = op_support_helpers::IsNodeLayoutNHWC(node_unit) ? scales_data[3] : scales_data[1];
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
      std::vector<uint8_t> unpacked_tensor;
      auto status = onnxruntime::utils::UnpackInitializerData(sizes_tensor, unpacked_tensor);
      if (!status.IsOK()) {
        LOGS_DEFAULT(ERROR) << "Error while unpacking sizes_tensor: " << status.ErrorMessage();
        return false;
      }

      int channel_idx = op_support_helpers::IsNodeLayoutNHWC(node_unit) ? 3 : 1;
      const int64_t* sizes_data = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
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
  }

  return true;
}

int32_t ResizeOpSupportChecker::GetMinSupportedNNAPIFeatureLevel(const NodeUnit& node_unit,
                                                                 const OpSupportCheckParams& /* params */) const {
  int32_t input_type;

  // This should not happen, but if it happens make sure this will require an impossible version
  if (!GetType(node_unit.Inputs()[0].node_arg, input_type))
    return std::numeric_limits<int32_t>::max();

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8)
    return ANEURALNETWORKS_FEATURE_LEVEL_3;

  return ANEURALNETWORKS_FEATURE_LEVEL_2;
}

bool ResizeOpSupportChecker::HasSupportedInputOutputsImpl(
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
    if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kInput))
      return false;

    if (!op_support_helpers::IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput))
      return false;
  }

  return true;
}

#pragma endregion



#pragma region op_gather

class GatherOpSupportChecker : public BaseOpSupportChecker {
 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;
  }

  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

void CreateGatherOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<GatherOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool GatherOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                               const OpSupportCheckParams& /* params */) const {
  const auto& inputs = node_unit.Inputs();
  Shape input_shape;

  if (!GetShape(inputs[0].node_arg, input_shape)) {
    return false;
  }

  if (input_shape.size() > 4 || input_shape.empty()) {
    LOGS_DEFAULT(VERBOSE) << "Gather only supports up to 1-4d shape, input is "
                          << input_shape.size() << "d shape";
    return false;
  }

  if (std::any_of(input_shape.cbegin(), input_shape.cend(), [](int32_t i) { return i == 0; })) {
    LOGS_DEFAULT(VERBOSE) << "Gather doesn't support dynamic input shape";
    return false;
  }

  // Here in GatherOpSupportChecker::IsOpSupportedImpl, we removed the restriction that 2nd input "indices" must be an initializer
  // to accommodate the support for some models such as mobileBERT. It doesn't need to be an initializer for int32 as NNAPI Gather
  // uses int32 for indices so the type matches.
  // However, we still require indices of other types to be an initializer as we convert the data to int32 during model building.
  // TODO: We could potentially support non-initializer inputs for the other types if we inserted a cast.
  const auto& indices_name = inputs[1].node_arg.Name();

  int32_t indices_type;
  if (!GetType(node_unit.Inputs()[1].node_arg, indices_type))
    return false;

  if (indices_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    if (!Contains(initializers, indices_name)) {
      LOGS_DEFAULT(VERBOSE) << "Indices of Gather must be known.";
      return false;
    }
  }

  return true;
}

#pragma endregion

#pragma region op_minmax

class MinMaxOpSupportChecker : public BaseOpSupportChecker {
 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;
  }

  // Min/Max opset 5- uses consumed_inputs attribute which is not supported for now
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 6; }

  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

void CreateMinMaxOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  CreateSharedOpSupportCheckerImpl<MinMaxOpSupportChecker>(
      op_type, op_registrations,
      {
          "Min",
          "Max",
      });
}

bool MinMaxOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
                                               const OpSupportCheckParams& /* params */) const {
  // TODO support 2+ inputs for Min/Max op
  if (node_unit.Inputs().size() != 2) {
    LOGS_DEFAULT(VERBOSE) << "[" << node_unit.OpType() << "] only supports 2 inputs, "
                          << "actual input number, " << node_unit.Inputs().size();
    return false;
  }

  return true;
}

#pragma endregion

#pragma region op_elu

class EluOpSupportChecker : public BaseOpSupportChecker {
 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_4;
  }

  // Elu opset 5- uses consumed_inputs attribute which is not supported for now
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 6; }
};

void CreateEluOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<EluOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

#pragma endregion

#pragma region op_slice

class SliceOpSupportChecker : public BaseOpSupportChecker {
 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_2;
  }

  // We only support slice from opset 10
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 10; }

  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

void CreateSliceOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<SliceOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool SliceOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                              const OpSupportCheckParams& /* params */) const {
  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  if (input_shape.size() > 4) {
    LOGS_DEFAULT(VERBOSE) << "Slice only supports 1-4d shape, input is "
                          << input_shape.size() << "d shape";
    return false;
  }

  // TODO, replace with std::find when we switch to c++17
  if (std::any_of(input_shape.cbegin(), input_shape.cend(), [](int32_t i) { return i == 0; })) {
    LOGS_DEFAULT(VERBOSE) << "Slice doesn't support dynamic input shape";
    return false;
  }

  if (!CheckIsInitializer(initializers, node_unit, node_unit.Inputs()[1].node_arg.Name(), "starts")) {
    return false;
  }
  if (!CheckIsInitializer(initializers, node_unit, node_unit.Inputs()[2].node_arg.Name(), "ends")) {
    return false;
  }
  const auto& inputs = node_unit.Inputs();
  if (inputs.size() > 3) {
    if (!CheckIsInitializer(initializers, node_unit, node_unit.Inputs()[3].node_arg.Name(), "axes")) {
      return false;
    }
    if (inputs.size() > 4) {
      if (!CheckIsInitializer(initializers, node_unit, node_unit.Inputs()[4].node_arg.Name(), "steps")) {
        return false;
      }
    }
  }

  return true;
}

#pragma endregion

#pragma region op_pad

class PadOpSupportChecker : public BaseOpSupportChecker {
 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& /* params */) const override {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;  // for ANEURALNETWORKS_PAD_V2
  }

  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override {
    // before Pad-11, inputs `pads` and `constant_value` were attributes
    // only support inputs now
    // Note: Could add support for attributes later.
    return 11;
  }

  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

void CreatePadOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<PadOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

bool PadOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                            const OpSupportCheckParams& /* params */) const {
  const auto& inputs = node_unit.Inputs();

  // only support 1-4d input shape
  // only support input with more than 0 elements
  {
    Shape input_shape;
    if (!GetShape(inputs[0].node_arg, input_shape)) {
      return false;
    }

    if (input_shape.size() > 4 || input_shape.empty()) {
      LOGS_DEFAULT(VERBOSE) << "Pad only supports up to 1-4d shape, input is "
                            << input_shape.size() << "d shape";
      return false;
    }

    if (std::find(input_shape.begin(), input_shape.end(), uint32_t{0}) != input_shape.end()) {
      LOGS_DEFAULT(VERBOSE) << "Pad input with zero elements is not supported";
      return false;
    }
  }

  // only support "constant" mode
  // Note: Could possibly add support for "reflect" later using ANEURALNETWORKS_MIRROR_PAD.
  {
    NodeAttrHelper helper{node_unit};
    const auto mode = helper.Get("mode", "constant");
    if (mode != "constant") {
      LOGS_DEFAULT(VERBOSE) << "Mode is not supported: " << mode;
      return false;
    }
  }

  // only support if `pads` input is known and does not contain negative values
  {
    const auto pads_initializer_it = initializers.find(inputs[1].node_arg.Name());
    if (pads_initializer_it == initializers.end()) {
      LOGS_DEFAULT(VERBOSE) << "pads must be known";
      return false;
    }

    const ONNX_NAMESPACE::TensorProto& pads_initializer = *pads_initializer_it->second;
    std::vector<uint8_t> unpacked_tensor;
    auto status = onnxruntime::utils::UnpackInitializerData(pads_initializer, unpacked_tensor);
    if (!status.IsOK()) {
      LOGS_DEFAULT(ERROR) << "Error while unpacking pads initializer: " << status.ErrorMessage();
      return false;
    }

    int64_t pad_value;
    ORT_ENFORCE(unpacked_tensor.size() % sizeof(pad_value) == 0);
    for (size_t i = 0; i < unpacked_tensor.size(); i += sizeof(pad_value)) {
      memcpy(&pad_value, &unpacked_tensor[i], sizeof(pad_value));
      if (pad_value < 0) {
        LOGS_DEFAULT(VERBOSE) << "Negative pad value is not supported: pads["
                              << i / sizeof(pad_value) << "] = " << pad_value;
        return false;
      }
    }
  }

  // only support if `constant_value` input is known
  // Note: Could add support for non-constant initializer later. Then we need to ensure it is a scalar (with shape []).
  if (inputs.size() > 2) {
    if (!Contains(initializers, inputs[2].node_arg.Name())) {
      LOGS_DEFAULT(VERBOSE) << "constant_value must be known";
      return false;
    }
  }

  return true;
}

#pragma endregion op_pad

/* #pragma region CreateGetOpSupportCheckers

// The reason we use macros to create OpBuilders is for easy exclusion in build if certain op(s) are not used
// such that we can reduce binary size.
// This is for multiple ops share the same OpSupportChecker, we only need create one for all of them
#define NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER(OP_TYPE, SUPPORT_CHECKER_NAME) \
  SUPPORT_CHECKER_NAME::CreateSharedOpSupportChecker(OP_TYPE, op_registrations);

// This is for ops with dedicated OpSupportChecker
#define NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER(OP_TYPE, SUPPORT_CHECKER_NAME)                                 \
  do {                                                                                                        \
    op_registrations.support_checkers.push_back(std::make_unique<SUPPORT_CHECKER_NAME>());                    \
    op_registrations.op_support_checker_map.emplace(OP_TYPE, op_registrations.support_checkers.back().get()); \
  } while (0)

static OpSupportCheckerRegistrations CreateOpSupportCheckerRegistrations() {
  OpSupportCheckerRegistrations op_registrations;

  // Support checkers handle a single op
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("BatchNormalization", BatchNormalizationOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Cast", CastOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Clip", ClipOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Concat", ConcatOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("DepthToSpace", DepthToSpaceOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("DequantizeLinear", DequantizeLinearOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Elu", EluOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Flatten", FlattenOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Gather", GatherOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("LRN", LRNOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Pad", PadOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("QuantizeLinear", QuantizeLinearOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Reshape", ReshapeOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Resize", ResizeOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Slice", SliceOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Softmax", SoftMaxOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Squeeze", SqueezeOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Transpose", TransposeOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Unsqueeze", UnsqueezeOpSupportChecker);

  // Identity is always supported, we use BaseOpSupportChecker as default
  NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Identity", BaseOpSupportChecker);

  // Relu is always supported, we use BaseOpSupportChecker as default
  NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Relu", BaseOpSupportChecker);

  // Support Checkers shared among similar ops
  {
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Add", BinaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Div", BinaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Mul", BinaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Pow", BinaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("PRelu", BinaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("QLinearAdd", BinaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("QLinearMul", BinaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Sub", BinaryOpSupportChecker);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("AveragePool", PoolOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("GlobalAveragePool", PoolOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("GlobalMaxPool", PoolOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("MaxPool", PoolOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("QLinearAveragePool", PoolOpSupportChecker);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Conv", ConvOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("QLinearConv", ConvOpSupportChecker);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Gemm", GemmOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("MatMul", GemmOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("QLinearMatMul", GemmOpSupportChecker);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Abs", UnaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Exp", UnaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Floor", UnaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Log", UnaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Neg", UnaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("QLinearSigmoid", UnaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Sigmoid", UnaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Sin", UnaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Sqrt", UnaryOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Tanh", UnaryOpSupportChecker);
  }

  {
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Max", MinMaxOpSupportChecker);
    NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER("Min", MinMaxOpSupportChecker);
  }

  return op_registrations;
}

const std::unordered_map<std::string, const IOpSupportChecker*>& GetOpSupportCheckers() {
  static const OpSupportCheckerRegistrations op_registrations = CreateOpSupportCheckerRegistrations();
  return op_registrations.op_support_checker_map;
}

#pragma endregion */

}  // namespace nnapi
}  // namespace onnxruntime
