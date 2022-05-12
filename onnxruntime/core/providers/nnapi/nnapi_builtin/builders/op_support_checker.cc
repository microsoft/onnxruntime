// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "op_support_checker.h"

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph.h"
#include "core/providers/common.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/providers/shared/utils/utils.h"
#include "helper.h"

namespace onnxruntime {
namespace nnapi {

#pragma region helpers

struct OpSupportCheckerRegistrations {
  std::vector<std::unique_ptr<IOpSupportChecker>> support_checkers;
  std::unordered_map<std::string, const IOpSupportChecker*> op_support_checker_map;
};

template <class T>
void CreateSharedOpSupportCheckerImpl(const std::string& op_type,
                                      OpSupportCheckerRegistrations& op_registrations,
                                      const std::vector<std::string>& op_types) {
  // The shared OpSupportChecker is already in the OpSupportCheckerRegistrations
  if (op_registrations.op_support_checker_map.find(op_type) != op_registrations.op_support_checker_map.cend())
    return;

  op_registrations.support_checkers.push_back(std::make_unique<T>());
  for (const auto& op : op_types) {
    op_registrations.op_support_checker_map.emplace(op, op_registrations.support_checkers.back().get());
  }
}

static bool HasExternalInitializer(const InitializedTensorSet& initializers, const NodeUnit& node_unit) {
  const auto is_ext_initializer =
      [&](const NodeArg& node_arg) {
        const auto& input_name(node_arg.Name());
        if (!Contains(initializers, input_name))
          return false;

        const auto& tensor = *initializers.at(input_name);
        if (tensor.has_data_location() &&
            tensor.data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
          LOGS_DEFAULT(VERBOSE) << "Initializer [" << input_name
                                << "] with external data location are not currently supported";
          return true;
        }

        return false;
      };

  const auto& inputs = node_unit.Inputs();
  for (const auto& input : inputs) {
    if (is_ext_initializer(input.node_arg))
      return true;

    if (!input.quant_param)
      continue;

    if (is_ext_initializer(input.quant_param->scale))
      return true;

    if (input.quant_param->zero_point && is_ext_initializer(*input.quant_param->zero_point))
      return true;
  }

  return false;
}

inline bool IsNodeLayoutNHWC(const NodeUnit& node_unit) {
  return node_unit.Domain() == kMSInternalNHWCDomain;
}

static bool IsQuantizationScaleSupported(const InitializedTensorSet& initializers,
                                         const NodeUnitIODef& io_def,
                                         const OpSupportCheckParams& params,
                                         const std::string& op_type,
                                         bool is_quant_matmul,
                                         bool is_conv_matmul_u8s8_weight) {
  const auto scale_name = io_def.quant_param->scale.Name();
  auto it = initializers.find(scale_name);
  if (it == initializers.cend()) {
    LOGS_DEFAULT(VERBOSE) << "The scale of " << op_type << " must be an initializer tensor";
    return false;
  }

  const auto& scale_tensor = *it->second;
  int64_t scales_dim = scale_tensor.dims().empty() ? 1 : scale_tensor.dims()[0];
  if (!is_conv_matmul_u8s8_weight) {
    if (scales_dim != 1) {
      LOGS_DEFAULT(VERBOSE) << op_type << " does not support per-channel quantization, "
                            << " for now, only u8s8 QlinearConv supports per-channel quantization on API 29+";
      return false;
    }
  } else if (scales_dim != 1) {
    // For u8s8 Qlinear[Conv/MatMul], we support
    // 1. Per-tensor, the weight will be transformed to uint8 later
    // 2. Per-channel, only from Android API level 29
    if (is_quant_matmul) {
      LOGS_DEFAULT(VERBOSE) << "QLinearMatMul does not support per-channel quantization";
      return false;
    }

    if (params.android_feature_level < ANEURALNETWORKS_FEATURE_LEVEL_3) {
      LOGS_DEFAULT(VERBOSE) << op_type << " only supports per-channel quantization on Android API 29+, "
                            << "system NNAPI feature level: " << params.android_feature_level;
      return false;
    }

    Shape weight_shape;
    if (!GetShape(io_def.node_arg, weight_shape))
      return false;

    if (weight_shape[0] != scales_dim) {
      LOGS_DEFAULT(VERBOSE) << op_type << " mismatch int8 per-channel quantization weight,"
                            << " weight dimension[0] " << weight_shape[0]
                            << " scale dimension " << scales_dim;
      return false;
    }
  }

  return true;
}

static bool IsQuantizationZeroPointSupported(const InitializedTensorSet& initializers,
                                             const NodeUnitIODef& io_def,
                                             const std::string& op_type,
                                             const Path& model_path,
                                             bool is_quant_matmul,
                                             bool is_conv_matmul_u8s8_weight) {
  // zero point is optional here
  if (!io_def.quant_param->zero_point)
    return true;

  const auto& zero_point_name = io_def.quant_param->zero_point->Name();
  if (!Contains(initializers, zero_point_name)) {
    LOGS_DEFAULT(VERBOSE) << "The zero point of " << op_type << " must be an initializer tensor";
    return false;
  }

  const auto& zero_tensor = *initializers.at(zero_point_name);
  int64_t zero_dim = zero_tensor.dims().empty() ? 1 : zero_tensor.dims()[0];

  if (!is_conv_matmul_u8s8_weight) {
    if (zero_dim != 1) {
      LOGS_DEFAULT(VERBOSE) << op_type << " does not support per-channel quantization, "
                            << " for now, only u8s8 QlinearConv supports per-channel quantization on API 29+";
      return false;
    }
  } else {
    // For u8s8 Qlinear[Conv/MatMul], we support
    // 1. Per-tensor, the weight will be transformed to uint8 later
    // 2. Per-channel, only from Android API level 29
    if (zero_tensor.data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8) {
      LOGS_DEFAULT(VERBOSE) << "u8s8 Qlinear[Conv/MatMul] only supports int8 zero point for weight, "
                            << "actual zero point type: [" << zero_tensor.data_type() << "]";
      return false;
    }

    if (zero_dim != 1) {
      if (is_quant_matmul) {
        LOGS_DEFAULT(VERBOSE) << "QLinearMatMul does not support per-channel quantization";
        return false;
      }
    }

    // For onnx, u8s8 QlinearConv, the weight zero point can be a scalar,
    // or a tensor with same channel as weight, for NNAPI we only support it be
    // 0 (scalar) or all 0 (tensor), NNAPI will assume the zero point for per-channel
    // quantization is 0 there is no input for it
    Shape weight_shape;
    if (!GetShape(io_def.node_arg, weight_shape))
      return false;

    if (weight_shape[0] != zero_dim && zero_dim != 1) {
      LOGS_DEFAULT(VERBOSE) << op_type << " mismatch int8 per-channel quantization weight,"
                            << " weight dimension[0] " << weight_shape[0]
                            << " zero point dimension " << zero_dim;
      return false;
    }

    std::vector<uint8_t> unpacked_tensor;
    auto status = onnxruntime::utils::UnpackInitializerData(zero_tensor, model_path, unpacked_tensor);
    if (!status.IsOK()) {
      LOGS_DEFAULT(ERROR) << "Qlinear[Conv/MatMul] error when unpack zero tensor: " << zero_point_name
                          << ", error msg: " << status.ErrorMessage();
      return false;
    }

    // Verify all onnx weight zero point(s) are 0(s)
    const int8_t* zero_points = reinterpret_cast<const int8_t*>(unpacked_tensor.data());
    for (size_t i = 0; i < unpacked_tensor.size(); i++) {
      if (zero_points[i] != 0) {
        LOGS_DEFAULT(VERBOSE) << "u8s8 Qlinear[Conv/MatMul]  only support 0 as zero point, "
                              << "zero_points[" << i << "] has value: " << zero_points[i];
        return false;
      }
    }
  }

  return true;
}

// Check if the given quantized input(s) or output(s) is supported
static bool IsQuantizedIOSupported(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                   const std::vector<size_t>& indices, const OpSupportCheckParams& params, ArgType arg_type) {
  const auto& op_type = node_unit.OpType();
  auto quant_op_type = GetQuantizedOpType(node_unit);

  ORT_ENFORCE(quant_op_type != QuantizedOpType::Unknown, "[", op_type, "] is not a quantized op");

  const bool is_input = arg_type == ArgType::kInput;
  const bool is_quant_conv = IsQuantizedConv(quant_op_type);
  const bool is_quant_matmul = (quant_op_type == QuantizedOpType::QLinearMatMul) || (quant_op_type == QuantizedOpType::QDQMatMul);
  const bool is_quant_gemm = (quant_op_type == QuantizedOpType::QDQGemm);
  const bool is_quant_matmul_or_gemm = is_quant_matmul || is_quant_gemm;
  const auto& io_defs = is_input ? node_unit.Inputs() : node_unit.Outputs();

  for (const auto idx : indices) {
    if (idx >= io_defs.size()) {
      LOGS_DEFAULT(VERBOSE) << (is_input ? "Input" : "Output") << " index,  " << idx
                            << " >= size, " << io_defs.size()
                            << " of NodeUnit: " << node_unit.Name();
      return false;
    }

    const auto& io_def = io_defs[idx];
    ORT_ENFORCE(io_def.quant_param.has_value(), "Input index,  ", idx, " has no quant_param");

    // If this op is Qlinear[Conv/MatMul], we want to check u8s8 support for weight tensor (or B tensor for QlinearMatMul)
    const bool is_conv_matmul_weight = is_input && (is_quant_conv || is_quant_matmul_or_gemm) && idx == 1;
    bool is_conv_matmul_u8s8_weight = false;

    if (is_conv_matmul_weight) {
      int32_t weight_type;
      if (!GetType(io_def.node_arg, weight_type))
        return false;
      is_conv_matmul_u8s8_weight = weight_type == ONNX_NAMESPACE::TensorProto_DataType_INT8;
    }

    int32_t input_type;
    if (!GetType(io_def.node_arg, input_type))
      return false;

    // We only support u8 for most of the inputs and all outputs, with the exception for Quantized MatMul and Conv,
    // which allows s8 weight (u8s8)
    // TODO, add support of s8s8
    if (input_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8 &&
        !(input_type == ONNX_NAMESPACE::TensorProto_DataType_INT8 && is_conv_matmul_u8s8_weight)) {
      LOGS_DEFAULT(VERBOSE) << op_type << "NodeUnit [" << node_unit.Name()
                            << "], type [" << op_type << "]'s "
                            << (is_input ? "Input" : "Output") << " index  [" << idx
                            << "] has unsupported type [" << input_type << "]";
      return false;
    }

    // Check scale and zero point
    if (!IsQuantizationScaleSupported(initializers, io_def, params, op_type,
                                      is_quant_matmul, is_conv_matmul_u8s8_weight)) {
      return false;
    }

    if (!IsQuantizationZeroPointSupported(initializers, io_def, op_type, node_unit.ModelPath(),
                                          is_quant_matmul, is_conv_matmul_u8s8_weight)) {
      return false;
    }
  }

  return true;
}

// Some Quantized NNAPI operations have required output scale and zero point
// e.g. Softmax (uint8) requires output scale be 1.f/256 and zp be 0
// This helper function checks if the given io_def has required scale and zp
static bool HasRequiredScaleAndZeroPoint(const InitializedTensorSet& initializers,
                                         const std::string& op_desc,
                                         const NodeUnitIODef& io_def,
                                         const Path& path,
                                         float required_scale, int32_t required_zp) {
  float scale = 0.0f;
  int32_t zp = 0;
  auto status = GetQuantizationScaleAndZeroPoint(initializers, io_def, path,
                                                 scale, zp);
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << op_desc
                        << " GetQuantizationScaleAndZeroPoint failed, message: "
                        << status.ErrorMessage();
    return false;
  }

  if (scale != required_scale) {
    LOGS_DEFAULT(VERBOSE) << op_desc
                          << " scale can only be [" << required_scale
                          << "], actual scale: " << scale;
    return false;
  }

  if (zp != required_zp) {
    LOGS_DEFAULT(VERBOSE) << op_desc
                          << "] zero point can only be [" << required_zp
                          << "], actual zero point: " << scale;
    return false;
  }

  return true;
}

#pragma endregion helpers

#pragma region op_base

class BaseOpSupportChecker : public IOpSupportChecker {
 public:
  virtual ~BaseOpSupportChecker() = default;
  bool IsOpSupported(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                     const OpSupportCheckParams& params) const override;

  // This is for ops which are by default supported and do not have their own impl of OpSupportChecker
  // for those ops (Relu, Identity) we use BaseOpSupportChecker
  static void CreateSharedOpSupportChecker(
      const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);

 protected:
  virtual bool IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const NodeUnit& /* node_unit */,
                                 const OpSupportCheckParams& /* params */) const {
    return true;
  }

  virtual bool IsQuantizedOp(const NodeUnit& /* node_unit */) const { return false; }

  virtual int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                                   const OpSupportCheckParams& /* params */) const {
    // ANEURALNETWORKS_FEATURE_LEVEL_1 is the baseline version of NNAPI,
    // There is no NNAPI support for Android API level 26-
    return ANEURALNETWORKS_FEATURE_LEVEL_1;
  }

  virtual bool HasSupportedInputOutputsImpl(
      const InitializedTensorSet& initializers, const NodeUnit& node_unit,
      const OpSupportCheckParams& params) const;

  virtual int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const { return 1; }
  virtual int GetMaxSupportedOpSet(const NodeUnit& /* node_unit */) const { return 15; }

  // Check if this node_unit's type is supported
  // SingleNode type NodeUnit is supported
  // QDQGroup type NodeUnit is by default unsupported, and this can be individually overwritten by inherited classes
  virtual bool IsNodeUnitTypeSupported(const NodeUnit& node_unit) const;

 private:
  bool HasSupportedOpSet(const NodeUnit& node_unit) const;
  bool HasSupportedInputOutputs(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                const OpSupportCheckParams& params) const;
};

/* static */ void BaseOpSupportChecker::CreateSharedOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  CreateSharedOpSupportCheckerImpl<BaseOpSupportChecker>(
      op_type, op_registrations,
      {
          "Relu",
          "Identity",
      });
}

bool BaseOpSupportChecker::IsOpSupported(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                         const OpSupportCheckParams& params) const {
  int32_t required_feature_level = GetMinSupportedNNAPIFeatureLevel(node_unit, params);
  if (required_feature_level > params.android_feature_level) {
    LOGS_DEFAULT(VERBOSE) << "Current Android API level [" << params.android_feature_level
                          << "], Operator [" << node_unit.OpType()
                          << "] is only supported on API >" << required_feature_level;
    return false;
  }

  if (!IsNodeUnitTypeSupported(node_unit))
    return false;

  if (!HasSupportedInputOutputs(initializers, node_unit, params))
    return false;

  // We do not support external initializers for now
  if (HasExternalInitializer(initializers, node_unit))
    return false;

  if (!HasSupportedOpSet(node_unit))
    return false;

  return IsOpSupportedImpl(initializers, node_unit, params);
}

bool BaseOpSupportChecker::HasSupportedInputOutputs(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                                    const OpSupportCheckParams& params) const {
  // We do not support unknown(null) input shape
  auto has_supported_shape = [](const NodeArg& node_arg, const std::string& name, const std::string op_type) {
    const auto* shape_proto = node_arg.Shape();
    if (!shape_proto) {
      LOGS_DEFAULT(VERBOSE) << "Node [" << name << "] type [" << op_type
                            << "] Input [" << node_arg.Name() << "] has no shape";
      return false;
    }

    // We do not support dynamic shape input for now
    for (const auto& dim : shape_proto->dim()) {
      if (!dim.has_dim_value()) {
        LOGS_DEFAULT(VERBOSE) << "Dynamic shape is not supported for now, for input:" << node_arg.Name();
        return false;
      }
    }
    return true;
  };

  for (const auto& input : node_unit.Inputs()) {
    if (!has_supported_shape(input.node_arg, node_unit.Name(), node_unit.OpType()))
      return false;

    if (input.quant_param.has_value()) {
      if (!has_supported_shape(input.quant_param->scale, node_unit.Name(), node_unit.OpType()))
        return false;

      // zero point is optional
      if (input.quant_param->zero_point &&
          !has_supported_shape(*input.quant_param->zero_point, node_unit.Name(), node_unit.OpType()))
        return false;
    }
  }
  return HasSupportedInputOutputsImpl(initializers, node_unit, params);
}

bool BaseOpSupportChecker::HasSupportedInputOutputsImpl(
    const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
    const OpSupportCheckParams& /* params */) const {
  // We only check the type of input 0 by default
  // specific op builder can override this
  const auto& input = node_unit.Inputs()[0].node_arg;

  int32_t input_type;
  if (!GetType(input, input_type))
    return false;

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    LOGS_DEFAULT(VERBOSE) << "[" << node_unit.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

bool BaseOpSupportChecker::HasSupportedOpSet(const NodeUnit& node_unit) const {
  auto since_version = node_unit.SinceVersion();
  if (since_version < GetMinSupportedOpSet(node_unit) || since_version > GetMaxSupportedOpSet(node_unit)) {
    LOGS_DEFAULT(VERBOSE) << node_unit.OpType() << " opset [" << since_version
                          << "] is only supported for opset ["
                          << GetMinSupportedOpSet(node_unit) << ", "
                          << GetMaxSupportedOpSet(node_unit) << "]";
    return false;
  }

  return true;
}

bool BaseOpSupportChecker::IsNodeUnitTypeSupported(const NodeUnit& node_unit) const {
  if (node_unit.UnitType() == NodeUnit::Type::QDQGroup) {
    LOGS_DEFAULT(VERBOSE) << "QDQ NodeUnit [" << node_unit.OpType()
                          << "] is not supported for now";

    return false;
  }

  return true;
}

#pragma endregion op_base

#pragma region op_binary

class BinaryOpSupportChecker : public BaseOpSupportChecker {
 public:
  static void CreateSharedOpSupportChecker(
      const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);

 private:
  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& node_unit,
                                           const OpSupportCheckParams& params) const override;
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
  bool HasSupportedInputOutputsImpl(
      const InitializedTensorSet& initializers, const NodeUnit& node_unit,
      const OpSupportCheckParams& params) const override;
  int GetMinSupportedOpSet(const NodeUnit& node_unit) const override;

  bool IsNodeUnitTypeSupported(const NodeUnit& node_unit) const override;
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

/* static */ void BinaryOpSupportChecker::CreateSharedOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  CreateSharedOpSupportCheckerImpl<BinaryOpSupportChecker>(
      op_type, op_registrations,
      {
          "Add",
          "Sub",
          "Mul",
          "Div",
          "QLinearAdd",
          "QLinearMul",
          "Pow",
          "PRelu",
      });
}

bool BinaryOpSupportChecker::IsNodeUnitTypeSupported(const NodeUnit& node_unit) const {
  if (node_unit.UnitType() == NodeUnit::Type::QDQGroup) {
    const auto quant_type = GetQuantizedOpType(node_unit);
    return quant_type == QuantizedOpType::QDQAdd ||
           quant_type == QuantizedOpType::QDQMul;
  }

  return true;
}

bool BinaryOpSupportChecker::IsQuantizedOp(const NodeUnit& node_unit) const {
  const auto quant_type = GetQuantizedOpType(node_unit);
  return quant_type == QuantizedOpType::QLinearAdd ||
         quant_type == QuantizedOpType::QLinearMul ||
         quant_type == QuantizedOpType::QDQAdd ||
         quant_type == QuantizedOpType::QDQMul;
}

int32_t BinaryOpSupportChecker::GetMinSupportedNNAPIFeatureLevel(
    const NodeUnit& node_unit, const OpSupportCheckParams& /* params */) const {
  const auto& op(node_unit.OpType());
  if (op == "Sub" || op == "Div") {
    return ANEURALNETWORKS_FEATURE_LEVEL_2;
  }

  if (op == "Pow" || op == "PRelu") {
    return ANEURALNETWORKS_FEATURE_LEVEL_3;
  }

  return ANEURALNETWORKS_FEATURE_LEVEL_1;
}

int BinaryOpSupportChecker::GetMinSupportedOpSet(const NodeUnit& node_unit) const {
  const auto& op(node_unit.OpType());

  // Add/Sub/Mul/Div/Pow opset 6- has broadcast attributes we do not support now
  if (op == "Add" || op == "Sub" || op == "Mul" || op == "Div" || op == "Pow") {
    return 7;
  }

  return 1;
}

bool BinaryOpSupportChecker::HasSupportedInputOutputsImpl(
    const InitializedTensorSet& initializers, const NodeUnit& node_unit,
    const OpSupportCheckParams& params) const {
  bool is_quantized_op = IsQuantizedOp(node_unit);
  bool is_pow = node_unit.OpType() == "Pow";
  if (!is_quantized_op && !is_pow)
    return BaseOpSupportChecker::HasSupportedInputOutputsImpl(initializers, node_unit, params);

  if (is_quantized_op) {
    // QLinearAdd/QDQAdd/QLinearMul/QDQMul
    if (!HasValidBinaryOpQuantizedInputTypes(node_unit))
      return false;

    if (!IsQuantizedIOSupported(initializers, node_unit, {0, 1}, params, ArgType::kInput))
      return false;

    if (!IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput))
      return false;
  }

  // Pow we only support both input as fp32 now
  if (is_pow) {
    int32_t input_type_1;
    if (!GetType(node_unit.Inputs()[0].node_arg, input_type_1))
      return false;

    int32_t input_type_2;
    if (!GetType(node_unit.Inputs()[1].node_arg, input_type_2))
      return false;

    if (input_type_1 != ONNX_NAMESPACE::TensorProto_DataType_FLOAT || input_type_1 != input_type_2) {
      LOGS_DEFAULT(VERBOSE) << "Pow only supports fp32 inputs, actual input type"
                            << ", Input type 1: " << input_type_1
                            << ", Input type 2: " << input_type_2;
      return false;
    }
  }

  return true;
}

bool BinaryOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
                                               const OpSupportCheckParams& /* params */) const {
  const auto& op_type(node_unit.OpType());
  const auto& inputs = node_unit.Inputs();
  Shape input1_shape, input2_shape;
  if (!GetShape(inputs[0].node_arg, input1_shape) ||
      !GetShape(inputs[1].node_arg, input2_shape))
    return false;

  const auto input1_size = input1_shape.size();
  const auto input2_size = input2_shape.size();
  if (input1_size > 4 || input2_size > 4) {
    LOGS_DEFAULT(VERBOSE) << op_type << " only support up to 4d shape, input1 is "
                          << input1_size << "d shape, input 2 is "
                          << input2_size << "d shape";
    return false;
  }

  return true;
}

#pragma endregion

#pragma region op_transpose

class TransposeOpSupportChecker : public BaseOpSupportChecker {
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

bool TransposeOpSupportChecker::IsQuantizedOp(const NodeUnit& node_unit) const {
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQTranspose;
}

bool TransposeOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
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

bool TransposeOpSupportChecker::HasSupportedInputOutputsImpl(
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

#pragma endregion

#pragma region op_reshape

class ReshapeOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  // Reshape opset 4- uses attributes for new shape which we do not support for now
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 5; }
  bool HasSupportedInputOutputsImpl(
      const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
      const OpSupportCheckParams& /* params */) const override;
  bool IsNodeUnitTypeSupported(const NodeUnit& /* node_unit */) const override { return true; }
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

bool ReshapeOpSupportChecker::IsQuantizedOp(const NodeUnit& node_unit) const {
  return GetQuantizedOpType(node_unit) == QuantizedOpType::QDQReshape;
}

bool ReshapeOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                                                const OpSupportCheckParams& /* params */) const {
  const auto& inputs = node_unit.Inputs();
  const auto& perm_name = inputs[1].node_arg.Name();
  if (!Contains(initializers, perm_name)) {
    LOGS_DEFAULT(VERBOSE) << "New shape of reshape must be known";
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

  const auto& perm_tensor = *initializers.at(perm_name);
  std::vector<uint8_t> unpacked_tensor;
  auto status = onnxruntime::utils::UnpackInitializerData(perm_tensor, unpacked_tensor);
  if (!status.IsOK()) {
    LOGS_DEFAULT(ERROR) << "Error while unpacking perm_tensor: " << status.ErrorMessage();
    return false;
  }
  const int64_t* raw_perm = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
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

bool ReshapeOpSupportChecker::HasSupportedInputOutputsImpl(
    const InitializedTensorSet& initializers, const NodeUnit& node_unit,
    const OpSupportCheckParams& params) const {
  if (!IsQuantizedOp(node_unit)) {
    return BaseOpSupportChecker::HasSupportedInputOutputsImpl(initializers, node_unit, params);
  }

  if (!IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kInput)) {
    return false;
  }

  if (!IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput)) {
    return false;
  }

  return true;
}

#pragma endregion

#pragma region op_batchnormalization

class BatchNormalizationOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  // BatchNormalization opset 6- has unsupported attributes
  int GetMinSupportedOpSet(const NodeUnit& /* node_unit */) const override { return 7; }
};

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
 public:
  static void CreateSharedOpSupportChecker(
      const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);

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

/* static */ void PoolOpSupportChecker::CreateSharedOpSupportChecker(
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
    if (!IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kInput))
      return false;

    if (!IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput))
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

#pragma region op_conv

class ConvOpSupportChecker : public BaseOpSupportChecker {
 public:
  static void CreateSharedOpSupportChecker(
      const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);

 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;

  int32_t GetMinSupportedNNAPIFeatureLevel(const NodeUnit& /* node_unit */,
                                           const OpSupportCheckParams& params) const override {
    return params.use_nchw ? ANEURALNETWORKS_FEATURE_LEVEL_3 : ANEURALNETWORKS_FEATURE_LEVEL_2;
  }

  bool HasSupportedInputOutputsImpl(
      const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
      const OpSupportCheckParams& /* params */) const override;
  bool IsNodeUnitTypeSupported(const NodeUnit& /* node_unit */) const override { return true; }
  bool IsQuantizedOp(const NodeUnit& node_unit) const override;
};

/* static */ void ConvOpSupportChecker::CreateSharedOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  CreateSharedOpSupportCheckerImpl<ConvOpSupportChecker>(
      op_type, op_registrations,
      {
          "Conv",
          "QLinearConv",
      });
}

bool ConvOpSupportChecker::IsQuantizedOp(const NodeUnit& node_unit) const {
  return IsQuantizedConv(GetQuantizedOpType(node_unit));
}

bool ConvOpSupportChecker::HasSupportedInputOutputsImpl(
    const InitializedTensorSet& initializers, const NodeUnit& node_unit,
    const OpSupportCheckParams& params) const {
  if (!IsQuantizedOp(node_unit))
    return BaseOpSupportChecker::HasSupportedInputOutputsImpl(initializers, node_unit, params);

  // QLinearConv only supports input of uint8 for now
  if (!HasValidBinaryOpQuantizedInputTypes(node_unit))
    return false;

  if (!IsQuantizedIOSupported(initializers, node_unit, {0, 1}, params, ArgType::kInput))
    return false;

  if (!IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput))
    return false;

  return true;
}

bool ConvOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
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
  if (Contains(initializers, weight_name)) {
    const auto& tensor = *initializers.at(weight_name);
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
    LOGS_DEFAULT(VERBOSE) << "The weight of convolution must be known";
    return false;
  }

  if (is_quant_conv) {
    if (inputs.size() > 2 && !Contains(initializers, inputs[2].node_arg.Name())) {
      LOGS_DEFAULT(VERBOSE) << "Bias of QLinearConv must be known";
      return false;
    }
  }

  return true;
}

#pragma endregion

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

  if (!IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kInput)) {
    return false;
  }

  if (!IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput)) {
    return false;
  }

  // NNAPI requires the scale be 1.f/256 and zero point to be 0
  if (!HasRequiredScaleAndZeroPoint(initializers,
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
 public:
  static void CreateSharedOpSupportChecker(
      const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);

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

  if (!IsQuantizedIOSupported(initializers, node_unit, {0, 1}, params, ArgType::kInput))
    return false;

  if (!IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput))
    return false;

  return true;
}

/* static */ void GemmOpSupportChecker::CreateSharedOpSupportChecker(
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
 public:
  static void CreateSharedOpSupportChecker(
      const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);

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

/* static */ void UnaryOpSupportChecker::CreateSharedOpSupportChecker(
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

  if (!IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kInput))
    return false;

  if (!IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput))
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
  if (!HasRequiredScaleAndZeroPoint(initializers,
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
    if (!IsQuantizedIOSupported(initializers, node_unit, input_indices, params, ArgType::kInput)) {
      return false;
    }

    if (!IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput)) {
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
        if (!HasRequiredScaleAndZeroPoint(initializers,
                                          MakeString("Op [", op_type, "] name [", op_name, "] input ", input_idx),
                                          node_unit.Inputs()[input_idx],
                                          node_unit.ModelPath(),
                                          input_scales[0] /* required_scale */,
                                          input_zps[0] /* required_zp */)) {
          return false;
        }
      }

      // NNAPI (28-) requires the output scale and zp be the same as the input 0
      if (!HasRequiredScaleAndZeroPoint(initializers,
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
    return IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput);
  }
};

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
    return IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kInput);
  }
};

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
      float scale_c = IsNodeLayoutNHWC(node_unit) ? scales_data[3] : scales_data[1];
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

      int channel_idx = IsNodeLayoutNHWC(node_unit) ? 3 : 1;
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
    if (!IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kInput))
      return false;

    if (!IsQuantizedIOSupported(initializers, node_unit, {0}, params, ArgType::kOutput))
      return false;
  }

  return true;
}

#pragma endregion

#pragma region op_flatten

class FlattenOpSupportChecker : public BaseOpSupportChecker {
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                         const OpSupportCheckParams& params) const override;
};

bool FlattenOpSupportChecker::IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const NodeUnit& node_unit,
                                                const OpSupportCheckParams& /* params */) const {
  Shape input_shape;
  if (!GetShape(node_unit.Inputs()[0].node_arg, input_shape))
    return false;

  if (input_shape.size() > 4 || input_shape.empty()) {
    LOGS_DEFAULT(VERBOSE) << "Flatten only supports up to 1-4d shape, input is "
                          << input_shape.size() << "d shape";
    return false;
  }

  int32_t dim_1 = 1;
  int32_t dim_2 = 1;
  GetFlattenOutputShape(node_unit, input_shape, dim_1, dim_2);

  if (dim_1 == 0 && dim_2 == 0) {
    LOGS_DEFAULT(VERBOSE) << "The dynamic input shape " << Shape2String(input_shape)
                          << " is not supported";
    return false;
  }

  return true;
}

#pragma endregion

#pragma region op_minmax

class MinMaxOpSupportChecker : public BaseOpSupportChecker {
 public:
  static void CreateSharedOpSupportChecker(
      const std::string& op_type, OpSupportCheckerRegistrations& op_registrations);

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

/* static */ void MinMaxOpSupportChecker::CreateSharedOpSupportChecker(
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

#pragma region CreateGetOpSupportCheckers

// The reason we use macros to create OpBuilders is for easy exclusion in build if certain op(s) are not used
// such that we can reduce binary size.
// This is for multiple ops share the same OpSupportChecker, we only need create one for all of them
#define NNAPI_EP_ADD_SHARED_OP_SUPPORT_CHECKER(OP_TYPE, SUPPORT_CHECKER_NAME) \
  SUPPORT_CHECKER_NAME::CreateSharedOpSupportChecker(OP_TYPE, op_registrations);

// This is for ops with dedicated OpSupportChecker
#define NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER(OP_TYPE, SUPPORT_CHECKER_NAME)                                 \
  {                                                                                                           \
    op_registrations.support_checkers.push_back(std::make_unique<SUPPORT_CHECKER_NAME>());                    \
    op_registrations.op_support_checker_map.emplace(OP_TYPE, op_registrations.support_checkers.back().get()); \
  }

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
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("LRN", LRNOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("QuantizeLinear", QuantizeLinearOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Reshape", ReshapeOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Resize", ResizeOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Slice", SliceOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Softmax", SoftMaxOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Squeeze", SqueezeOpSupportChecker);
  NNAPI_EP_ADD_SINGLE_OP_SUPPORT_CHECKER("Transpose", TransposeOpSupportChecker);

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

#pragma endregion

}  // namespace nnapi
}  // namespace onnxruntime
