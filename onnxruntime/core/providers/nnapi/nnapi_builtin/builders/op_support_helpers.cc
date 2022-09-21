// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "op_support_helpers.h"

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/shared/utils/utils.h"

namespace onnxruntime::nnapi::op_support_helpers {

bool IsQuantizationScaleSupported(const InitializedTensorSet& initializers,
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

bool IsQuantizationZeroPointSupported(const InitializedTensorSet& initializers,
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

bool IsQuantizedIOSupported(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
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

bool HasRequiredScaleAndZeroPoint(const InitializedTensorSet& initializers,
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

}  // namespace onnxruntime::nnapi::op_support_helpers
