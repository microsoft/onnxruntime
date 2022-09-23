// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_helpers.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_checker_factory.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_support_checker.h"

namespace onnxruntime {
namespace nnapi {

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

}  // namespace nnapi
}  // namespace onnxruntime
