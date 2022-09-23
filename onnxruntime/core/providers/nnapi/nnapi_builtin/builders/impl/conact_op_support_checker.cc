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

void CreateConcatOpSupportChecker(
    const std::string& op_type, OpSupportCheckerRegistrations& op_registrations) {
  op_registrations.support_checkers.push_back(std::make_unique<ConcatOpSupportChecker>());
  op_registrations.op_support_checker_map.emplace(op_type, op_registrations.support_checkers.back().get());
}

}  // namespace nnapi
}  // namespace onnxruntime
