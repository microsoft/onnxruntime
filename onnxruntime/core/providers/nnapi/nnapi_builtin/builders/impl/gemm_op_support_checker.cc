// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_helpers.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_checker_factory.h"
#include "core/providers/nnapi/nnapi_builtin/builders/impl/base_op_support_checker.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder_helpers.h"

namespace onnxruntime {
namespace nnapi {

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

}  // namespace nnapi
}  // namespace onnxruntime
