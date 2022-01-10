// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/math/gemm.h"

#include "core/codegen/mti/math/matmul_ops.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include <tvm/topi/broadcast.h>

// Using namespace topi for override operator +-*/
namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor _multiply(const tvm::PrimExpr& factor, const tvm::te::Tensor &x, std::string tag = tvm::topi::kElementWise) {
  return tvm::te::compute(
      x->shape,
      [&](const tvm::Array<tvm::tir::Var>& i) {
        auto factor_val = tvm::cast(x->dtype, factor);
        return factor_val * x(i);  // NOLINT(*)
      },
      "_multiply", tag);
}

tvm::te::Tensor Gemm(const tvm::te::Tensor& A, const tvm::te::Tensor& B, const tvm::te::Tensor& C,
                 bool trans_A, bool trans_B, float alpha, float beta,
                 const std::string& name) {
  auto A_dot_B = MatMul2D(A, B, trans_A, trans_B, name + "_matmul2d");
  tvm::PrimExpr alphaExpr = tvm::tir::make_const(A->dtype, alpha);
  if (beta != 0) {
    tvm::PrimExpr betaExpr = tvm::tir::make_const(A->dtype, beta);
    return Rename(tvm::topi::add(_multiply(alphaExpr, A_dot_B), _multiply(betaExpr, C)), name);
  } else {
    return Rename(_multiply(alphaExpr, A_dot_B), name);
  }
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
