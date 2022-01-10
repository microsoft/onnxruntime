// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/math/unary_ops.h"

#include "core/codegen/common/settings.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include <stdexcept>
#include <tvm/topi/broadcast.h>
#include <tvm/topi/elemwise.h>
#include <tvm/topi/transform.h>

// Using namespace topi for override operator +-*/
namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor Abs(const tvm::te::Tensor& X, const std::string& name) {
  return tvm::topi::abs(X, name);
}

tvm::te::Tensor _max(const tvm::te::Tensor &x, const tvm::PrimExpr& factor, std::string tag = tvm::topi::kElementWise) {
  return tvm::te::compute(
      x->shape,
      [&](const tvm::Array<tvm::tir::Var>& i) {
        auto factor_val = tvm::cast(x->dtype, factor);
        return tvm::max(x(i), factor_val);  // NOLINT(*)
      },
      "_max", tag);
}

tvm::te::Tensor _multiply(const tvm::PrimExpr& factor, const tvm::te::Tensor &x, std::string tag = tvm::topi::kElementWise) {
  return tvm::te::compute(
      x->shape,
      [&](const tvm::Array<tvm::tir::Var>& i) {
        auto factor_val = tvm::cast(x->dtype, factor);
        return factor_val * x(i);  // NOLINT(*)
      },
      "_multiply", tag);
}

tvm::te::Tensor _divide(const tvm::PrimExpr& factor, const tvm::te::Tensor &x, std::string tag = tvm::topi::kElementWise) {
  return tvm::te::compute(
      x->shape,
      [&](const tvm::Array<tvm::tir::Var>& i) {
        auto factor_val = tvm::cast(x->dtype, factor);
        return factor_val / x(i);  // NOLINT(*)
      },
      "_divide", tag);
}

tvm::te::Tensor _add(const tvm::PrimExpr& cst, const tvm::te::Tensor &x, std::string tag = tvm::topi::kElementWise) {
  return tvm::te::compute(
      x->shape,
      [&](const tvm::Array<tvm::tir::Var>& i) {
        auto cst_val = tvm::cast(x->dtype, cst);
        return cst_val + x(i);  // NOLINT(*)
      },
      "_add", tag);
}

tvm::te::Tensor _subtract(const tvm::PrimExpr& cst, const tvm::te::Tensor &x, std::string tag = tvm::topi::kElementWise) {
  return tvm::te::compute(
      x->shape,
      [&](const tvm::Array<tvm::tir::Var>& i) {
        auto cst_val = tvm::cast(x->dtype, cst);
        return cst_val - x(i);  // NOLINT(*)
      },
      "_subtract", tag);
}

tvm::te::Tensor _subtract(const tvm::te::Tensor &x, const tvm::PrimExpr& cst, std::string tag = tvm::topi::kElementWise) {
  return tvm::te::compute(
      x->shape,
      [&](const tvm::Array<tvm::tir::Var>& i) {
        auto cst_val = tvm::cast(x->dtype, cst);
        return cst_val - x(i);  // NOLINT(*)
      },
      "_subtract2", tag);
}

tvm::te::Tensor Affine(const tvm::te::Tensor& X, float alpha, float beta, const std::string& name) {
   tvm::PrimExpr alphaExpr = tvm::tir::make_const(X->dtype, alpha);
   tvm::PrimExpr betaExpr = tvm::tir::make_const(X->dtype, beta);
  return Rename(tvm::topi::add(_multiply(alphaExpr, X), betaExpr), name);
}

tvm::te::Tensor Ceil(const tvm::te::Tensor& X, const std::string& name) {
  return tvm::topi::ceil(X, name);
}

tvm::te::Tensor Clip(const tvm::te::Tensor& X,  tvm::PrimExpr min_value,  tvm::PrimExpr max_value, const std::string& name) {
  auto Y = tvm::te::compute(
      X->shape,
      [&](const tvm::Array<tvm::tir::Var>& indices) {
        return tvm::min(tvm::max(X(indices), min_value), max_value);
      },
      name);
  return Y;
}

tvm::te::Tensor Elu(const tvm::te::Tensor& X, float alpha, const std::string& name) {
   tvm::PrimExpr alphaExpr = tvm::tir::make_const(X->dtype, alpha);
   tvm::PrimExpr cst1 = tvm::tir::make_const(X->dtype, 1);
  return Rename(tvm::topi::subtract(Relu(X), _multiply(alphaExpr, Relu(_subtract(cst1, Exp(X))))), name);
}

tvm::te::Tensor Exp(const tvm::te::Tensor& X, const std::string& name) {
  return tvm::te::compute(
      X->shape,
      [&](const tvm::Array<tvm::tir::Var>& indices) {
        return tvm::exp(X(indices));
      },
      name);
}

tvm::te::Tensor Floor(const tvm::te::Tensor& X, const std::string& name) {
  return tvm::topi::floor(X, name);
}

tvm::te::Tensor HardSigmoid(const tvm::te::Tensor& X, float alpha, float beta, const std::string& name) {
   tvm::PrimExpr alphaExpr = tvm::tir::make_const(X->dtype, alpha);
   tvm::PrimExpr betaExpr = tvm::tir::make_const(X->dtype, beta);
  return tvm::topi::maximum(0, tvm::topi::minimum(1, _add(betaExpr, _multiply(alphaExpr, X))), name);
}

tvm::te::Tensor LeakyRelu(const tvm::te::Tensor& X, float alpha, const std::string& name) {
   tvm::PrimExpr alphaExpr = tvm::tir::make_const(X->dtype, alpha);
  return Rename(tvm::topi::subtract(Relu(X), _multiply(alphaExpr, Relu(Neg(X)))), name);
}

tvm::te::Tensor Log(const tvm::te::Tensor& X, const std::string& name) {
  return tvm::te::compute(
      X->shape,
      [&](const tvm::Array<tvm::tir::Var>& indices) {
        return tvm::log(X(indices));
      },
      name);
}

tvm::te::Tensor Neg(const tvm::te::Tensor& X, const std::string& name) {
  return tvm::topi::negative(X, name);
}

tvm::te::Tensor ParametricSoftplus(const tvm::te::Tensor& X, float alpha, float beta, const std::string& name) {
  tvm::PrimExpr alphaExpr = tvm::tir::make_const(X->dtype, alpha);
  tvm::PrimExpr betaExpr = tvm::tir::make_const(X->dtype, beta);
  return Rename(_multiply(alphaExpr, Softplus(_multiply(betaExpr, X))), name);
}

tvm::te::Tensor Reciprocal(const tvm::te::Tensor& X, const std::string& name) {
  tvm::PrimExpr cst1 = tvm::tir::make_const(X->dtype, 1);
  return Rename(_divide(cst1, X), name);
}

tvm::te::Tensor Relu(const tvm::te::Tensor& X, const std::string& name) {
  tvm::PrimExpr cst0 = tvm::tir::make_const(X->dtype, 0);
  return _max(X, cst0, name);
}

tvm::te::Tensor ScaledTanh(const tvm::te::Tensor& X, float alpha, float beta, const std::string& name) {
  tvm::PrimExpr alphaExpr = tvm::tir::make_const(X->dtype, alpha);
  tvm::PrimExpr betaExpr = tvm::tir::make_const(X->dtype, beta);
  return Rename(_multiply(alphaExpr, Tanh(_multiply(betaExpr, X))), name);
}

tvm::te::Tensor Selu(const tvm::te::Tensor& X, float alpha, float gamma, const std::string& name) {
  tvm::PrimExpr alphaExpr = tvm::tir::make_const(X->dtype, -alpha);
  tvm::PrimExpr gammaExpr = tvm::tir::make_const(X->dtype, gamma);
  tvm::PrimExpr cst1 = tvm::tir::make_const(X->dtype, 1);
  return Rename(_multiply(gammaExpr, (_multiply(alphaExpr, tvm::topi::add(Relu(_subtract(cst1, Exp(X))), Relu(X))))), name);
}

tvm::te::Tensor Sigmoid(const tvm::te::Tensor& X, const std::string& name) {
  return tvm::te::compute(
      X->shape,
      [&](const tvm::Array<tvm::tir::Var>& indices) {
        return tvm::tir::Select(X(indices) > 0,
                                1 / (1 + tvm::exp(-X(indices))),
                                tvm::exp(X(indices)) / (tvm::exp(X(indices)) + 1));
      },
      name);
}

tvm::te::Tensor SignNoZero(const tvm::te::Tensor& X, const std::string& name) {
  tvm::PrimExpr cst2 = tvm::tir::make_const(X->dtype, 2);
  tvm::PrimExpr cst1 = tvm::tir::make_const(X->dtype, 1);
  return Rename(_subtract(_multiply(cst2, tvm::topi::greater_equal(X, 0)), cst1), name);
}

tvm::te::Tensor Softplus(const tvm::te::Tensor& X, const std::string& name) {
  tvm::PrimExpr cst1 = tvm::tir::make_const(X->dtype, 1);
  return Rename(Log(_add(cst1, tvm::topi::add(Exp(Neg(Abs(X))), Relu(X)))), name);
}

tvm::te::Tensor Softsign(const tvm::te::Tensor& X, const std::string& name) {
  tvm::PrimExpr cst1 = tvm::tir::make_const(X->dtype, 1);
  return Rename(tvm::topi::divide(X, (_add(cst1, Abs(X)))), name);
}

tvm::te::Tensor Sqrt(const tvm::te::Tensor& X, const std::string& name) {
  return tvm::topi::sqrt(X, name);
}

tvm::te::Tensor Tanh(const tvm::te::Tensor& X, const std::string& name) {
  return tvm::te::compute(
      X->shape,
      [&](const tvm::Array<tvm::tir::Var>& indices) {
        return tvm::tir::Select(X(indices) < 0,
                                (tvm::exp(2 * X(indices)) - 1) / (tvm::exp(2 * X(indices)) + 1),
                                (1 - tvm::exp(-2 * X(indices))) / (1 + tvm::exp(-2 * X(indices))));
      },
      name);
}

tvm::te::Tensor ThresholdedRelu(const tvm::te::Tensor& X, float alpha, const std::string& name) {
   tvm::PrimExpr alphaExpr = tvm::tir::make_const(X->dtype, alpha);
  return tvm::topi::where(tvm::topi::greater(X, alphaExpr), X, tvm::topi::full_like(X, tvm::tir::make_zero(X->dtype)), name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
