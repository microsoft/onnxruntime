// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/math/unary_ops.h"

#include "core/codegen/common/settings.h"
#include "core/codegen/mti/math/halide_ops.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include <stdexcept>
#include <topi/broadcast.h>
#include <topi/elemwise.h>
#include <topi/transform.h>

// Using namespace topi for override operator +-*/
using namespace topi;

namespace onnxruntime {
namespace tvm_codegen {

tvm::Expr exp(const tvm::Expr& x_full) {
  // Only support f32 fast math now
  if (x_full.type().element_of() == tvm::Float(32)) {
    codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();
    if (settings.HasOption(codegen::CodeGenSettings::kCodeGenFastMath)) {
      if (settings.OptionMatches(codegen::CodeGenSettings::kCodeGenFastMath,
                                 codegen::CodeGenSettings::kCodeGenFastMath_Polynormial)) {
        return halideir_exp(x_full);
      } else if (settings.OptionMatches(codegen::CodeGenSettings::kCodeGenFastMath,
                                        codegen::CodeGenSettings::kCodeGenFastMath_ShortPolynormial)) {
        // TODO: implement fast_exp
        return halideir_exp(x_full);
      }
    }
  }
  return tvm::exp(x_full);
}

tvm::Expr log(const tvm::Expr& x_full) {
  // Only support f32 fast math now
  if (x_full.type().element_of() == tvm::Float(32)) {
    codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();
    if (settings.HasOption(codegen::CodeGenSettings::kCodeGenFastMath)) {
      if (settings.OptionMatches(codegen::CodeGenSettings::kCodeGenFastMath,
                                 codegen::CodeGenSettings::kCodeGenFastMath_Polynormial)) {
        return halideir_log(x_full);
      } else if (settings.OptionMatches(codegen::CodeGenSettings::kCodeGenFastMath,
                                        codegen::CodeGenSettings::kCodeGenFastMath_ShortPolynormial)) {
        return fast_log(x_full);
      }
    }
  }
  return tvm::log(x_full);
}

tvm::Tensor Abs(const tvm::Tensor& X, const std::string& name) {
  return abs(X, name);
}

tvm::Tensor Affine(const tvm::Tensor& X, float alpha, float beta, const std::string& name) {
  return Rename(alpha * X + beta, name);
}

tvm::Tensor Ceil(const tvm::Tensor& X, const std::string& name) {
  return topi::ceil(X, name);
}

tvm::Tensor Clip(const tvm::Tensor& X, float min_value, float max_value, const std::string& name) {
  auto Y = tvm::compute(
      X->shape,
      [&](const tvm::Array<tvm::Var>& indices) {
        return tvm::min(tvm::max(X(indices), min_value), max_value);
      },
      name);
  return Y;
}

tvm::Tensor Elu(const tvm::Tensor& X, float alpha, const std::string& name) {
  return Rename(Relu(X) - alpha * Relu(1 - Exp(X)), name);
}

tvm::Tensor Exp(const tvm::Tensor& X, const std::string& name) {
  return tvm::compute(
      X->shape,
      [&](const tvm::Array<tvm::Var>& indices) {
        return tvm_codegen::exp(X(indices));
      },
      name);
}

tvm::Tensor Floor(const tvm::Tensor& X, const std::string& name) {
  return topi::floor(X, name);
}

tvm::Tensor HardSigmoid(const tvm::Tensor& X, float alpha, float beta, const std::string& name) {
  return maximum(0, minimum(1, alpha * X + beta), name);
}

tvm::Tensor LeakyRelu(const tvm::Tensor& X, float alpha, const std::string& name) {
  return Rename(Relu(X) - alpha * Relu(0 - X), name);
}

tvm::Tensor Log(const tvm::Tensor& X, const std::string& name) {
  return tvm::compute(
      X->shape,
      [&](const tvm::Array<tvm::Var>& indices) {
        return tvm_codegen::log(X(indices));
      },
      name);
}

tvm::Tensor Neg(const tvm::Tensor& X, const std::string& name) {
  return negative(X, name);
}

tvm::Tensor ParametricSoftplus(const tvm::Tensor& X, float alpha, float beta, const std::string& name) {
  return Rename(alpha * Softplus(beta * X), name);
}

tvm::Tensor Reciprocal(const tvm::Tensor& X, const std::string& name) {
  return Rename(1 / X, name);
}

tvm::Tensor Relu(const tvm::Tensor& X, const std::string& name) {
  return maximum(X, 0, name);
}

tvm::Tensor ScaledTanh(const tvm::Tensor& X, float alpha, float beta, const std::string& name) {
  return Rename(alpha * Tanh(beta * X), name);
}

tvm::Tensor Selu(const tvm::Tensor& X, float alpha, float gamma, const std::string& name) {
  return Rename(gamma * (-alpha * Relu(1 - Exp(X)) + Relu(X)), name);
}

tvm::Tensor Sigmoid(const tvm::Tensor& X, const std::string& name) {
  return tvm::compute(
      X->shape,
      [&](const tvm::Array<tvm::Var>& indices) {
        return tvm::ir::Select::make(X(indices) > 0,
                                     1 / (1 + tvm_codegen::exp(-X(indices))),
                                     tvm_codegen::exp(X(indices)) / (tvm_codegen::exp(X(indices)) + 1));
      },
      name);
}

tvm::Tensor SignNoZero(const tvm::Tensor& X, const std::string& name) {
  return Rename(greater_equal(X, 0) * 2 - 1, name);
}

tvm::Tensor Softplus(const tvm::Tensor& X, const std::string& name) {
  return Rename(Log(1 + Exp(Neg(Abs(X)))) + Relu(X), name);
}

tvm::Tensor Softsign(const tvm::Tensor& X, const std::string& name) {
  return Rename(X / (1 + Abs(X)), name);
}

tvm::Tensor Sqrt(const tvm::Tensor& X, const std::string& name) {
  return sqrt(X, name);
}

tvm::Tensor Tanh(const tvm::Tensor& X, const std::string& name) {
  return tvm::compute(
      X->shape,
      [&](const tvm::Array<tvm::Var>& indices) {
        return tvm::ir::Select::make(X(indices) < 0,
                                     (tvm_codegen::exp(2 * X(indices)) - 1) / (tvm_codegen::exp(2 * X(indices)) + 1),
                                     (1 - tvm_codegen::exp(-2 * X(indices))) / (1 + tvm_codegen::exp(-2 * X(indices))));
      },
      name);
}

tvm::Tensor ThresholdedRelu(const tvm::Tensor& X, float alpha, const std::string& name) {
  return topi::where(greater(X, alpha), X, topi::full_like(X, tvm::make_zero(X->dtype)), name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
