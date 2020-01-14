// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/mti_x86/math/unary_ops.h"

#include "core/codegen/mti/math/unary_ops.h"
#include "core/providers/nuphar/common/nuphar_settings.h"
#include "core/providers/nuphar/mti_x86/math/halide_ops.h"
#include "core/codegen/mti/mti_tvm_utils.h"

#include <topi/broadcast.h>
#include <topi/elemwise.h>
#include <topi/transform.h>

// Using namespace topi for override operator +-*/
using namespace topi;

namespace onnxruntime {
namespace nuphar {

// polynomial sigmoid/tanh implementation copied from core/providers/cpu/rnn/rnn_helpers.cc.cc
const float alpha_1 = 4.89352455891786e-03f;
const float alpha_3 = 6.37261928875436e-04f;
const float alpha_5 = 1.48572235717979e-05f;
const float alpha_7 = 5.12229709037114e-08f;
const float alpha_9 = -8.60467152213735e-11f;
const float alpha_11 = 2.00018790482477e-13f;
const float alpha_13 = -2.76076847742355e-16f;

const float beta_0 = 4.89352518554385e-03f;
const float beta_2 = 2.26843463243900e-03f;
const float beta_4 = 1.18534705686654e-04f;
const float beta_6 = 1.19825839466702e-06f;

const float sigmoid_bound = 20.0f;
const float tanh_bound = 10.0f;

tvm::Expr exp(const tvm::Expr& x_full) {
  // Only support f32 fast math now
  if (x_full.type().element_of() == tvm::Float(32)) {
    codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();

    if (settings.HasOption(kNupharFastMath)) {
      return halideir_exp(x_full);
    }
  }
  return tvm::exp(x_full);
}

tvm::Expr log(const tvm::Expr& x_full) {
  // Only support f32 fast math now
  if (x_full.type().element_of() == tvm::Float(32)) {
    codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();
    if (settings.HasOption(kNupharFastMath)) {
      if (settings.OptionMatches(kNupharFastMath,
                                 kNupharFastMath_Polynormial)) {
        return halideir_log(x_full);
      } else if (settings.OptionMatches(kNupharFastMath,
                                        kNupharFastMath_ShortPolynormial)) {
        return fast_log(x_full);
      }
    }
  }
  return tvm::log(x_full);
}

tvm::Tensor Erf(const tvm::Tensor& X, const std::string& name) {
  return tvm::compute(
      X->shape,
      [&](const tvm::Array<tvm::Var>& indices) {
        return halideir_erf(X(indices));
      },
      name);
}

tvm::Tensor Exp(const tvm::Tensor& X, const std::string& name) {
  return tvm::compute(
      X->shape,
      [&](const tvm::Array<tvm::Var>& indices) {
        return nuphar::exp(X(indices));
      },
      name);
}

tvm::Tensor Log(const tvm::Tensor& X, const std::string& name) {
  return tvm::compute(
      X->shape,
      [&](const tvm::Array<tvm::Var>& indices) {
        return nuphar::log(X(indices));
      },
      name);
}

tvm::Tensor ParametricSoftplus(const tvm::Tensor& X, float alpha, float beta, const std::string& name) {
  return tvm_codegen::Rename(alpha * Softplus(beta * X), name);
}

tvm::Tensor ScaledTanh(const tvm::Tensor& X, float alpha, float beta, const std::string& name) {
  return tvm_codegen::Rename(alpha * Tanh(beta * X), name);
}

tvm::Tensor Selu(const tvm::Tensor& X, float alpha, float gamma, const std::string& name) {
  return tvm_codegen::Rename(gamma * (-alpha * tvm_codegen::Relu(1 - Exp(X)) + tvm_codegen::Relu(X)), name);
}

tvm::Tensor SigmoidDeepCPU(const tvm::Tensor& X, const std::string& name) {
  return tvm::compute(
      X->shape,
      [&](const tvm::Array<tvm::Var>& indices) {
        auto x = 0.5f * max(min(X(indices), sigmoid_bound), -sigmoid_bound);
        auto x2 = x * x;
        auto p = x2 * alpha_13 + alpha_11;
        p = x2 * p + alpha_9;
        p = x2 * p + alpha_7;
        p = x2 * p + alpha_5;
        p = x2 * p + alpha_3;
        p = x2 * p + alpha_1;
        p = x * p;
        auto q = x2 * beta_6 + beta_4;
        q = x2 * q + beta_2;
        q = x2 * q + beta_0;
        return 0.5f * (1 + (p / q));
      },
      name);
}

tvm::Tensor Sigmoid(const tvm::Tensor& X, const std::string& name) {
  codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();

  if (settings.HasOption(kNupharFastActivation)) {
    return SigmoidDeepCPU(X, name);
  }

  return tvm::compute(
      X->shape,
      [&](const tvm::Array<tvm::Var>& indices) {
        return tvm::ir::Select::make(X(indices) > 0,
                                     1 / (1 + nuphar::exp(-X(indices))),
                                     nuphar::exp(X(indices)) / (nuphar::exp(X(indices)) + 1));
      },
      name);
}

tvm::Tensor Softplus(const tvm::Tensor& X, const std::string& name) {
  return tvm_codegen::Rename(Log(1 + Exp(tvm_codegen::Neg(tvm_codegen::Abs(X)))) + tvm_codegen::Relu(X), name);
}

tvm::Tensor TanhDeepCPU(const tvm::Tensor& X, const std::string& name) {
  return tvm::compute(
      X->shape,
      [&](const tvm::Array<tvm::Var>& indices) {
        auto x = max(min(X(indices), tanh_bound), -tanh_bound);
        auto x2 = x * x;
        auto p = x2 * alpha_13 + alpha_11;
        p = x2 * p + alpha_9;
        p = x2 * p + alpha_7;
        p = x2 * p + alpha_5;
        p = x2 * p + alpha_3;
        p = x2 * p + alpha_1;
        p = x * p;
        auto q = x2 * beta_6 + beta_4;
        q = x2 * q + beta_2;
        q = x2 * q + beta_0;
        return p / q;
      },
      name);
}

tvm::Tensor Tanh(const tvm::Tensor& X, const std::string& name) {
  codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();
  if (settings.HasOption(kNupharFastActivation)) {
    return TanhDeepCPU(X, name);
  }

  return tvm::compute(
      X->shape,
      [&](const tvm::Array<tvm::Var>& indices) {
        return tvm::ir::Select::make(X(indices) < 0,
                                     (nuphar::exp(2 * X(indices)) - 1) / (nuphar::exp(2 * X(indices)) + 1),
                                     (1 - nuphar::exp(-2 * X(indices))) / (1 + nuphar::exp(-2 * X(indices))));
      },
      name);
}

}  // namespace nuphar
}  // namespace onnxruntime
