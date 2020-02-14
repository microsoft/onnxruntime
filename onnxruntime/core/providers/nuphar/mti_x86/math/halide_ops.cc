// Halide Copyright info:
/*
Copyright (c) 2012-2018 MIT CSAIL, Google Inc., and other contributors

Developed by:

  The Halide team
  http://halide-lang.org

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file contains some ops that were copied from Halide (with small modifications).

#include "core/providers/nuphar/mti_x86/math/halide_ops.h"

#include <topi/elemwise.h>

namespace onnxruntime {
namespace nuphar {

// Evaluate a float polynomial efficiently, taking instruction latency
// into account. The high order terms come first. n is the number of
// terms, which is the degree plus one.
static tvm::Expr evaluate_polynomial(const tvm::Expr& x, float* coeff, int n) {
  DCHECK(n >= 2);
  tvm::Expr x2 = x * x;

  tvm::Expr even_terms = coeff[0];
  tvm::Expr odd_terms = coeff[1];

  for (int i = 2; i < n; i++) {
    if ((i & 1) == 0) {
      if (coeff[i] == 0.0f) {
        even_terms *= x2;
      } else {
        even_terms = even_terms * x2 + coeff[i];
      }
    } else {
      if (coeff[i] == 0.0f) {
        odd_terms *= x2;
      } else {
        odd_terms = odd_terms * x2 + coeff[i];
      }
    }
  }

  if ((n & 1) == 0) {
    return even_terms * x + odd_terms;
  } else {
    return odd_terms * x + even_terms;
  }
}

// Fast math ops based on those from Syrah (http://github.com/boulos/syrah). Thanks, Solomon!

// Factor a float into 2^exponent * reduced, where reduced is between 0.75 and 1.5
static void range_reduce_log(const tvm::Expr& input, tvm::Expr* reduced, tvm::Expr* exponent) {
  tvm::Type type = input.type();
  tvm::Type int_type = tvm::Int(32, type.lanes());
  tvm::Expr int_version = tvm::reinterpret(int_type, input);

  // single precision = SEEE EEEE EMMM MMMM MMMM MMMM MMMM MMMM
  // exponent mask    = 0111 1111 1000 0000 0000 0000 0000 0000
  //                    0x7  0xF  0x8  0x0  0x0  0x0  0x0  0x0
  // non-exponent     = 1000 0000 0111 1111 1111 1111 1111 1111
  //                  = 0x8  0x0  0x7  0xF  0xF  0xF  0xF  0xF
  tvm::Expr non_exponent_mask = tvm::make_const(int_type, 0x807fffff);

  // Extract a version with no exponent (between 1.0 and 2.0)
  tvm::Expr no_exponent = int_version & non_exponent_mask;

  // If > 1.5, we want to divide by two, to normalize back into the
  // range (0.75, 1.5). We can detect this by sniffing the high bit
  // of the mantissa.
  tvm::Expr new_exponent = no_exponent >> 22;

  tvm::Expr new_biased_exponent = 127 - new_exponent;
  tvm::Expr old_biased_exponent = int_version >> 23;
  *exponent = old_biased_exponent - new_biased_exponent;

  tvm::Expr blended = (int_version & non_exponent_mask) | (new_biased_exponent << 23);

  *reduced = tvm::reinterpret(type, blended);
}

tvm::Expr halideir_log(const tvm::Expr& x_full) {
  tvm::Type type = x_full.type();
  DCHECK(type.element_of() == tvm::Float(32));

  tvm::Expr nan = tvm::make_const(tvm::Float(32), 0x7FF8000000000000);
  // tvm::Expr nan = tvm::ir::Call::make(type, "nan_f32", {}, tvm::ir::Call::PureExtern);
  tvm::Expr neg_inf = tvm::make_const(tvm::Float(32), 0xFFF0000000000000);
  // tvm::Expr neg_inf = tvm::ir::Call::make(type, "neg_inf_f32", {}, tvm::ir::Call::PureExtern);

  tvm::Expr use_nan = x_full < 0.0f;       // log of a negative returns nan
  tvm::Expr use_neg_inf = x_full == 0.0f;  // log of zero is -inf
  tvm::Expr exceptional = use_nan | use_neg_inf;

  // Avoid producing nans or infs by generating ln(1.0f) instead and
  // then fixing it later.
  tvm::Expr patched = tvm::ir::Select::make(exceptional, tvm::make_const(type, 1.0), x_full);
  tvm::Expr reduced, exponent;
  range_reduce_log(patched, &reduced, &exponent);

  // Very close to the Taylor series for log about 1, but tuned to
  // have minimum relative error in the reduced domain (0.75 - 1.5).

  float coeff[] = {
      0.05111976432738144643f,
      -0.11793923497136414580f,
      0.14971993724699017569f,
      -0.16862004708254804686f,
      0.19980668101718729313f,
      -0.24991211576292837737f,
      0.33333435275479328386f,
      -0.50000106292873236491f,
      1.0f,
      0.0f};
  tvm::Expr x1 = reduced - 1.0f;
  tvm::Expr result = evaluate_polynomial(x1, coeff, sizeof(coeff) / sizeof(coeff[0]));

  result += tvm::cast(type, exponent) * logf(2.0);

  result = tvm::ir::Select::make(exceptional, tvm::ir::Select::make(use_nan, nan, neg_inf), result);

  // This introduces lots of common subexpressions
  //result = common_subexpression_elimination(result);

  return result;
}

tvm::Expr raise_to_integer_power(const tvm::Expr& e, int64_t p) {
  tvm::Expr result;
  if (p == 0) {
    result = tvm::make_const(e.type(), 1);
  } else if (p == 1) {
    result = e;
  } else if (p < 0) {
    result = tvm::make_const(e.type(), 1) / raise_to_integer_power(e, -p);
  } else {
    // p is at least 2
    tvm::Expr y = raise_to_integer_power(e, p >> 1);
    if (p & 1)
      result = y * y * e;
    else
      result = y * y;
  }
  return result;
}

/** Return one floating point expression raised to the power of
 * another. The type of the result is given by the type of the first
 * argument. If the first argument is not a floating-point type, it is
 * cast to Float(32). For Float(32), cleanly vectorizable, and
 * accurate up to the last few bits of the mantissa. Gets worse when
 * approaching overflow. Vectorizes cleanly. */
tvm::Expr halideir_pow(tvm::Expr x, tvm::Expr y) {
  if (const int64_t* i = as_const_int(y)) {
    return raise_to_integer_power(x, *i);
  }

  if (x.type() == HalideIR::Float(64)) {
    y = tvm::cast(HalideIR::Float(64), y);
    return HalideIR::Internal::Call::make(HalideIR::Float(64), "pow_f64", {x, y}, HalideIR::Internal::Call::PureExtern);
  } else if (x.type() == HalideIR::Float(16)) {
    y = tvm::cast(HalideIR::Float(16), y);
    return HalideIR::Internal::Call::make(HalideIR::Float(16), "pow_f16", {x, y}, HalideIR::Internal::Call::PureExtern);
  } else {
    x = tvm::cast(HalideIR::Float(32), x);
    y = tvm::cast(HalideIR::Float(32), y);
    return HalideIR::Internal::Call::make(HalideIR::Float(32), "pow_f32", {x, y}, HalideIR::Internal::Call::PureExtern);
  }
}

tvm::Expr halideir_erf(const tvm::Expr& x_full) {
  DCHECK(x_full.type() == HalideIR::Float(32));

  // Extract the sign and magnitude.
  tvm::Expr sign = tvm::ir::Select::make(x_full < 0, -1.0f, 1.0f);
  tvm::Expr x = abs(x_full);

  // An approximation very similar to one from Abramowitz and
  // Stegun, but tuned for values > 1. Takes the form 1 - P(x)^-16.
  float c1[] = {0.0000818502f,
                -0.0000026500f,
                0.0009353904f,
                0.0081960206f,
                0.0430054424f,
                0.0703310579f,
                1.0f};
  tvm::Expr approx1 = evaluate_polynomial(x, c1, sizeof(c1) / sizeof(c1[0]));

  approx1 = 1.0f - halideir_pow(approx1, -16);

  // An odd polynomial tuned for values < 1. Similar to the Taylor
  // expansion of erf.
  float c2[] = {-0.0005553339f,
                0.0048937243f,
                -0.0266849239f,
                0.1127890132f,
                -0.3761207240f,
                1.1283789803f};

  tvm::Expr approx2 = evaluate_polynomial(x * x, c2, sizeof(c2) / sizeof(c2[0]));
  approx2 *= x;

  // Switch between the two approximations based on the magnitude.
  tvm::Expr y = tvm::ir::Select::make(x > 1.0f, approx1, approx2);

  //Expr result = common_subexpression_elimination(sign * y);

  return sign * y;
}

tvm::Expr fast_log(const tvm::Expr& x) {
  DCHECK(x.type() == tvm::Float(32));

  tvm::Expr reduced, exponent;
  range_reduce_log(x, &reduced, &exponent);

  tvm::Expr x1 = reduced - 1.0f;

  float coeff[] = {
      0.07640318789187280912f,
      -0.16252961013874300811f,
      0.20625219040645212387f,
      -0.25110261010892864775f,
      0.33320464908377461777f,
      -0.49997513376789826101f,
      1.0f,
      0.0f};

  tvm::Expr result = evaluate_polynomial(x1, coeff, sizeof(coeff) / sizeof(coeff[0]));
  result += tvm::cast(x.type(), exponent) * logf(2);
  //result = common_subexpression_elimination(result);
  return result;
}

tvm::Expr halideir_exp(const tvm::Expr& x_full) {
  tvm::Type type = x_full.type();
  DCHECK(type.element_of() == tvm::Float(32));

  float ln2_part1 = 0.6931457519f;
  float ln2_part2 = 1.4286067653e-6f;
  float one_over_ln2 = 1.0f / logf(2.0f);

  tvm::Expr scaled = x_full * one_over_ln2;
  tvm::Expr k_real = tvm::floor(scaled);
  tvm::Expr k = tvm::cast(tvm::Int(32, type.lanes()), k_real);

  tvm::Expr x = x_full - k_real * ln2_part1;
  x -= k_real * ln2_part2;

  float coeff[] = {
      0.00031965933071842413f,
      0.00119156835564003744f,
      0.00848988645943932717f,
      0.04160188091348320655f,
      0.16667983794100929562f,
      0.49999899033463041098f,
      1.0f,
      1.0f};
  tvm::Expr result = evaluate_polynomial(x, coeff, sizeof(coeff) / sizeof(coeff[0]));

  // Compute 2^k.
  int fpbias = 127;
  tvm::Expr biased = k + fpbias;

  tvm::Expr inf = tvm::make_const(tvm::Float(32), 0x7FF0000000000000);
  // Expr inf = Call::make(type, "inf_f32", {}, Call::PureExtern);

  // Shift the bits up into the exponent field and reinterpret this
  // thing as float.
  tvm::Expr two_to_the_n = tvm::reinterpret(type, biased << 23);
  result *= two_to_the_n;

  // Catch overflow and underflow
  result = tvm::ir::Select::make(biased < 255, result, inf);
  result = tvm::ir::Select::make(biased > 0, result, tvm::make_zero(type));

  // This introduces lots of common subexpressions
  // result = common_subexpression_elimination(result);
  return result;
}

}  // namespace nuphar
}  // namespace onnxruntime
