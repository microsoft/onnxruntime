// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/mti_x86/math/halide_ops.h"
#include "topi/broadcast.h"
#include "tvm/ir.h"

namespace onnxruntime {
namespace nuphar {

tvm::Tensor Pow(tvm::Tensor A, tvm::Tensor B, const std::string& name = "pow") {
  return topi::power(A, B);
}

tvm::Tensor Pow(tvm::Tensor A, tvm::Expr B, const std::string& name = "pow") {
  // special case for integer pow passed in
  const tvm::ir::FloatImm* op = B.as<tvm::ir::FloatImm>();
  if (op != nullptr) {
    int64_t i = (int64_t)(op->value);
    if ((double)i == op->value) {
      B = tvm::make_const(HalideIR::Int(64), i);  // replace B with integer for halideir_pow
    }
  }
  return tvm::compute(
      A->shape,
      [&](const tvm::Array<tvm::Var>& indices) {
        return halideir_pow(A(indices), B);
      },
      name);
}

tvm::Tensor Pow(tvm::Expr A, tvm::Tensor B, const std::string& name = "pow") {
  return tvm::compute(
      B->shape,
      [&](const tvm::Array<tvm::Var>& indices) {
        return halideir_pow(A, B(indices));
      },
      name);
}

}  // namespace nuphar
}  // namespace onnxruntime
