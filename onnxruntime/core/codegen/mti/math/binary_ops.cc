// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/math/binary_ops.h"

#include "core/codegen/mti/math/unary_ops.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/tensor/cast_ops.h"
#include <topi/broadcast.h>

// Using namespace topi for override operator +-*/
using namespace topi;

namespace onnxruntime {
namespace tvm_codegen {

#define TVM_BINARY_OP1(op, expr)                                                            \
  tvm::Tensor op(const tvm::Tensor& lhs, const tvm::Tensor& rhs, const std::string& name) { \
    return Rename(expr, name);                                                              \
  }                                                                                         \
  tvm::Tensor op(const tvm::Tensor& lhs, const tvm::Expr& rhs, const std::string& name) {   \
    return Rename(expr, name);                                                              \
  }

#define TVM_BINARY_OP(op, expr)                                                           \
  TVM_BINARY_OP1(op, expr)                                                                \
  tvm::Tensor op(const tvm::Expr& lhs, const tvm::Tensor& rhs, const std::string& name) { \
    return Rename(expr, name);                                                            \
  }

TVM_BINARY_OP(Add, lhs + rhs);
TVM_BINARY_OP(Div, lhs / rhs);
TVM_BINARY_OP(Max, maximum(lhs, rhs));
TVM_BINARY_OP(Min, minimum(lhs, rhs));
TVM_BINARY_OP(Mul, lhs* rhs);
TVM_BINARY_OP1(PRelu, Relu(lhs) - rhs * Relu(0 - lhs));
TVM_BINARY_OP(Sub, lhs - rhs);

tvm::Tensor Equal(const tvm::Tensor& lhs, const tvm::Tensor& rhs, const std::string& name) {
  return topi::equal(lhs, rhs, name);
}
tvm::Tensor Equal(const tvm::Tensor& lhs, const tvm::Expr& rhs, const std::string& name) {
  return topi::equal(lhs, rhs, name);
}
tvm::Tensor Equal(const tvm::Expr& lhs, const tvm::Tensor& rhs, const std::string& name) {
  return topi::equal(lhs, rhs, name);
}

tvm::Tensor Greater(const tvm::Tensor& lhs, const tvm::Tensor& rhs, const std::string& name) {
  return topi::greater(lhs, rhs, name);
}
tvm::Tensor Greater(const tvm::Tensor& lhs, const tvm::Expr& rhs, const std::string& name) {
  return topi::greater(lhs, rhs, name);
}
tvm::Tensor Greater(const tvm::Expr& lhs, const tvm::Tensor& rhs, const std::string& name) {
  return topi::greater(lhs, rhs, name);
}

tvm::Tensor Less(const tvm::Tensor& lhs, const tvm::Tensor& rhs, const std::string& name) {
  return topi::less(lhs, rhs, name);
}
tvm::Tensor Less(const tvm::Tensor& lhs, const tvm::Expr& rhs, const std::string& name) {
  return topi::less(lhs, rhs, name);
}
tvm::Tensor Less(const tvm::Expr& lhs, const tvm::Tensor& rhs, const std::string& name) {
  return topi::less(lhs, rhs, name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
