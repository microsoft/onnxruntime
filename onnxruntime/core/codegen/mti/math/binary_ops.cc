// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/math/binary_ops.h"

#include "core/codegen/mti/math/unary_ops.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/tensor/cast_ops.h"
#include <tvm/topi/broadcast.h>

// Using namespace topi for override operator +-*/
namespace onnxruntime {
namespace tvm_codegen {

#define TVM_BINARY_OP1(op, expr)                                                            \
  tvm::te::Tensor op(const tvm::te::Tensor& lhs, const tvm::te::Tensor& rhs, const std::string& name) { \
    return Rename(expr, name);                                                              \
  }                                                                                         \
  tvm::te::Tensor op(const tvm::te::Tensor& lhs, const  tvm::PrimExpr& rhs, const std::string& name) {   \
    return Rename(expr, name);                                                              \
  }

#define TVM_BINARY_OP(op, expr)                                                           \
  TVM_BINARY_OP1(op, expr)                                                                \
  tvm::te::Tensor op(const  tvm::PrimExpr& lhs, const tvm::te::Tensor& rhs, const std::string& name) { \
    return Rename(expr, name);                                                            \
  }

TVM_BINARY_OP(Add, tvm::topi::add(lhs, rhs));
TVM_BINARY_OP(Div, tvm::topi::divide(lhs, rhs));
TVM_BINARY_OP(Max, tvm::topi::maximum(lhs, rhs));
TVM_BINARY_OP(Min, tvm::topi::minimum(lhs, rhs));
TVM_BINARY_OP(Mul, tvm::topi::multiply(lhs, rhs));
TVM_BINARY_OP1(PRelu, tvm::topi::subtract(Relu(lhs), tvm::topi::multiply(rhs, Relu(Neg(lhs)))));
TVM_BINARY_OP(Sub, tvm::topi::subtract(lhs, rhs));

tvm::te::Tensor Equal(const tvm::te::Tensor& lhs, const tvm::te::Tensor& rhs, const std::string& name) {
  return tvm::topi::equal(lhs, rhs, name);
}
tvm::te::Tensor Equal(const tvm::te::Tensor& lhs, const  tvm::PrimExpr& rhs, const std::string& name) {
  return tvm::topi::equal(lhs, rhs, name);
}
tvm::te::Tensor Equal(const  tvm::PrimExpr& lhs, const tvm::te::Tensor& rhs, const std::string& name) {
  return tvm::topi::equal(lhs, rhs, name);
}

tvm::te::Tensor Greater(const tvm::te::Tensor& lhs, const tvm::te::Tensor& rhs, const std::string& name) {
  return tvm::topi::greater(lhs, rhs, name);
}
tvm::te::Tensor Greater(const tvm::te::Tensor& lhs, const  tvm::PrimExpr& rhs, const std::string& name) {
  return tvm::topi::greater(lhs, rhs, name);
}
tvm::te::Tensor Greater(const  tvm::PrimExpr& lhs, const tvm::te::Tensor& rhs, const std::string& name) {
  return tvm::topi::greater(lhs, rhs, name);
}

tvm::te::Tensor Less(const tvm::te::Tensor& lhs, const tvm::te::Tensor& rhs, const std::string& name) {
  return tvm::topi::less(lhs, rhs, name);
}
tvm::te::Tensor Less(const tvm::te::Tensor& lhs, const  tvm::PrimExpr& rhs, const std::string& name) {
  return tvm::topi::less(lhs, rhs, name);
}
tvm::te::Tensor Less(const  tvm::PrimExpr& lhs, const tvm::te::Tensor& rhs, const std::string& name) {
  return tvm::topi::less(lhs, rhs, name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
