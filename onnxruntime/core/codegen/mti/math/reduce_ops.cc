// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/math/reduce_ops.h"

#include "core/codegen/mti/math/binary_ops.h"
#include "core/codegen/mti/math/unary_ops.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include <tvm/topi/reduction.h>
#include <tvm/tir/var.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor ArgMax(const tvm::te::Tensor& X, int64_t axis, bool keep_dims, const std::string& name) {
  return Rename(tvm::topi::argmax(X, ToTvmArrayInt({axis}), keep_dims), name);
}

tvm::te::Tensor ArgMin(const tvm::te::Tensor& X, int64_t axis, bool keep_dims, const std::string& name) {
  return Rename(tvm::topi::argmin(X, ToTvmArrayInt({axis}), keep_dims), name);
}

tvm::te::Tensor ReduceL1(const tvm::te::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  return ReduceSum(Abs(X), axes, keep_dims, name);
}

tvm::te::Tensor ReduceL2(const tvm::te::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  return Sqrt(ReduceSumSquare(X, axes, keep_dims), name);
}

tvm::te::Tensor ReduceLogSum(const tvm::te::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  return Log(ReduceSum(X, axes, keep_dims), name);
}

tvm::te::Tensor ReduceLogSumExp(const tvm::te::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  tvm::te::Tensor reduce_max = ReduceMax(X, axes, true);
  tvm::te::Tensor exp_delta = Exp(Sub(X, reduce_max));
  tvm::te::Tensor reduce_max_keep_dims = ReduceMax(X, axes, keep_dims);
  return Add(ReduceLogSum(exp_delta, axes, keep_dims), reduce_max_keep_dims, name);
}

tvm::te::Tensor ReduceMax(const tvm::te::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  return Rename(tvm::topi::max(X, ToTvmArrayInt(axes), keep_dims), name);
}

tvm::te::Tensor ReduceMean(const tvm::te::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  tvm::te::Tensor reduce_sum = ReduceSum(X, axes, keep_dims);
   tvm::PrimExpr count = tvm::tir::make_const(reduce_sum->dtype, 1.0f);
  if (axes.empty()) {
    for (const auto& dim : X->shape)
      count = count * dim;
  } else {
    for (int64_t axis : axes) {
      int64_t i = HandleNegativeAxis(axis, X->shape.size());
      count = count * X->shape[i];
    }
  }
  return tvm::te::compute(
      reduce_sum->shape,
      [&](const tvm::Array<tvm::tir::Var>& i) {
        return reduce_sum(i) / count;
      },
      name);
}

tvm::te::Tensor ReduceMin(const tvm::te::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  return Rename(tvm::topi::min(X, ToTvmArrayInt(axes), keep_dims), name);
}

tvm::te::Tensor ReduceProd(const tvm::te::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  auto prod = [](tvm::PrimExpr source, tvm::Array<tvm::tir::IterVar> rdom, tvm::Array<tvm::PrimExpr> init = {}, tvm::Span span=tvm::Span()) {
    tvm::tir::Var x("x", source.dtype(), span), y("y", source.dtype(), span);
    tvm::PrimExpr result = tvm::tir::Mul(x, y, span);
    tvm::PrimExpr Rename_element = tvm::tir::make_const(source.dtype(), 1.0f, span);
    tvm::tir::CommReducer combiner =
        tvm::tir::CommReducer({x}, {y}, {result}, {Rename_element}, span);
    return tvm::tir::Reduce(combiner, {source}, rdom, tvm::tir::make_const(tvm::DataType::Bool(1), true), 0, init, span);
  };

  return Rename(tvm::topi::CommReduce(X, ToTvmArrayInt(axes), prod, keep_dims, true), name);
}

tvm::te::Tensor ReduceSum(const tvm::te::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  return Rename(tvm::topi::sum(X, ToTvmArrayInt(axes), keep_dims), name);
}

tvm::te::Tensor ReduceSumSquare(const tvm::te::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  return Rename(tvm::topi::sum(Mul(X, X), ToTvmArrayInt(axes), keep_dims), name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
