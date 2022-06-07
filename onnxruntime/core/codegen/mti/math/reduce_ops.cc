// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/math/reduce_ops.h"

#include "core/codegen/mti/math/binary_ops.h"
#include "core/codegen/mti/math/unary_ops.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include <topi/reduction.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor ArgMax(const tvm::Tensor& X, int64_t axis, bool keep_dims, const std::string& name) {
  return Rename(topi::argmax(X, ToTvmArrayInt({axis}), keep_dims), name);
}

tvm::Tensor ArgMin(const tvm::Tensor& X, int64_t axis, bool keep_dims, const std::string& name) {
  return Rename(topi::argmin(X, ToTvmArrayInt({axis}), keep_dims), name);
}

tvm::Tensor ReduceL1(const tvm::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  return ReduceSum(Abs(X), axes, keep_dims, name);
}

tvm::Tensor ReduceL2(const tvm::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  return Sqrt(ReduceSumSquare(X, axes, keep_dims), name);
}

tvm::Tensor ReduceLogSum(const tvm::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  return Log(ReduceSum(X, axes, keep_dims), name);
}

tvm::Tensor ReduceLogSumExp(const tvm::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  tvm::Tensor reduce_max = ReduceMax(X, axes, true);
  tvm::Tensor exp_delta = Exp(Sub(X, reduce_max));
  tvm::Tensor reduce_max_keep_dims = ReduceMax(X, axes, keep_dims);
  return Add(ReduceLogSum(exp_delta, axes, keep_dims), reduce_max_keep_dims, name);
}

tvm::Tensor ReduceMax(const tvm::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  return Rename(topi::max(X, ToTvmArrayInt(axes), keep_dims), name);
}

tvm::Tensor ReduceMean(const tvm::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  tvm::Tensor reduce_sum = ReduceSum(X, axes, keep_dims);
  tvm::Expr count = tvm::make_const(reduce_sum->dtype, 1.0f);
  if (axes.empty()) {
    for (const auto& dim : X->shape)
      count = count * dim;
  } else {
    for (int64_t axis : axes) {
      int64_t i = HandleNegativeAxis(axis, X->shape.size());
      count = count * X->shape[i];
    }
  }
  return tvm::compute(
      reduce_sum->shape,
      [&](const tvm::Array<tvm::Var>& i) {
        return reduce_sum(i) / count;
      },
      name);
}

tvm::Tensor ReduceMin(const tvm::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  return Rename(topi::min(X, ToTvmArrayInt(axes), keep_dims), name);
}

tvm::Tensor ReduceProd(const tvm::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  auto prod = [](tvm::Expr source, tvm::Array<tvm::IterVar> rdom) {
    tvm::Var x("x", source.type()), y("y", source.type());
    tvm::Expr Rename_element = tvm::make_const(source.type(), 1.0f);
    tvm::ir::CommReducer combiner =
        tvm::ir::CommReducerNode::make({x}, {y}, {x * y}, {Rename_element});
    return tvm::ir::Reduce::make(combiner, {source}, rdom, tvm::make_const(tvm::Bool(1), true), 0);
  };

  return Rename(topi::CommReduce(X, ToTvmArrayInt(axes), prod, keep_dims, true), name);
}

tvm::Tensor ReduceSum(const tvm::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  return Rename(topi::sum(X, ToTvmArrayInt(axes), keep_dims), name);
}

tvm::Tensor ReduceSumSquare(const tvm::Tensor& X, const std::vector<int64_t>& axes, bool keep_dims, const std::string& name) {
  return Rename(topi::sum(Mul(X, X), ToTvmArrayInt(axes), keep_dims), name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
