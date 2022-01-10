// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/slice.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include <climits>
#include <gsl/gsl>
#include <tvm/topi/transform.h>

namespace onnxruntime {
namespace tvm_codegen {

// local constexpr for INT_MAX
constexpr int64_t max_range = INT_MAX;

 tvm::PrimExpr position(const  tvm::PrimExpr& dim, const tvm::Integer& offset, bool allow_out_of_bound = false) {
  if (offset->value >= max_range) {
    return allow_out_of_bound ? dim : dim - 1;
  } else if (offset->value <= -max_range) {
    return tvm::tir::make_const(tvm::DataType::Int(32), allow_out_of_bound ? -1 : 0);
  } else {
    tvm::arith::Analyzer analyzer;
    if (offset->value >= 0) {
      return analyzer.Simplify(tvm::tir::Min(offset, dim + (allow_out_of_bound ? 0 : -1)));
    } else {
      return analyzer.Simplify(dim + tvm::tir::Max(offset, -dim + (allow_out_of_bound ? -1 : 0)));
    }
  }
}

tvm::te::Tensor Slice(const tvm::te::Tensor& X,
                  const std::vector<int64_t>& starts,
                  const std::vector<int64_t>& ends,
                  const std::vector<int64_t>& axes1,
                  const std::vector<int64_t>& steps,
                  const std::string& name) {
  MTI_ASSERT(starts.size() == ends.size());
  MTI_ASSERT(starts.size() == axes1.size());
  MTI_ASSERT(starts.size() == steps.size());

  std::vector<int64_t> axes;
  for (const auto& i : axes1) {
    axes.push_back(HandleNegativeAxis(i, X->shape.size()));
  }

  tvm::Array<tvm::PrimExpr> output_shape;
  bool empty = false;
  for (int64_t i = 0; i < gsl::narrow<int64_t>(X->shape.size()); ++i) {
    auto axes_iter = std::find(axes.begin(), axes.end(), i);
    if (axes_iter != axes.end()) {
      auto axis = axes_iter - axes.begin();
      tvm::PrimExpr start = position(X->shape[i], starts[axis]);
      tvm::PrimExpr end = position(X->shape[i], ends[axis], /*allow_out_of_bound*/ true);
      tvm::arith::Analyzer analyzer;
      auto dim = analyzer.Simplify(tvm::floordiv((end - start + tvm::Integer(steps[axis] + (steps[axis] < 0 ? 1 : -1))), tvm::Integer(steps[axis])));
      auto int_dim = tvm::tir::as_const_int(dim);
      if (int_dim && *int_dim <= 0) {
        output_shape.push_back(0);
        empty = true;
      } else {
        output_shape.push_back(dim);
      }
    } else {
      output_shape.push_back(X->shape[i]);
    }
  }

  if (empty) {
    return MakeZeroTensor(output_shape, X->dtype, name);
  }

  return tvm::te::compute(
      output_shape,
      [&](const tvm::Array<tvm::tir::Var>& ovars) {
        tvm::Array<tvm::PrimExpr> ivars;
        tvm::arith::Analyzer analyzer;
        for (size_t i = 0; i < X->shape.size(); ++i) {
          auto axes_iter = std::find(axes.begin(), axes.end(), i);
          if (axes_iter != axes.end()) {
            auto axis = axes_iter - axes.begin();
            ivars.push_back(analyzer.Simplify(ovars[i] * tvm::Integer(steps[axis]) + position(X->shape[i], starts[axis])));
          } else {
            ivars.push_back(ovars[i]);
          }
        }
        return X(ivars);
      },
      name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
