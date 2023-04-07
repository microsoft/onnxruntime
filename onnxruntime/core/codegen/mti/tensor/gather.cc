// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/gather.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/common/gsl.h"
#include <topi/transform.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Gather(const tvm::Tensor& t,
                   int64_t axis,
                   const tvm::Tensor& indices,
                   const std::string& name) {
  // handle negative axis
  axis = HandleNegativeAxis(axis, gsl::narrow<int64_t>(t->shape.size()));
  size_t axis_t = gsl::narrow<size_t>(axis);

  tvm::Array<tvm::Expr> output_shape;
  for (size_t i = 0; i < axis_t; ++i)
    output_shape.push_back(t->shape[i]);

  for (size_t i = 0; i < indices->shape.size(); ++i)
    output_shape.push_back(indices->shape[i]);

  for (size_t i = axis_t + 1; i < t->shape.size(); ++i)
    output_shape.push_back(t->shape[i]);

  tvm::Expr idx_upper_bound = t->shape[axis_t];
  auto l = [&](const tvm::Array<tvm::Var>& ovars) {
    tvm::Array<tvm::Expr> ivars;
    for (size_t i = 0; i < t->shape.size(); ++i) {
      if (i < axis_t) {
        ivars.push_back(ovars[i]);
      } else if (i == axis_t) {
        tvm::Array<tvm::Expr> idx_vars;
        for (size_t d = 0; d < indices->shape.size(); ++d)
          idx_vars.push_back(ovars[axis_t + d]);
        // make sure idx is clamped in the range of [-idx_upper_bound, idx_upper_bound - 1]
        tvm::Expr real_idx = tvm_codegen::ClampIndex(indices(idx_vars), idx_upper_bound);
        ivars.push_back(tvm::cast(tvm::Int(32), real_idx));  // tvm indices must be Int32
      } else {
        ivars.push_back(ovars[i - 1 + indices->shape.size()]);
      }
    }
    return t(ivars);
  };

  return tvm::compute(output_shape, l, name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
