// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/gather_elements.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include <topi/transform.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor GatherElements(const tvm::Tensor& t,
                           int64_t axis,
                           const tvm::Tensor& indices,
                           const std::string& name) {
  tvm::Array<tvm::Expr> output_shape;
  int64_t indices_rank = static_cast<int64_t>(indices->shape.size());
  // output shape is the same as indices
  for (int64_t i = 0; i < indices_rank; ++i)
    output_shape.push_back(indices->shape[i]);

  tvm::Expr idx_upper_bound = t->shape[axis];
  auto l = [&](const tvm::Array<tvm::Var>& ovars) {
    tvm::Array<tvm::Expr> ivars;
    for (int64_t i = 0; i < indices_rank; i++) {
      if (i == axis) {
        tvm::Array<tvm::Expr> idx_vars;
        for (int64_t j = 0; j < indices_rank; j++)
          idx_vars.push_back(ovars[j]);
        // make sure idx is clamped in the range of [-idx_upper_bound, idx_upper_bound - 1]
        tvm::Expr real_idx = tvm_codegen::ClampIndex(indices(idx_vars), idx_upper_bound);
        // tvm idx must be of Int(32)
        ivars.push_back(tvm::cast(tvm::Int(32), real_idx));
      } else {
        ivars.push_back(ovars[i]);
      }
    }
    return t(ivars);
  };

  return tvm::compute(output_shape, l, name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
