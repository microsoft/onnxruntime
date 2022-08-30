// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/concat_ops.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/common/gsl.h"
#include <topi/transform.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Concat(const tvm::Array<tvm::Tensor>& inputs,
                   int64_t axis,
                   const std::string& name) {
  return ConcatSafe(inputs, axis, name);
}

// Note topi's implementation requires control flow within iterations to avoid out-of-bound access.
// Therefore, MTI implements a ConcatSafe that does not have out-of-bound access,
// and does not requires control or predicate.
tvm::Tensor ConcatSafe(const tvm::Array<tvm::Tensor>& inputs,
                       int64_t axis,
                       const std::string& name) {
  axis = HandleNegativeAxis(axis, gsl::narrow<int64_t>(inputs[0]->shape.size()));
  MTI_ASSERT(axis < gsl::narrow<int64_t>(inputs[0]->shape.size()) && "axis out of bounds");

  tvm::Array<tvm::Expr> axis_sizes;
  for (auto t : inputs) {
    axis_sizes.push_back(t->shape[axis]);
  }

  tvm::Expr join_size = axis_sizes[0];
  for (size_t i = 1; i < axis_sizes.size(); ++i) {
    join_size += axis_sizes[i];
  }
  join_size = tvm::ir::Simplify(join_size);
  tvm::Array<tvm::Expr> out_shape;
  for (size_t i = 0; i < inputs[0]->shape.size(); ++i) {
    out_shape.push_back(i == gsl::narrow<size_t>(axis) ? join_size : inputs[0]->shape[i]);
  }

  return tvm::compute(
      out_shape, [&](const tvm::Array<tvm::Var>& ovars) {
        tvm::Array<tvm::Expr> indices;

        // preset
        tvm::Expr min = 0;
        tvm::Expr extent = axis_sizes[0];
        tvm::Expr offset = 0;
        tvm::Expr ret;

        //input i = 0
        for (size_t j = 0; j < ovars.size(); ++j) {
          if (j == gsl::narrow<size_t>(axis)) {
            tvm::Expr ivar = ovars[j];
            indices.push_back(tvm::max(tvm::min(ivar, min + extent - 1), min));
          } else {
            indices.push_back(ovars[j]);
          }
        }
        ret = inputs[0](indices);

        for (size_t i = 1; i < inputs.size(); ++i) {
          offset += extent;
          tvm::Expr min = 0;
          extent = axis_sizes[i];
          auto j = gsl::narrow<size_t>(axis);
          tvm::Expr ivar = ovars[j] - offset;
          indices.Set(j, tvm::max(tvm::min(ivar, min + extent - 1), min));

          ret = tvm::ir::Select::make(ivar >= 0,
                                      inputs[i](indices),
                                      ret);
        }

        return ret;
      },
      name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
