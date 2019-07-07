// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/slice.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include <climits>
#include <topi/transform.h>
#include <tvm/ir_pass.h>

namespace onnxruntime {
namespace tvm_codegen {

static const int64_t max_range = INT_MAX;

tvm::Expr position(const tvm::Expr& dim, const tvm::Integer& offset) {
  if (offset->value >= max_range)
    return dim;
  else if (offset->value < 0)
    return dim + offset;
  else
    return offset;
}

tvm::Tensor Slice(const tvm::Tensor& X,
                  const tvm::Array<tvm::Integer>& starts,
                  const tvm::Array<tvm::Integer>& ends,
                  const std::string& name) {
  tvm::Array<tvm::Expr> output_shape;
  for (size_t i = 0; i < X->shape.size(); ++i) {
    tvm::Expr start = position(X->shape[i], starts[i]);
    tvm::Expr end = position(X->shape[i], ends[i]);
    output_shape.push_back(tvm::ir::Simplify(end - start));
  }
  return tvm::compute(
      output_shape,
      [&](const tvm::Array<tvm::Var>& ovars) {
        tvm::Array<tvm::Expr> ivars;
        for (size_t i = 0; i < X->shape.size(); ++i)
          ivars.push_back(ovars[i] + tvm::ir::Simplify(position(X->shape[i], starts[i])));

        return X(ivars);
      },
      name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
