// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/expand.h"
#include "core/codegen/mti/common.h"

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Expand(const tvm::Tensor& X, const tvm::Array<tvm::Expr>& new_shape, const std::string& name) {
  MTI_ASSERT(new_shape.size() >= X->shape.size());
  return tvm::compute(
      new_shape,
      [&](const tvm::Array<tvm::Var>& out_indices) {
        tvm::Array<tvm::Expr> indices;
        size_t broadcasted_rank = new_shape.size() - X->shape.size();
        for (size_t d = broadcasted_rank; d < new_shape.size(); ++d) {
          if (tvm::is_const_int(X->shape[d - broadcasted_rank], 1)) {
            indices.push_back(tvm::make_zero(HalideIR::Int(32)));
          } else {
            indices.push_back(out_indices[d]);
          }
        }
        return X(indices);
      },
      name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
