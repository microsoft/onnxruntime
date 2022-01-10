// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/tensor/expand.h"
#include "core/codegen/mti/common.h"

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor Expand(const tvm::te::Tensor& X, const tvm::Array<tvm::PrimExpr>& new_shape, const std::string& name) {
  MTI_ASSERT(new_shape.size() >= X->shape.size());
  return tvm::te::compute(
      new_shape,
      [&](const tvm::Array<tvm::tir::Var>& out_indices) {
        tvm::Array<tvm::PrimExpr> indices;
        size_t broadcasted_rank = new_shape.size() - X->shape.size();
        for (size_t d = broadcasted_rank; d < new_shape.size(); ++d) {
          if (tvm::tir::is_const_int(X->shape[d - broadcasted_rank], 1)) {
            indices.push_back(tvm::tir::make_zero(tvm::DataType::Int(32)));
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
