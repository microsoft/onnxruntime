// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/tensor/where.h"

#include <topi/broadcast.h>
#include <topi/transform.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Where(const tvm::Tensor& B,
                  const tvm::Tensor& X,
                  const tvm::Tensor& Y,
                  const std::string& name) {
  size_t rank = std::max(std::max(B->shape.size(), X->shape.size()), Y->shape.size());
  tvm::Array<tvm::Expr> output_shape;
  for (size_t i = 0; i < rank; ++i) {
    tvm::Expr dim = tvm::make_const(HalideIR::Int(32), 1);
    bool broadcasted =
        BroadcastDim(B->shape, i, rank, dim) &&
        BroadcastDim(X->shape, i, rank, dim) &&
        BroadcastDim(Y->shape, i, rank, dim);
    MTI_ASSERT(broadcasted);
    output_shape.push_back(dim);
  }

  return topi::where(topi::broadcast_to(B, output_shape),
                     topi::broadcast_to(X, output_shape),
                     topi::broadcast_to(Y, output_shape),
                     name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
