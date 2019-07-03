// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/math/gemm.h"

#include "core/codegen/mti/math/matmul_ops.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include <topi/broadcast.h>

// Using namespace topi for override operator +-*/
using namespace topi;

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Gemm(const tvm::Tensor& A, const tvm::Tensor& B, const tvm::Tensor& C,
                 bool trans_A, bool trans_B, float alpha, float beta,
                 const std::string& name) {
  auto A_dot_B = MatMul2D(A, B, trans_A, trans_B, name + "_matmul2d");
  if (beta != 0) {
    return Rename(alpha * A_dot_B + (beta * C), name);
  } else {
    return Rename(alpha * A_dot_B, name);
  }
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
