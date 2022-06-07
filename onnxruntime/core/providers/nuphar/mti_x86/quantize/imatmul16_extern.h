// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace nuphar {

tvm::Tensor
IMatMul16ExternMKL(const tvm::Tensor& A,
                   const tvm::Tensor& B,
                   const tvm::Array<tvm::Expr>& output_shape,
                   int input_dim,
                   int embed_dim,
                   const std::string& name = "IMatMul16ExternMKL");

tvm::Tensor
IMatMul16ExternAVX2(const tvm::Tensor& A,
                    const tvm::Tensor& B,
                    const tvm::Array<tvm::Expr>& output_shape,
                    int input_dim,
                    int embed_dim,
                    const std::string& name = "IMatMul16ExternAVX2");

}  // namespace nuphar
}  // namespace onnxruntime
