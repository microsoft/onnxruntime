// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Conv1D(const tvm::Tensor& input,
                   const tvm::Tensor& filter,
                   const tvm::Array<tvm::Expr>& output_shape,
                   const tvm::Array<tvm::Expr>& stride,
                   const tvm::Array<tvm::Expr>& padding,
                   const std::string& name = "conv1d");

tvm::Tensor Conv2D(const tvm::Tensor& input,
                   const tvm::Tensor& filter,
                   const tvm::Array<tvm::Expr>& output_shape,
                   const tvm::Array<tvm::Expr>& stride,
                   const tvm::Array<tvm::Expr>& padding,
                   const std::string& name = "conv2d");

tvm::Tensor Conv2D_native(const tvm::Tensor& input,
                          const tvm::Tensor& filter,
                          const tvm::Array<tvm::Expr>& output_shape,
                          const tvm::Array<tvm::Expr>& stride,
                          const tvm::Array<tvm::Expr>& padding,
                          const std::string& name = "conv2d_native");

tvm::Tensor Conv2D_gemm(const tvm::Tensor& input,
                        const tvm::Tensor& filter,
                        const tvm::Array<tvm::Expr>& output_shape,
                        const tvm::Array<tvm::Expr>& stride,
                        const tvm::Array<tvm::Expr>& padding,
                        const std::string& name = "conv2d_gemm");

}  // namespace tvm_codegen
}  // namespace onnxruntime
