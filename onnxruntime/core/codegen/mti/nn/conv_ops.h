// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <tvm/te/operation.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor Conv1D(const tvm::te::Tensor& input,
                   const tvm::te::Tensor& filter,
                   const tvm::Array<tvm::PrimExpr>& output_shape,
                   const tvm::Array<tvm::PrimExpr>& stride,
                   const tvm::Array<tvm::PrimExpr>& padding,
                   const std::string& name = "conv1d");

tvm::te::Tensor Conv2D(const tvm::te::Tensor& input,
                   const tvm::te::Tensor& filter,
                   const tvm::Array<tvm::PrimExpr>& output_shape,
                   const tvm::Array<tvm::PrimExpr>& stride,
                   const tvm::Array<tvm::PrimExpr>& padding,
                   const std::string& name = "conv2d");

tvm::te::Tensor Conv2D_native(const tvm::te::Tensor& input,
                          const tvm::te::Tensor& filter,
                          const tvm::Array<tvm::PrimExpr>& output_shape,
                          const tvm::Array<tvm::PrimExpr>& stride,
                          const tvm::Array<tvm::PrimExpr>& padding,
                          const std::string& name = "conv2d_native");

tvm::te::Tensor Conv2D_gemm(const tvm::te::Tensor& input,
                        const tvm::te::Tensor& filter,
                        const tvm::Array<tvm::PrimExpr>& output_shape,
                        const tvm::Array<tvm::PrimExpr>& stride,
                        const tvm::Array<tvm::PrimExpr>& padding,
                        const std::string& name = "conv2d_gemm");

}  // namespace tvm_codegen
}  // namespace onnxruntime
