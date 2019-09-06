// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace nuphar {

tvm::Tensor
IMatMulExternAVX2(const tvm::Tensor& transposed_quantized_param,
                  const tvm::Tensor& Q_X,
                  const tvm::Array<tvm::Expr>& output_shape,
                  int input_dim,
                  int embed_dim,
                  const std::string& name = "IMatMulExternAVX2");

tvm::Tensor
IMatMulExternMKL(const tvm::Tensor& transposed_quantized_param,
                 const tvm::Tensor& Q_X,
                 const tvm::Array<tvm::Expr>& output_shape,
                 int input_dim,
                 int embed_dim,
                 const std::string& name = "IMatMulExternMKL");

}  // namespace nuphar
}  // namespace onnxruntime
