// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace nuphar {

tvm::Tensor
QMatMulAsymmetricAVX2(const tvm::Tensor& transposed_quantized_param,
                      const tvm::Tensor& Q_X,
                      const tvm::Array<tvm::Expr>& output_shape,
                      int input_dim,
                      int embed_dim,
                      const std::string& name = "QMatMulAsymmetricAVX2");

tvm::Tensor
QMatMulAsymmetricMKL(const tvm::Tensor& transposed_quantized_param,
                     const tvm::Tensor& Q_X,
                     const tvm::Array<tvm::Expr>& output_shape,
                     int input_dim,
                     int embed_dim,
                     const std::string& name = "QMatMulAsymmetricMKL");

}  // namespace nuphar
}  // namespace onnxruntime
