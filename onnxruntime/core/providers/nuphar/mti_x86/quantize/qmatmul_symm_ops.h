// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace nuphar_codegen {

tvm::Array<tvm::Tensor>
QMatMulSymmetricMKL(const tvm::Tensor& transposed_quantized_param,
                    const tvm::Tensor& Q_X,
                    const tvm::Expr& batch_seq_dim,
                    int input_dim,
                    int embed_dim,
                    const std::string& name = "QMatMulSymmetricMKL");

tvm::Array<tvm::Tensor>
QMatMulSymmetricAVX2(const tvm::Tensor& transposed_quantized_param,
                     const tvm::Tensor& Q_X,
                     const tvm::Expr& batch_seq_dim,
                     int input_dim,
                     int embed_dim,
                     const std::string& name = "QMatMulSymmetricAVX2");

}  // namespace nuphar_codegen
}  // namespace onnxruntime
