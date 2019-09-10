// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/nn/pool_ops.h"

#include <topi/nn/pooling.h>

namespace onnxruntime {
namespace tvm_codegen {

// TODO: topi only support 2d-pool, MaxPool1d and MaxPool3d will need to be added if necessary.
// only support version < 8 for topi doesn't come with implementation to output index tensor
tvm::Tensor MaxPool(
    const tvm::Tensor& input,
    const tvm::Array<tvm::Expr>& kernel_size,
    const tvm::Array<tvm::Expr>& stride_size,
    const tvm::Array<tvm::Expr>& padding_size,
    const std::string& layout,
    bool count_include_pad) {
  return topi::nn::pool(input, kernel_size, stride_size, padding_size,
                        topi::nn::kMaxPool,
                        false,
                        layout,
                        count_include_pad);
}

tvm::Tensor AveragePool(
    const tvm::Tensor& input,
    const tvm::Array<tvm::Expr>& kernel_size,
    const tvm::Array<tvm::Expr>& stride_size,
    const tvm::Array<tvm::Expr>& padding_size,
    const std::string& layout,
    bool count_include_pad) {
  return topi::nn::pool(input, kernel_size, stride_size, padding_size,
                        topi::nn::kAvgPool,
                        false,
                        layout,
                        count_include_pad);
}

tvm::Tensor GlobalMaxPool(
    const tvm::Tensor& input,
    const std::string& layout) {
  return topi::nn::global_pool(input,
                               topi::nn::kMaxPool,
                               layout);
}

tvm::Tensor GlobalAveragePool(
    const tvm::Tensor& input,
    const std::string& layout) {
  return topi::nn::global_pool(input,
                               topi::nn::kAvgPool,
                               layout);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
