// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor MaxPool(const tvm::Tensor& input,
                    const tvm::Array<tvm::Expr>& kernel_size,
                    const tvm::Array<tvm::Expr>& stride_size,
                    const tvm::Array<tvm::Expr>& padding_size,
                    const std::string& layout,
                    bool count_include_pad);

tvm::Tensor AveragePool(const tvm::Tensor& input,
                        const tvm::Array<tvm::Expr>& kernel_size,
                        const tvm::Array<tvm::Expr>& stride_size,
                        const tvm::Array<tvm::Expr>& padding_size,
                        const std::string& layout,
                        bool count_include_pad);

tvm::Tensor GlobalMaxPool(const tvm::Tensor& input,
                          const std::string& layout);

tvm::Tensor GlobalAveragePool(const tvm::Tensor& input,
                              const std::string& layout);

}  // namespace tvm_codegen
}  // namespace onnxruntime
