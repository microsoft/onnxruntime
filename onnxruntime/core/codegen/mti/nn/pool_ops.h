// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {

// Forward declaration
struct PoolAttributes;

namespace tvm_codegen {

tvm::Tensor MaxPool(const tvm::Tensor& input,
                    const PoolAttributes& pool_attrs,
                    const tvm::Array<tvm::Expr>& output_shape,
                    const std::string& name = "max_pool");

tvm::Tensor AveragePool(const tvm::Tensor& input,
                        const PoolAttributes& pool_attrs,
                        const tvm::Array<tvm::Expr>& output_shape,
                        const std::string& name = "average_pool");

tvm::Tensor GlobalMaxPool(const tvm::Tensor& input,
                          const PoolAttributes& pool_attrs,
                          const tvm::Array<tvm::Expr>& output_shape,
                          const std::string& name = "global_max_pool");

tvm::Tensor GlobalAveragePool(const tvm::Tensor& input,
                              const PoolAttributes& pool_attrs,
                              const tvm::Array<tvm::Expr>& output_shape,
                              const std::string& name = "global_average_pool");

}  // namespace tvm_codegen
}  // namespace onnxruntime
