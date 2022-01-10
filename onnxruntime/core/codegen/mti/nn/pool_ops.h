// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/te/operation.h>

namespace onnxruntime {

// Forward declaration
struct PoolAttributes;

namespace tvm_codegen {

tvm::te::Tensor MaxPool(const tvm::te::Tensor& input,
                    const PoolAttributes& pool_attrs,
                    const tvm::Array<tvm::PrimExpr>& output_shape,
                    const std::string& name = "max_pool");

tvm::te::Tensor AveragePool(const tvm::te::Tensor& input,
                        const PoolAttributes& pool_attrs,
                        const tvm::Array<tvm::PrimExpr>& output_shape,
                        const std::string& name = "average_pool");

tvm::te::Tensor GlobalMaxPool(const tvm::te::Tensor& input,
                          const PoolAttributes& pool_attrs,
                          const tvm::Array<tvm::PrimExpr>& output_shape,
                          const std::string& name = "global_max_pool");

tvm::te::Tensor GlobalAveragePool(const tvm::te::Tensor& input,
                              const PoolAttributes& pool_attrs,
                              const tvm::Array<tvm::PrimExpr>& output_shape,
                              const std::string& name = "global_average_pool");

}  // namespace tvm_codegen
}  // namespace onnxruntime
