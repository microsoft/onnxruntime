// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

// ONNX Split semantics
tvm::Array<tvm::Tensor> Split(const tvm::Tensor& X,
                              const tvm::Array<tvm::Expr>& split_sizes,
                              int64_t axis,
                              const std::string& name = "split");

// Another common Split interface
// Split with chunck indices
tvm::Array<tvm::Tensor> SplitWithIndices(const tvm::Tensor& X,
                                         const tvm::Array<tvm::Integer>& split_sizes,
                                         int64_t axis,
                                         const std::string& name = "split_with_indices");

}  // namespace tvm_codegen
}  // namespace onnxruntime
