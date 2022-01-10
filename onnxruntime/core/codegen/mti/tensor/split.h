// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/te/operation.h>

namespace onnxruntime {
namespace tvm_codegen {

// ONNX Split semantics
tvm::Array<tvm::te::Tensor> Split(const tvm::te::Tensor& X,
                              const tvm::Array<tvm::PrimExpr>& split_sizes,
                              int64_t axis,
                              const std::string& name = "split");

// Another common Split interface
// Split with chunck indices
tvm::Array<tvm::te::Tensor> SplitWithIndices(const tvm::te::Tensor& X,
                                         const tvm::Array<tvm::Integer>& split_sizes,
                                         int64_t axis,
                                         const std::string& name = "split_with_indices");

}  // namespace tvm_codegen
}  // namespace onnxruntime
