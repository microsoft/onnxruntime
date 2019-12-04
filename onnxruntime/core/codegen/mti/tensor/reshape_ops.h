// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Flatten(const tvm::Tensor& X, int64_t axis, const std::string& name = "flatten");
tvm::Tensor Identity(const tvm::Tensor& X, const std::string& name = "identity");
tvm::Tensor Reshape(const tvm::Tensor& X, const tvm::Array<tvm::Expr>& new_shape, const std::string& name = "reshape");

}  // namespace tvm_codegen
}  // namespace onnxruntime
