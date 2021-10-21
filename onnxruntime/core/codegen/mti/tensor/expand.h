// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Expand(const tvm::Tensor& X, const tvm::Array<tvm::Expr>& new_shape, const std::string& name = "expand");

}  // namespace tvm_codegen
}  // namespace onnxruntime
