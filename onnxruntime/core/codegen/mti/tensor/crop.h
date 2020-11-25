// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Crop(const tvm::Tensor& t,
                 const tvm::Array<tvm::Expr>& border,
                 const tvm::Array<tvm::Expr>& scale = {},
                 const std::string& name = "crop");

}  // namespace tvm_codegen
}  // namespace onnxruntime
