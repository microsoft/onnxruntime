// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/te/operation.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor Expand(const tvm::te::Tensor& X, const tvm::Array<tvm::PrimExpr>& new_shape, const std::string& name = "expand");

}  // namespace tvm_codegen
}  // namespace onnxruntime
