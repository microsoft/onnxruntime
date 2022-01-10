// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/te/operation.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor Flatten(const tvm::te::Tensor& X, int64_t axis, const std::string& name = "flatten");
tvm::te::Tensor Identity(const tvm::te::Tensor& X, const std::string& name = "identity");
tvm::te::Tensor Reshape(const tvm::te::Tensor& X, const tvm::Array<tvm::PrimExpr>& new_shape, const std::string& name = "reshape");

}  // namespace tvm_codegen
}  // namespace onnxruntime
