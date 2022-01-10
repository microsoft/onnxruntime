// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/te/operation.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor Concat(const tvm::Array<tvm::te::Tensor>& inputs, int64_t axis, const std::string& name = "concat");
tvm::te::Tensor ConcatSafe(const tvm::Array<tvm::te::Tensor>& inputs, int64_t axis, const std::string& name = "concat_safe");

}  // namespace tvm_codegen
}  // namespace onnxruntime
