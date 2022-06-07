// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Concat(const tvm::Array<tvm::Tensor>& inputs, int64_t axis, const std::string& name = "concat");
tvm::Tensor ConcatSafe(const tvm::Array<tvm::Tensor>& inputs, int64_t axis, const std::string& name = "concat_safe");

}  // namespace tvm_codegen
}  // namespace onnxruntime
