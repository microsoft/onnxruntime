// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Slice(const tvm::Tensor& X,
                  const tvm::Array<tvm::Integer>& starts,
                  const tvm::Array<tvm::Integer>& ends,
                  const std::string& name = "slice");

}  // namespace tvm_codegen
}  // namespace onnxruntime
