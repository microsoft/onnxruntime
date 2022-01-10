// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/te/operation.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor Transpose(const tvm::te::Tensor& X,
                      const tvm::Array<tvm::Integer>& axes,
                      const std::string& name = "transpose");

}  // namespace tvm_codegen
}  // namespace onnxruntime
