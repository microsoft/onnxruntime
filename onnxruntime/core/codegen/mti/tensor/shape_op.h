// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/te/operation.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor Shape(const tvm::te::Tensor& X, const std::string& name = "shape");

}  // namespace tvm_codegen
}  // namespace onnxruntime
