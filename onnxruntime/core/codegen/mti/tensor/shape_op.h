// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <tvm/tvm.h>

#include <string>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Shape(const tvm::Tensor& X, const std::string& name = "shape");

}  // namespace tvm_codegen
}  // namespace onnxruntime
