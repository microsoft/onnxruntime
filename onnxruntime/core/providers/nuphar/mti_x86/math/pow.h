// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace nuphar {

tvm::Tensor Pow(tvm::Tensor A, tvm::Tensor B, const std::string& name = "pow");
tvm::Tensor Pow(tvm::Expr A, tvm::Tensor B, const std::string& name = "pow");
tvm::Tensor Pow(tvm::Tensor A, tvm::Expr B, const std::string& name = "pow");

}  // namespace nuphar
}  // namespace onnxruntime
