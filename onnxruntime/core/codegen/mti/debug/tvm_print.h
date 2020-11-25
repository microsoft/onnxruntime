// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Array<tvm::Tensor> PrintTVMTensorExtern(
    const tvm::Tensor& X,
    const std::string& name = "PrintTVM2DTensorExtern");

tvm::Tensor PrintImmutable(const tvm::Tensor& X);

void Print(tvm::Tensor& X);

}  // namespace tvm_codegen
}  // namespace onnxruntime
