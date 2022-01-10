// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <tvm/te/tensor.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Array<tvm::te::Tensor> PrintTVMTensorExtern(
    const tvm::te::Tensor& X,
    const std::string& name = "PrintTVM2DTensorExtern");

tvm::te::Tensor PrintImmutable(const tvm::te::Tensor& X);

void Print(tvm::te::Tensor& X);

}  // namespace tvm_codegen
}  // namespace onnxruntime
