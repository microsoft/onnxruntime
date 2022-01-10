// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/te/operation.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor Crop(const tvm::te::Tensor& t,
                 const tvm::Array<tvm::PrimExpr>& border,
                 const tvm::Array<tvm::PrimExpr>& scale = {},
                 const std::string& name = "crop");

}  // namespace tvm_codegen
}  // namespace onnxruntime
