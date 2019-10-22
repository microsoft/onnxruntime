// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Cast(const tvm::Tensor& X, tvm::Type type, const std::string& name = "cast");
tvm::Tensor CastToUInt8Bool(const tvm::Tensor& X, const std::string& name = "cast_uint8_bool");

}  // namespace tvm_codegen
}  // namespace onnxruntime
