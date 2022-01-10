// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/te/operation.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor CastTensor(const tvm::te::Tensor& X, tvm::DataType type, const std::string& name = "cast");
tvm::te::Tensor CastTensorToUInt8Bool(const tvm::te::Tensor& X, const std::string& name = "cast_uint8_bool");

}  // namespace tvm_codegen
}  // namespace onnxruntime
