// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/te/operation.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor Where(const tvm::te::Tensor& B,
                  const tvm::te::Tensor& X,
                  const tvm::te::Tensor& Y,
                  const std::string& name = "where");

}  // namespace tvm_codegen
}  // namespace onnxruntime
