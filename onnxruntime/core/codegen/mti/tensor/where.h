// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Where(const tvm::Tensor& B,
                  const tvm::Tensor& X,
                  const tvm::Tensor& Y,
                  const std::string& name = "where");

}  // namespace tvm_codegen
}  // namespace onnxruntime
