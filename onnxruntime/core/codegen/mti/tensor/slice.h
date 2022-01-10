// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/te/operation.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor Slice(const tvm::te::Tensor& X,
                  const std::vector<int64_t>& starts,
                  const std::vector<int64_t>& ends,
                  const std::vector<int64_t>& axes,
                  const std::vector<int64_t>& steps,
                  const std::string& name = "slice");

}  // namespace tvm_codegen
}  // namespace onnxruntime
