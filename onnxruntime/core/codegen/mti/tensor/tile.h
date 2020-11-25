// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Tile(const tvm::Tensor& t,
                 const std::vector<int64_t>& repeats,
                 const std::string& name = "tile");

}  // namespace tvm_codegen
}  // namespace onnxruntime
