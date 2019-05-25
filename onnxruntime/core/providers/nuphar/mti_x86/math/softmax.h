// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace nuphar_codegen {

tvm::Tensor Softmax(const tvm::Tensor& input, int64_t axis, const std::string& name = "Softmax");

}  // namespace nuphar_codegen
}  // namespace onnxruntime
