// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace nuphar {

tvm::Tensor Softmax(const tvm::Tensor& input, int64_t axis, int64_t vector_width, const std::string& name = "Softmax");

}  // namespace nuphar
}  // namespace onnxruntime
