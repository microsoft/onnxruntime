// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace nuphar {

tvm::Tensor Scatter(const tvm::Tensor& t,
                    int64_t axis,
                    const tvm::Tensor& indices,
                    const tvm::Tensor& updates,
                    const std::string& name = "scatter");

}  // namespace nuphar
}  // namespace onnxruntime
