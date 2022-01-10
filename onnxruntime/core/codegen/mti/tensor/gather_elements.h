// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/te/operation.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor GatherElements(const tvm::te::Tensor& t,
                           int64_t axis,
                           const tvm::te::Tensor& indices,
                           const std::string& name = "gather_elements");

}  // namespace tvm_codegen
}  // namespace onnxruntime
