// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

// ONNX Pad semantics
tvm::Tensor Pad(const tvm::Tensor& t,
                const tvm::Array<tvm::Expr>& pad_before,
                const tvm::Array<tvm::Expr>& pad_after,
                float pad_value = 0.0f,
                const std::string& mode = "constant",
                const std::string& name = "pad");

// Other common Pad interfaces
// Pad for a given shape
tvm::Tensor Pad(const tvm::Tensor& t,
                const tvm::Array<tvm::Expr>& output_shape,
                const tvm::Expr& pad_value,
                const std::string& name = "pad");

// Pad for the last dim only.
// This is widely used for weight layout to guard alignment
tvm::Tensor PadLastDim(const tvm::Tensor& t,
                       const int32_t align_size,
                       const tvm::Expr& pad_value,
                       const std::string& name = "pad_last_dim");

}  // namespace tvm_codegen
}  // namespace onnxruntime
