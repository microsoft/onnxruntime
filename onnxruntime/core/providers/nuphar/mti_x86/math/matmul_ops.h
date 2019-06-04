// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace nuphar_codegen {

tvm::Tensor MatMul2D(const tvm::Tensor& A, const tvm::Tensor& B, bool trans_a = false, bool trans_b = false, const std::string& name = "matmul2d");

bool MatMulExternCpu(
    const tvm::Tensor& A,
    const tvm::Tensor& B,
    tvm::Tensor& Y,
    bool trans_a = false,
    bool trans_b = false,
    const std::string& name = "matmul_extern_cpu");

tvm::Tensor MatMul(const tvm::Tensor& A, const tvm::Tensor& B, const std::string& name);

}  // namespace nuphar_codegen
}  // namespace onnxruntime
