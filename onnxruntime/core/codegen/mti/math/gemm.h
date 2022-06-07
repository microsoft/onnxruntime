// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Gemm(const tvm::Tensor& p_A, const tvm::Tensor& p_B, const tvm::Tensor& p_C,
                 bool trans_A, bool trans_B, float alpha, float beta,
                 const std::string& name = "gemm");

}  // namespace tvm_codegen
}  // namespace onnxruntime
