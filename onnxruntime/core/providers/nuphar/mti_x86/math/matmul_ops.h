// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace nuphar {

tvm::Tensor MatMul2D(const tvm::Tensor& A, const tvm::Tensor& B, bool trans_a = false, bool trans_b = false, const std::string& name = "matmul2d");

bool GemmExternCpu(
    const tvm::Tensor& A,
    const tvm::Tensor& B,
    tvm::Tensor& Y,
    bool trans_a = false,
    bool trans_b = false,
    const std::string& name = "matmul_extern_cpu");

bool MatMulExternCpu(
    const tvm::Tensor& A,
    const tvm::Tensor& B,
    tvm::Tensor& Y,
    const std::vector<int32_t>* permute_A,
    const std::vector<int32_t>* permute_B,
    const std::string& name = "matmul_permute_extern_cpu");

bool CanPermuteBeFusedInMatMul(const std::vector<int32_t>& perm);

tvm::Tensor MatMul(const tvm::Tensor& A, const tvm::Tensor& B, const std::string& name);

}  // namespace nuphar
}  // namespace onnxruntime
