// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// ROCm implementation of MatMulNBits.
//
// Strategy (matches llama.cpp's gfx900 approach):
//   - For every batch size: dequantize the packed int4/int8 weight matrix to
//     fp32/fp16 on the GPU (pure HIP, no CUTLASS), then call hipBLAS GEMM.
//
// This is the correct path for gfx900 because:
//   (a) gfx900 has no native dp4a instruction, so pure HIP dequant+GEMM is
//       actually faster than the dp4a MMQ tiling approach for dense matmul.
//   (b) The CUTLASS fpA_intB GEMM used by the CUDA EP requires NVIDIA SM>=75
//       and the hipified llm/ tree is excluded from ROCm builds entirely.

#pragma once

#include "core/common/safeint.h"
#include "core/providers/rocm/rocm_kernel.h"
#include "core/providers/rocm/shared_inc/fpgeneric.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

using namespace onnxruntime::rocm;

template <typename T>
class MatMulNBits final : public RocmKernel {
 public:
  MatMulNBits(const OpKernelInfo& info) : RocmKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("K", &K_).IsOK());
    ORT_ENFORCE(info.GetAttr<int64_t>("N", &N_).IsOK());
    ORT_ENFORCE(info.GetAttr<int64_t>("block_size", &block_size_).IsOK());
    ORT_ENFORCE(info.GetAttr<int64_t>("bits", &nbits_).IsOK());

    constexpr int kInputIndexZeroPoints = 3;
    constexpr int kInputIndexGroupIndex = 4;
    constexpr int kInputIndexBias = 5;
    has_zero_points_ = info.GetInputCount() > kInputIndexZeroPoints &&
                       info.node().InputDefs()[kInputIndexZeroPoints]->Exists();
    has_g_idx_ = info.GetInputCount() > kInputIndexGroupIndex &&
                 info.node().InputDefs()[kInputIndexGroupIndex]->Exists();
    has_bias_ = info.GetInputCount() > kInputIndexBias &&
                info.node().InputDefs()[kInputIndexBias]->Exists();
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t nbits_;
  bool has_zero_points_{false};
  bool has_g_idx_{false};
  bool has_bias_{false};
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
