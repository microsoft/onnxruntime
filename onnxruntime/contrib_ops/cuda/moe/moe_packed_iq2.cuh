// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/quantization/gguf_quant_cuda.cuh"

namespace onnxruntime::contrib::cuda {

// FC1 and FC2 retain their code-load/sign specializations while sharing the
// dense projection's canonical grid and complete-operand BF16 reconstruction.
template <bool kCodesWord = true, bool kSignBits = false, bool kAlignedRecordWindow = false>
__device__ __forceinline__ void MoEPackedIq2Fragment(
    const uint8_t* row, int offset, uint32_t (&low)[4], uint32_t (&high)[4]) {
  uint4 lo, hi;
  gguf_cuda::DecodeIq2SubBlock16Bf16<kCodesWord, kSignBits, kAlignedRecordWindow>(
      row, offset / 32, (offset / 16) & 1, lo, hi);
  low[0] = lo.x;
  low[1] = lo.y;
  low[2] = lo.z;
  low[3] = lo.w;
  high[0] = hi.x;
  high[1] = hi.y;
  high[2] = hi.z;
  high[3] = hi.w;
}

}  // namespace onnxruntime::contrib::cuda
