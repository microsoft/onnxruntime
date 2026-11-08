// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace onnxruntime::contrib::cuda::packed_projection {

using bf16 = __nv_bfloat16;
__device__ __forceinline__ float B2F(bf16 v) { return __bfloat162float(v); }

// Source-exact packed BF16 MMA helpers shared by dense and routed projections.
// IQ2_XXS codebook: iq2xxs_grid in ggml/src/ggml-common.h from
// https://github.com/ggml-org/llama.cpp/tree/3de52ac38818978d6da8f386eda43a856e53714a
// (byte magnitudes re-encoded as BF16 pairs).
//
// MIT License
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
static __device__ const uint32_t kIq2xxsGridBf16[256][4] = {
    {0x41004100u, 0x41004100u, 0x41004100u, 0x41004100u},
    {0x4100422cu, 0x41004100u, 0x41004100u, 0x41004100u},
    {0x41c841c8u, 0x41004100u, 0x41004100u, 0x41004100u},
    {0x422c4100u, 0x41004100u, 0x41004100u, 0x41004100u},
    {0x422c422cu, 0x41004100u, 0x41004100u, 0x41004100u},
    {0x410041c8u, 0x410041c8u, 0x41004100u, 0x41004100u},
    {0x41c84100u, 0x410041c8u, 0x41004100u, 0x41004100u},
    {0x41004100u, 0x4100422cu, 0x41004100u, 0x41004100u},
    {0x4100422cu, 0x4100422cu, 0x41004100u, 0x41004100u},
    {0x422c4100u, 0x4100422cu, 0x41004100u, 0x41004100u},
    {0x422c422cu, 0x4100422cu, 0x41004100u, 0x41004100u},
    {0x410041c8u, 0x41c84100u, 0x41004100u, 0x41004100u},
    {0x41c84100u, 0x41c84100u, 0x41004100u, 0x41004100u},
    {0x41004100u, 0x41c841c8u, 0x41004100u, 0x41004100u},
    {0x422c4100u, 0x41c841c8u, 0x41004100u, 0x41004100u},
    {0x410041c8u, 0x41c8422cu, 0x41004100u, 0x41004100u},
    {0x41c84100u, 0x41c8422cu, 0x41004100u, 0x41004100u},
    {0x41004100u, 0x422c4100u, 0x41004100u, 0x41004100u},
    {0x4100422cu, 0x422c4100u, 0x41004100u, 0x41004100u},
    {0x422c422cu, 0x422c4100u, 0x41004100u, 0x41004100u},
    {0x4100422cu, 0x422c422cu, 0x41004100u, 0x41004100u},
    {0x410041c8u, 0x41004100u, 0x410041c8u, 0x41004100u},
    {0x41c84100u, 0x41004100u, 0x410041c8u, 0x41004100u},
    {0x41004100u, 0x410041c8u, 0x410041c8u, 0x41004100u},
    {0x41c841c8u, 0x410041c8u, 0x410041c8u, 0x41004100u},
    {0x41004100u, 0x41c84100u, 0x410041c8u, 0x41004100u},
    {0x41c84100u, 0x422c4100u, 0x410041c8u, 0x41004100u},
    {0x422c4100u, 0x422c41c8u, 0x410041c8u, 0x41004100u},
    {0x41004100u, 0x41004100u, 0x4100422cu, 0x41004100u},
    {0x4100422cu, 0x41004100u, 0x4100422cu, 0x41004100u},
    {0x4100422cu, 0x4100422cu, 0x4100422cu, 0x41004100u},
    {0x4100422cu, 0x422c4100u, 0x4100422cu, 0x41004100u},
    {0x410041c8u, 0x41004100u, 0x41c84100u, 0x41004100u},
    {0x41c84100u, 0x41004100u, 0x41c84100u, 0x41004100u},
    {0x41004100u, 0x410041c8u, 0x41c84100u, 0x41004100u},
    {0x410041c8u, 0x4100422cu, 0x41c84100u, 0x41004100u},
    {0x41c84100u, 0x4100422cu, 0x41c84100u, 0x41004100u},
    {0x41004100u, 0x41c84100u, 0x41c84100u, 0x41004100u},
    {0x4100422cu, 0x41c84100u, 0x41c84100u, 0x41004100u},
    {0x422c4100u, 0x41c84100u, 0x41c84100u, 0x41004100u},
    {0x41004100u, 0x41c8422cu, 0x41c84100u, 0x41004100u},
    {0x410041c8u, 0x422c4100u, 0x41c84100u, 0x41004100u},
    {0x41c84100u, 0x422c4100u, 0x41c84100u, 0x41004100u},
    {0x41004100u, 0x422c41c8u, 0x41c84100u, 0x41004100u},
    {0x41c84100u, 0x422c422cu, 0x41c84100u, 0x41004100u},
    {0x41004100u, 0x41004100u, 0x41c841c8u, 0x41004100u},
    {0x4100422cu, 0x41004100u, 0x41c841c8u, 0x41004100u},
    {0x422c4100u, 0x41004100u, 0x41c841c8u, 0x41004100u},
    {0x41004100u, 0x4100422cu, 0x41c841c8u, 0x41004100u},
    {0x41c8422cu, 0x41c84100u, 0x41c841c8u, 0x41004100u},
    {0x422c41c8u, 0x41c8422cu, 0x41c841c8u, 0x41004100u},
    {0x41004100u, 0x422c4100u, 0x41c841c8u, 0x41004100u},
    {0x410041c8u, 0x422c41c8u, 0x41c841c8u, 0x41004100u},
    {0x422c41c8u, 0x41004100u, 0x41c8422cu, 0x41004100u},
    {0x41004100u, 0x410041c8u, 0x41c8422cu, 0x41004100u},
    {0x41004100u, 0x41c84100u, 0x41c8422cu, 0x41004100u},
    {0x41c84100u, 0x422c4100u, 0x41c8422cu, 0x41004100u},
    {0x41c84100u, 0x422c422cu, 0x41c8422cu, 0x41004100u},
    {0x41004100u, 0x41004100u, 0x422c4100u, 0x41004100u},
    {0x41c841c8u, 0x41004100u, 0x422c4100u, 0x41004100u},
    {0x422c4100u, 0x41004100u, 0x422c4100u, 0x41004100u},
    {0x41c84100u, 0x410041c8u, 0x422c4100u, 0x41004100u},
    {0x422c4100u, 0x4100422cu, 0x422c4100u, 0x41004100u},
    {0x410041c8u, 0x41c84100u, 0x422c4100u, 0x41004100u},
    {0x41c84100u, 0x41c84100u, 0x422c4100u, 0x41004100u},
    {0x41004100u, 0x41c841c8u, 0x422c4100u, 0x41004100u},
    {0x4100422cu, 0x41c841c8u, 0x422c4100u, 0x41004100u},
    {0x422c4100u, 0x422c4100u, 0x422c4100u, 0x41004100u},
    {0x41c84100u, 0x41004100u, 0x422c41c8u, 0x41004100u},
    {0x41004100u, 0x41c84100u, 0x422c41c8u, 0x41004100u},
    {0x4100422cu, 0x41004100u, 0x422c422cu, 0x41004100u},
    {0x41c84100u, 0x410041c8u, 0x422c422cu, 0x41004100u},
    {0x410041c8u, 0x41004100u, 0x41004100u, 0x410041c8u},
    {0x41c84100u, 0x41004100u, 0x41004100u, 0x410041c8u},
    {0x41004100u, 0x410041c8u, 0x41004100u, 0x410041c8u},
    {0x410041c8u, 0x4100422cu, 0x41004100u, 0x410041c8u},
    {0x41004100u, 0x41c84100u, 0x41004100u, 0x410041c8u},
    {0x41004100u, 0x41c8422cu, 0x41004100u, 0x410041c8u},
    {0x41c84100u, 0x422c4100u, 0x41004100u, 0x410041c8u},
    {0x41004100u, 0x422c41c8u, 0x41004100u, 0x410041c8u},
    {0x41c841c8u, 0x422c41c8u, 0x41004100u, 0x410041c8u},
    {0x41004100u, 0x41004100u, 0x410041c8u, 0x410041c8u},
    {0x422c4100u, 0x41004100u, 0x410041c8u, 0x410041c8u},
    {0x41004100u, 0x4100422cu, 0x410041c8u, 0x410041c8u},
    {0x41004100u, 0x41c841c8u, 0x410041c8u, 0x410041c8u},
    {0x422c422cu, 0x41c841c8u, 0x410041c8u, 0x410041c8u},
    {0x41004100u, 0x422c4100u, 0x410041c8u, 0x410041c8u},
    {0x41c84100u, 0x4100422cu, 0x4100422cu, 0x410041c8u},
    {0x41c841c8u, 0x41c84100u, 0x4100422cu, 0x410041c8u},
    {0x41004100u, 0x41004100u, 0x41c84100u, 0x410041c8u},
    {0x422c4100u, 0x41004100u, 0x41c84100u, 0x410041c8u},
    {0x41004100u, 0x4100422cu, 0x41c84100u, 0x410041c8u},
    {0x41c841c8u, 0x4100422cu, 0x41c84100u, 0x410041c8u},
    {0x422c41c8u, 0x41c84100u, 0x41c84100u, 0x410041c8u},
    {0x41004100u, 0x422c4100u, 0x41c84100u, 0x410041c8u},
    {0x422c4100u, 0x410041c8u, 0x41c841c8u, 0x410041c8u},
    {0x4100422cu, 0x41c8422cu, 0x41c841c8u, 0x410041c8u},
    {0x41004100u, 0x41004100u, 0x41c8422cu, 0x410041c8u},
    {0x41c8422cu, 0x410041c8u, 0x41c8422cu, 0x410041c8u},
    {0x410041c8u, 0x41004100u, 0x422c4100u, 0x410041c8u},
    {0x41c84100u, 0x41004100u, 0x422c4100u, 0x410041c8u},
    {0x41004100u, 0x410041c8u, 0x422c4100u, 0x410041c8u},
    {0x41004100u, 0x41c84100u, 0x422c4100u, 0x410041c8u},
    {0x410041c8u, 0x422c4100u, 0x422c4100u, 0x410041c8u},
    {0x41004100u, 0x41004100u, 0x422c41c8u, 0x410041c8u},
    {0x41c841c8u, 0x41004100u, 0x422c41c8u, 0x410041c8u},
    {0x41004100u, 0x422c422cu, 0x422c41c8u, 0x410041c8u},
    {0x410041c8u, 0x41c841c8u, 0x422c422cu, 0x410041c8u},
    {0x41004100u, 0x41004100u, 0x41004100u, 0x4100422cu},
    {0x4100422cu, 0x41004100u, 0x41004100u, 0x4100422cu},
    {0x422c422cu, 0x41004100u, 0x41004100u, 0x4100422cu},
    {0x41c84100u, 0x41c84100u, 0x41004100u, 0x4100422cu},
    {0x410041c8u, 0x41c8422cu, 0x41004100u, 0x4100422cu},
    {0x41004100u, 0x422c4100u, 0x41004100u, 0x4100422cu},
    {0x4100422cu, 0x422c4100u, 0x41004100u, 0x4100422cu},
    {0x422c41c8u, 0x4100422cu, 0x410041c8u, 0x4100422cu},
    {0x422c4100u, 0x41c84100u, 0x410041c8u, 0x4100422cu},
    {0x41004100u, 0x41004100u, 0x4100422cu, 0x4100422cu},
    {0x4100422cu, 0x41004100u, 0x4100422cu, 0x4100422cu},
    {0x410041c8u, 0x41004100u, 0x41c84100u, 0x4100422cu},
    {0x41c84100u, 0x41004100u, 0x41c84100u, 0x4100422cu},
    {0x41004100u, 0x410041c8u, 0x41c84100u, 0x4100422cu},
    {0x41004100u, 0x41c84100u, 0x41c84100u, 0x4100422cu},
    {0x41c8422cu, 0x41c841c8u, 0x41c84100u, 0x4100422cu},
    {0x41004100u, 0x41004100u, 0x41c841c8u, 0x4100422cu},
    {0x410041c8u, 0x41c84100u, 0x41c841c8u, 0x4100422cu},
    {0x41c84100u, 0x41c8422cu, 0x41c841c8u, 0x4100422cu},
    {0x41004100u, 0x422c41c8u, 0x41c8422cu, 0x4100422cu},
    {0x422c4100u, 0x41004100u, 0x422c4100u, 0x4100422cu},
    {0x41004100u, 0x4100422cu, 0x422c4100u, 0x4100422cu},
    {0x41c84100u, 0x422c41c8u, 0x422c4100u, 0x4100422cu},
    {0x41c84100u, 0x41c84100u, 0x422c422cu, 0x4100422cu},
    {0x410041c8u, 0x41004100u, 0x41004100u, 0x41c84100u},
    {0x41c84100u, 0x41004100u, 0x41004100u, 0x41c84100u},
    {0x41004100u, 0x410041c8u, 0x41004100u, 0x41c84100u},
    {0x422c4100u, 0x410041c8u, 0x41004100u, 0x41c84100u},
    {0x410041c8u, 0x4100422cu, 0x41004100u, 0x41c84100u},
    {0x41c84100u, 0x4100422cu, 0x41004100u, 0x41c84100u},
    {0x41004100u, 0x41c84100u, 0x41004100u, 0x41c84100u},
    {0x422c4100u, 0x41c84100u, 0x41004100u, 0x41c84100u},
    {0x41c8422cu, 0x41c841c8u, 0x41004100u, 0x41c84100u},
    {0x41004100u, 0x41c8422cu, 0x41004100u, 0x41c84100u},
    {0x410041c8u, 0x422c4100u, 0x41004100u, 0x41c84100u},
    {0x41c84100u, 0x422c4100u, 0x41004100u, 0x41c84100u},
    {0x41004100u, 0x422c41c8u, 0x41004100u, 0x41c84100u},
    {0x41004100u, 0x41004100u, 0x410041c8u, 0x41c84100u},
    {0x41004100u, 0x4100422cu, 0x410041c8u, 0x41c84100u},
    {0x410041c8u, 0x41c8422cu, 0x410041c8u, 0x41c84100u},
    {0x41004100u, 0x422c4100u, 0x410041c8u, 0x41c84100u},
    {0x41c841c8u, 0x422c4100u, 0x410041c8u, 0x41c84100u},
    {0x410041c8u, 0x41004100u, 0x4100422cu, 0x41c84100u},
    {0x41004100u, 0x410041c8u, 0x4100422cu, 0x41c84100u},
    {0x422c4100u, 0x41c84100u, 0x4100422cu, 0x41c84100u},
    {0x41c8422cu, 0x41c841c8u, 0x4100422cu, 0x41c84100u},
    {0x422c4100u, 0x41c8422cu, 0x4100422cu, 0x41c84100u},
    {0x41004100u, 0x41004100u, 0x41c84100u, 0x41c84100u},
    {0x422c4100u, 0x41004100u, 0x41c84100u, 0x41c84100u},
    {0x41004100u, 0x4100422cu, 0x41c84100u, 0x41c84100u},
    {0x41004100u, 0x422c4100u, 0x41c84100u, 0x41c84100u},
    {0x422c41c8u, 0x422c41c8u, 0x41c84100u, 0x41c84100u},
    {0x4100422cu, 0x410041c8u, 0x41c841c8u, 0x41c84100u},
    {0x41c84100u, 0x4100422cu, 0x41c841c8u, 0x41c84100u},
    {0x41004100u, 0x41004100u, 0x41c8422cu, 0x41c84100u},
    {0x410041c8u, 0x41004100u, 0x422c4100u, 0x41c84100u},
    {0x41c84100u, 0x41004100u, 0x422c4100u, 0x41c84100u},
    {0x41004100u, 0x410041c8u, 0x422c4100u, 0x41c84100u},
    {0x41004100u, 0x41c84100u, 0x422c4100u, 0x41c84100u},
    {0x41c841c8u, 0x41c84100u, 0x422c4100u, 0x41c84100u},
    {0x41004100u, 0x41004100u, 0x422c41c8u, 0x41c84100u},
    {0x422c4100u, 0x41c841c8u, 0x422c41c8u, 0x41c84100u},
    {0x410041c8u, 0x41c8422cu, 0x422c41c8u, 0x41c84100u},
    {0x4100422cu, 0x422c4100u, 0x422c41c8u, 0x41c84100u},
    {0x41c841c8u, 0x41c84100u, 0x422c422cu, 0x41c84100u},
    {0x41004100u, 0x422c41c8u, 0x422c422cu, 0x41c84100u},
    {0x41004100u, 0x41004100u, 0x41004100u, 0x41c841c8u},
    {0x422c4100u, 0x41004100u, 0x41004100u, 0x41c841c8u},
    {0x410041c8u, 0x410041c8u, 0x41004100u, 0x41c841c8u},
    {0x422c41c8u, 0x410041c8u, 0x41004100u, 0x41c841c8u},
    {0x41004100u, 0x4100422cu, 0x41004100u, 0x41c841c8u},
    {0x41004100u, 0x422c4100u, 0x41004100u, 0x41c841c8u},
    {0x422c4100u, 0x422c4100u, 0x41004100u, 0x41c841c8u},
    {0x41c84100u, 0x41004100u, 0x410041c8u, 0x41c841c8u},
    {0x4100422cu, 0x41c84100u, 0x410041c8u, 0x41c841c8u},
    {0x41c84100u, 0x422c422cu, 0x410041c8u, 0x41c841c8u},
    {0x410041c8u, 0x422c41c8u, 0x4100422cu, 0x41c841c8u},
    {0x41004100u, 0x422c41c8u, 0x41c84100u, 0x41c841c8u},
    {0x4100422cu, 0x422c41c8u, 0x41c84100u, 0x41c841c8u},
    {0x422c422cu, 0x41004100u, 0x41c841c8u, 0x41c841c8u},
    {0x410041c8u, 0x41004100u, 0x41c8422cu, 0x41c841c8u},
    {0x41c84100u, 0x41c841c8u, 0x41c8422cu, 0x41c841c8u},
    {0x41004100u, 0x41004100u, 0x422c4100u, 0x41c841c8u},
    {0x410041c8u, 0x410041c8u, 0x422c4100u, 0x41c841c8u},
    {0x422c41c8u, 0x410041c8u, 0x422c4100u, 0x41c841c8u},
    {0x41c84100u, 0x41c8422cu, 0x422c4100u, 0x41c841c8u},
    {0x41004100u, 0x41c84100u, 0x422c41c8u, 0x41c841c8u},
    {0x422c4100u, 0x41004100u, 0x422c422cu, 0x41c841c8u},
    {0x41c84100u, 0x41004100u, 0x41004100u, 0x41c8422cu},
    {0x41004100u, 0x410041c8u, 0x41004100u, 0x41c8422cu},
    {0x41004100u, 0x41c84100u, 0x41004100u, 0x41c8422cu},
    {0x422c4100u, 0x41c8422cu, 0x41004100u, 0x41c8422cu},
    {0x41004100u, 0x41004100u, 0x410041c8u, 0x41c8422cu},
    {0x41c841c8u, 0x41c841c8u, 0x410041c8u, 0x41c8422cu},
    {0x422c4100u, 0x410041c8u, 0x4100422cu, 0x41c8422cu},
    {0x41004100u, 0x41c8422cu, 0x4100422cu, 0x41c8422cu},
    {0x41004100u, 0x41004100u, 0x41c84100u, 0x41c8422cu},
    {0x41c841c8u, 0x41004100u, 0x41c84100u, 0x41c8422cu},
    {0x41004100u, 0x410041c8u, 0x41c841c8u, 0x41c8422cu},
    {0x4100422cu, 0x410041c8u, 0x41c841c8u, 0x41c8422cu},
    {0x41c84100u, 0x422c4100u, 0x41c841c8u, 0x41c8422cu},
    {0x4100422cu, 0x41c84100u, 0x422c4100u, 0x41c8422cu},
    {0x41004100u, 0x41004100u, 0x41004100u, 0x422c4100u},
    {0x4100422cu, 0x41004100u, 0x41004100u, 0x422c4100u},
    {0x422c422cu, 0x41004100u, 0x41004100u, 0x422c4100u},
    {0x410041c8u, 0x41c84100u, 0x41004100u, 0x422c4100u},
    {0x4100422cu, 0x422c4100u, 0x41004100u, 0x422c4100u},
    {0x41c84100u, 0x41004100u, 0x410041c8u, 0x422c4100u},
    {0x422c4100u, 0x410041c8u, 0x410041c8u, 0x422c4100u},
    {0x41004100u, 0x41c84100u, 0x410041c8u, 0x422c4100u},
    {0x410041c8u, 0x410041c8u, 0x4100422cu, 0x422c4100u},
    {0x410041c8u, 0x41004100u, 0x41c84100u, 0x422c4100u},
    {0x41c84100u, 0x41004100u, 0x41c84100u, 0x422c4100u},
    {0x41004100u, 0x410041c8u, 0x41c84100u, 0x422c4100u},
    {0x41c841c8u, 0x410041c8u, 0x41c84100u, 0x422c4100u},
    {0x41004100u, 0x41c84100u, 0x41c84100u, 0x422c4100u},
    {0x41004100u, 0x41c8422cu, 0x41c84100u, 0x422c4100u},
    {0x41004100u, 0x41004100u, 0x41c841c8u, 0x422c4100u},
    {0x41c8422cu, 0x41c84100u, 0x41c841c8u, 0x422c4100u},
    {0x41c84100u, 0x422c41c8u, 0x41c841c8u, 0x422c4100u},
    {0x422c41c8u, 0x41004100u, 0x41c8422cu, 0x422c4100u},
    {0x41004100u, 0x41c84100u, 0x41c8422cu, 0x422c4100u},
    {0x41004100u, 0x41c8422cu, 0x41c8422cu, 0x422c4100u},
    {0x4100422cu, 0x41004100u, 0x422c4100u, 0x422c4100u},
    {0x41c84100u, 0x41004100u, 0x422c41c8u, 0x422c4100u},
    {0x410041c8u, 0x410041c8u, 0x422c422cu, 0x422c4100u},
    {0x41c84100u, 0x41004100u, 0x41004100u, 0x422c41c8u},
    {0x41004100u, 0x410041c8u, 0x41004100u, 0x422c41c8u},
    {0x41c84100u, 0x4100422cu, 0x41004100u, 0x422c41c8u},
    {0x41004100u, 0x41c84100u, 0x41004100u, 0x422c41c8u},
    {0x410041c8u, 0x422c422cu, 0x41004100u, 0x422c41c8u},
    {0x41c8422cu, 0x410041c8u, 0x410041c8u, 0x422c41c8u},
    {0x41004100u, 0x422c4100u, 0x410041c8u, 0x422c41c8u},
    {0x41c841c8u, 0x41c84100u, 0x4100422cu, 0x422c41c8u},
    {0x41004100u, 0x41004100u, 0x41c84100u, 0x422c41c8u},
    {0x4100422cu, 0x4100422cu, 0x41c84100u, 0x422c41c8u},
    {0x41c84100u, 0x41c84100u, 0x41c84100u, 0x422c41c8u},
    {0x410041c8u, 0x41c841c8u, 0x41c841c8u, 0x422c41c8u},
    {0x410041c8u, 0x422c4100u, 0x422c4100u, 0x422c41c8u},
    {0x41004100u, 0x4100422cu, 0x422c41c8u, 0x422c41c8u},
    {0x4100422cu, 0x41004100u, 0x41004100u, 0x422c422cu},
    {0x41004100u, 0x41c841c8u, 0x41004100u, 0x422c422cu},
    {0x41c841c8u, 0x422c4100u, 0x41004100u, 0x422c422cu},
    {0x422c41c8u, 0x41004100u, 0x410041c8u, 0x422c422cu},
    {0x41004100u, 0x41004100u, 0x4100422cu, 0x422c422cu},
    {0x422c4100u, 0x410041c8u, 0x41c84100u, 0x422c422cu},
    {0x41004100u, 0x41c841c8u, 0x422c4100u, 0x422c422cu},
    {0x41c84100u, 0x41004100u, 0x422c41c8u, 0x422c422cu},
};

__device__ __forceinline__ float WarpSum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;
}

__device__ __forceinline__ float DotInt4x32(const uint4 wv, const float* x) {
  const uint32_t w[4] = {wv.x, wv.y, wv.z, wv.w};
  float acc = 0.f;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const uint32_t byte = (w[q] >> (8 * b)) & 0xFFu;
      const float lo = __uint_as_float(0x4B000000u | (byte & 0xFu)) - 8388608.0f - 8.0f;
      const float hi = __uint_as_float(0x4B000000u | (byte >> 4)) - 8388608.0f - 8.0f;
      acc += lo * x[q * 8 + b * 2] + hi * x[q * 8 + b * 2 + 1];
    }
  }
  return acc;
}

__device__ __forceinline__ uint32_t Fp4PairPattern(uint32_t s) {
  return ((s & 0x00070007u) << 9) | ((s & 0x00080008u) << 12);
}
__device__ __forceinline__ float Fp8E4M3ToFloat(uint32_t byte) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(byte), __NV_E4M3)));
}

enum { kFmtQ8 = 0,
       kFmtQ4 = 1,
       kFmtFp4 = 2,
       kFmtIq2 = 3,
       kFmtIq2N = 4,
       kFmtQ8F16 = 5 };

__device__ __forceinline__ void Mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
  // ComputeInternal rejects devices without BF16 tensor cores.
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ uint32_t PackBf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ __nv_bfloat162 AsBf162(uint32_t v) { return *reinterpret_cast<__nv_bfloat162*>(&v); }
__device__ __forceinline__ uint32_t AsU32(__nv_bfloat162 v) { return *reinterpret_cast<uint32_t*>(&v); }

// (byte `s` of w) as the integer value in a float: 2^23 + byte
__device__ __forceinline__ float ByteToMagic(uint32_t w, int s) {
  return __uint_as_float(__byte_perm(w, 0x4B000000u, 0x7440u | s));
}

// IQ2_XXS decode tables in shared memory: 256 x 8 bf16 magnitudes and 256 x 8 bf16 sign masks (index = 8-bit sign
// pattern).
struct Iq2Tables {
  uint4 grid[256];
  uint4 sign[256];
};
__device__ __forceinline__ Iq2Tables& Iq2Smem() {
  __shared__ Iq2Tables tables;
  return tables;
}
__device__ __forceinline__ void Iq2TablesInit() {
  Iq2Tables& t = Iq2Smem();
  for (int i = threadIdx.x; i < 256; i += blockDim.x) {
    t.grid[i] = make_uint4(kIq2xxsGridBf16[i][0], kIq2xxsGridBf16[i][1], kIq2xxsGridBf16[i][2], kIq2xxsGridBf16[i][3]);
    // bf16 sign-bit masks of the 8 elements selected by the 8-bit sign pattern i
    t.sign[i] = make_uint4(
        ((i & 1u) << 15) | (((i >> 1) & 1u) << 31), (((i >> 2) & 1u) << 15) | (((i >> 3) & 1u) << 31),
        (((i >> 4) & 1u) << 15) | (((i >> 5) & 1u) << 31), (((i >> 6) & 1u) << 15) | (((i >> 7) & 1u) << 31));
  }
  __syncthreads();
}

__device__ __forceinline__ uint32_t LoadUnalignedWord(const uint8_t* p) {
  uint32_t word = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) word |= static_cast<uint32_t>(__ldg(p + i)) << (8 * i);
  return word;
}

template <bool ALIGNED = false>
__device__ __forceinline__ uint4 LoadPackedVector(const uint8_t* p) {
  if (ALIGNED || (reinterpret_cast<uintptr_t>(p) & 15) == 0)
    return __ldg(reinterpret_cast<const uint4*>(p));
  return make_uint4(LoadUnalignedWord(p), LoadUnalignedWord(p + 4),
                    LoadUnalignedWord(p + 8), LoadUnalignedWord(p + 12));
}

template <bool ALIGNED = false>
__device__ __forceinline__ uint16_t LoadPackedScalePair(const uint8_t* p) {
  if (ALIGNED || (reinterpret_cast<uintptr_t>(p) & 1) == 0)
    return *reinterpret_cast<const uint16_t*>(p);
  return static_cast<uint16_t>(__ldg(p)) | (static_cast<uint16_t>(__ldg(p + 1)) << 8);
}

// WIDE (int4 / fp4 only): a thread loads 16 bytes = 32 k of a row per chunk (chunk = 128 k, 64 contiguous bytes per row
// for the 4 threads t) instead of 8 bytes; the two 16-k halves are consumed as two MMA groups.
template <int FMT, bool WIDE, bool ALIGNED = false>
__device__ __forceinline__ void LoadWeightRow(const uint8_t* w, const void* sc, int64_t row, int K, int c, int t,
                                              uint4& raw, float* scale) {
  if constexpr (FMT == kFmtIq2N) {
    // chunk c = 64 k = two 32-k groups of 8-byte records; thread t owns half (t & 1) of group 2c + (t >> 1)
    const uint8_t* record = w + (row * (K / 32) + 2 * c + (t >> 1)) * 8;
    uint2 rec;
    if (ALIGNED || (reinterpret_cast<uintptr_t>(record) & 7) == 0) {
      rec = __ldg(reinterpret_cast<const uint2*>(record));
    } else {
      rec = make_uint2(0, 0);
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        rec.x |= static_cast<uint32_t>(__ldg(record + b)) << (8 * b);
        rec.y |= static_cast<uint32_t>(__ldg(record + 4 + b)) << (8 * b);
      }
    }
    const int h = t & 1;
    const float d =
        __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t*>(sc) + row * (K / 256) + (c >> 2))));
    const uint32_t sg = rec.y;
    const uint32_t s0 = (sg >> (14 * h)) & 127u, s1 = (sg >> (14 * h + 7)) & 127u;
    const uint32_t sa = s0 | ((__popc(s0) & 1) << 7), sb = s1 | ((__popc(s1) & 1) << 7);
    raw = make_uint4((rec.x >> (16 * h)) & 0xFFu, (rec.x >> (16 * h + 8)) & 0xFFu, sa, sb);
    scale[0] = d * (0.5f + static_cast<float>(sg >> 28)) * 0.25f;
  } else if constexpr (FMT == kFmtQ8 || FMT == kFmtQ8F16) {
    raw = LoadPackedVector<ALIGNED>(w + row * K + 64 * c + 16 * t);
    const int64_t index = row * (K / 32) + 2 * c + (t >> 1);
    if constexpr (FMT == kFmtQ8F16) {
      // Preserve source FP16 scales until the existing FP32 reconstruction / BF16 store.
      scale[0] = __half2float(reinterpret_cast<const __half*>(sc)[index]);
    } else {
      scale[0] = B2F(reinterpret_cast<const bf16*>(sc)[index]);
    }
  } else if constexpr (!WIDE) {
    const uint8_t* p = w + row * (K / 2) + 32 * c + 8 * t;
    const uint2 v = ALIGNED || (reinterpret_cast<uintptr_t>(p) & 7) == 0
                        ? __ldg(reinterpret_cast<const uint2*>(p))
                        : make_uint2(LoadUnalignedWord(p), LoadUnalignedWord(p + 4));
    raw = make_uint4(v.x, v.y, 0u, 0u);
    if constexpr (FMT == kFmtQ4 || FMT == kFmtIq2)
      scale[0] = B2F(reinterpret_cast<const bf16*>(sc)[row * (K / 32) + 2 * c + (t >> 1)]);
    else
      scale[0] = Fp8E4M3ToFloat(reinterpret_cast<const uint8_t*>(sc)[row * (K / 16) + 4 * c + t]) * 0x1p110f;
  } else {
    raw = LoadPackedVector<ALIGNED>(w + row * (K / 2) + 64 * c + 16 * t);
    if constexpr (FMT == kFmtQ4 || FMT == kFmtIq2) {
      scale[0] = B2F(reinterpret_cast<const bf16*>(sc)[row * (K / 32) + 4 * c + t]);
    } else {
      const uint16_t two = LoadPackedScalePair<ALIGNED>(
          reinterpret_cast<const uint8_t*>(sc) + row * (K / 16) + 8 * c + 2 * t);
      scale[0] = Fp8E4M3ToFloat(two & 0xFFu) * 0x1p110f;
      scale[1] = Fp8E4M3ToFloat(two >> 8) * 0x1p110f;
    }
  }
}

template <int FMT, bool WIDE, int MT>
struct StageT {
  static constexpr int NH = WIDE ? 2 : 1;  // 16-k halves per thread and chunk
  static constexpr int NSC = (FMT == kFmtFp4 && WIDE) ? 2 : 1;
  uint4 raw[MT][2];
  float sc[MT][2][NSC];
  uint4 x[2 * NH];
};

// Dequantised bf16x2 pairs for MMA step s: lo[s] = k-slots (s, s+4), hi[s] = (8+s, 12+s).
template <int FMT>
__device__ __forceinline__ void DequantRow(const uint4& raw, float scale, uint32_t* lo, uint32_t* hi) {
  if constexpr (FMT == kFmtIq2N) {
    const uint4 ga = __ldg(reinterpret_cast<const uint4*>(kIq2xxsGridBf16[raw.x])),
                gb = __ldg(reinterpret_cast<const uint4*>(kIq2xxsGridBf16[raw.y]));
    const uint32_t sa = raw.z, sb = raw.w;
    // bf16 sign-bit masks: element j of the 8 takes bit j of the sign pattern
    // (x * (2^(15-2k) + 2^(30-2k))) puts bit 2k of x at bit 15 and bit 2k+1 at bit 31 without a carry between the two
    // products
    auto mask = [](uint32_t sgn, int pair) {
      return (sgn * ((1u << (15 - 2 * pair)) + (1u << (30 - 2 * pair)))) & 0x80008000u;
    };
    // Match raw GGML decode: round the complete grid*scale operand, not scale alone.
    auto scaled_pair = [scale](uint32_t bits) {
      const auto pair = AsBf162(bits);
      return PackBf16(B2F(__low2bfloat16(pair)) * scale, B2F(__high2bfloat16(pair)) * scale);
    };
    lo[0] = scaled_pair(ga.x ^ mask(sa, 0));
    hi[0] = scaled_pair(ga.y ^ mask(sa, 1));
    lo[1] = scaled_pair(ga.z ^ mask(sa, 2));
    hi[1] = scaled_pair(ga.w ^ mask(sa, 3));
    lo[2] = scaled_pair(gb.x ^ mask(sb, 0));
    hi[2] = scaled_pair(gb.y ^ mask(sb, 1));
    lo[3] = scaled_pair(gb.z ^ mask(sb, 2));
    hi[3] = scaled_pair(gb.w ^ mask(sb, 3));
  } else if constexpr (FMT == kFmtQ8 || FMT == kFmtQ8F16) {
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      lo[s] = PackBf16((ByteToMagic(raw.x, s) - 8388736.0f) * scale, (ByteToMagic(raw.y, s) - 8388736.0f) * scale);
      hi[s] = PackBf16((ByteToMagic(raw.z, s) - 8388736.0f) * scale, (ByteToMagic(raw.w, s) - 8388736.0f) * scale);
    }
  } else if constexpr (FMT == kFmtIq2) {
    // LUT bytes indexed by code 0..7: bf16 of +8, +25, +43, -, -8, -25, -43, - (high / low byte planes)
    constexpr uint32_t kHiA = 0x00424141u, kHiB = 0x00C2C1C1u, kLoA = 0x002CC800u, kLoB = 0x002CC800u;
    const __nv_bfloat162 sc2 = __bfloat162bfloat162(__float2bfloat16(scale));
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t w = (s < 2) ? raw.x : raw.y;
      const uint32_t sel = (s & 1) ? (w >> 16) : (w & 0xFFFFu);
      const uint32_t H = __byte_perm(kHiA, kHiB, sel);
      const uint32_t L = __byte_perm(kLoA, kLoB, sel);
      lo[s] = AsU32(__hmul2(AsBf162(__byte_perm(L, H, 0x5140)), sc2));
      hi[s] = AsU32(__hmul2(AsBf162(__byte_perm(L, H, 0x7362)), sc2));
    }
  } else if constexpr (FMT == kFmtQ4) {
    const __nv_bfloat162 sc2 = __bfloat162bfloat162(__float2bfloat16(scale));
    const __nv_bfloat162 zp = __floats2bfloat162_rn(136.f, 136.f);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      // nibble s and s+4 of a word -> bf16 128 + q (exact), then (q - 8) * scale
      const uint32_t a = ((raw.x >> (4 * s)) & 0x000F000Fu) | 0x43004300u;
      const uint32_t b = ((raw.y >> (4 * s)) & 0x000F000Fu) | 0x43004300u;
      lo[s] = AsU32(__hmul2(__hsub2(AsBf162(a), zp), sc2));
      hi[s] = AsU32(__hmul2(__hsub2(AsBf162(b), zp), sc2));
    }
  } else {
    // e2m1 -> bf16 pattern worth value * 2^-126 (subnormal for |v| = 0.5); scale is pre-multiplied by 2^110 and
    // epilogue by 2^16.
    const __nv_bfloat162 sc2 = __bfloat162bfloat162(__float2bfloat16(scale));
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint32_t pa = raw.x >> (4 * s), pb = raw.y >> (4 * s);
      const uint32_t a = ((pa << 6) & 0x01C001C0u) | ((pa << 12) & 0x80008000u);
      const uint32_t b = ((pb << 6) & 0x01C001C0u) | ((pb << 12) & 0x80008000u);
      lo[s] = AsU32(__hmul2(AsBf162(a), sc2));
      hi[s] = AsU32(__hmul2(AsBf162(b), sc2));
    }
  }
}

// Preserve the source's predicated vector loads when alignment is guaranteed at launch.
template <int NH>
__device__ __forceinline__ void LoadAlignedX(const bf16* xr, bool valid, int c, int t, uint4* xo) {
  const uint4* xp = reinterpret_cast<const uint4*>(xr + 64 * NH * c + 16 * NH * t);
#pragma unroll
  for (int i = 0; i < 2 * NH; ++i) xo[i] = valid ? __ldg(xp + i) : make_uint4(0, 0, 0, 0);
}

// Loads the x fragments of one chunk: 16 (or 32) contiguous k of activation row g for this thread.
template <int NH, bool ALIGNED = false>
__device__ __forceinline__ void LoadX(const bf16* xr, bool valid, int c, int t, uint4* xo) {
  const uint16_t* xp = reinterpret_cast<const uint16_t*>(xr + 64 * NH * c + 16 * NH * t);
#pragma unroll
  for (int i = 0; i < 2 * NH; ++i) {
    if (!valid) {
      xo[i] = make_uint4(0, 0, 0, 0);
    } else if (ALIGNED || (reinterpret_cast<uintptr_t>(xp) & 15) == 0) {
      xo[i] = __ldg(reinterpret_cast<const uint4*>(xp) + i);
    } else {
      uint32_t words[4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
        words[j] = static_cast<uint32_t>(__ldg(xp + 8 * i + 2 * j)) |
                   (static_cast<uint32_t>(__ldg(xp + 8 * i + 2 * j + 1)) << 16);
      xo[i] = make_uint4(words[0], words[1], words[2], words[3]);
    }
  }
}

template <int FMT, bool WIDE, int MT>
__device__ __forceinline__ void ComputeStage(const StageT<FMT, WIDE, MT>& st, float (*acc)[4]) {
#pragma unroll
  for (int h = 0; h < StageT<FMT, WIDE, MT>::NH; ++h) {
    const uint4 xa = st.x[2 * h], xb = st.x[2 * h + 1];
    uint32_t bl[4], bh[4];
    if constexpr (FMT == kFmtIq2 || FMT == kFmtIq2N) {  // adjacent k pairs: MMA s consumes x[4s .. 4s+3]
      bl[0] = xa.x;
      bh[0] = xa.y;
      bl[1] = xa.z;
      bh[1] = xa.w;
      bl[2] = xb.x;
      bh[2] = xb.y;
      bl[3] = xb.z;
      bh[3] = xb.w;
    } else {
      bl[0] = __byte_perm(xa.x, xa.z, 0x5410);
      bl[1] = __byte_perm(xa.x, xa.z, 0x7632);
      bl[2] = __byte_perm(xa.y, xa.w, 0x5410);
      bl[3] = __byte_perm(xa.y, xa.w, 0x7632);
      bh[0] = __byte_perm(xb.x, xb.z, 0x5410);
      bh[1] = __byte_perm(xb.x, xb.z, 0x7632);
      bh[2] = __byte_perm(xb.y, xb.w, 0x5410);
      bh[3] = __byte_perm(xb.y, xb.w, 0x7632);
    }
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      uint32_t lo0[4], hi0[4], lo1[4], hi1[4];
      uint4 r0 = st.raw[m][0], r1 = st.raw[m][1];
      if constexpr (WIDE) {
        if (h == 1) {
          r0 = make_uint4(r0.z, r0.w, 0u, 0u);
          r1 = make_uint4(r1.z, r1.w, 0u, 0u);
        }
      }
      constexpr int sh = (FMT == kFmtFp4 && WIDE) ? 1 : 0;
      DequantRow<FMT>(r0, st.sc[m][0][h * sh], lo0, hi0);
      DequantRow<FMT>(r1, st.sc[m][1][h * sh], lo1, hi1);
#pragma unroll
      for (int s = 0; s < 4; ++s) Mma16816(acc[m], lo0[s], lo1[s], hi0[s], hi1[s], bl[s], bh[s]);
    }
  }
}

// Block-wide reduction of the per-warp K partials (red: [KW][MT*16][8] floats) and the epilogue callback f(row_local,
// col, value).
template <int MT, int KW, typename F>
__device__ __forceinline__ void ReduceEpilogue(float (*acc)[4], int warp, int g, int t, F&& f) {
  extern __shared__ float red[];
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    float* r0 = red + (warp * MT * 16 + m * 16 + g) * 8 + 2 * t;
    r0[0] = acc[m][0];
    r0[1] = acc[m][1];
    r0[8 * 8] = acc[m][2];
    r0[8 * 8 + 1] = acc[m][3];
  }
  __syncthreads();
  for (int i = threadIdx.x; i < MT * 16 * 8; i += KW * 32) {
    const int rl = i >> 3, col = i & 7;
    float v = 0.f;
#pragma unroll
    for (int k = 0; k < KW; ++k) v += red[(k * MT * 16 + rl) * 8 + col];
    f(rl, col, v);
  }
}

// Main loop shared by the dense and the MoE kernels: UNR chunks per warp in flight, next batch issued before the
// current one is consumed. load(c, stage) / the compute step are chunk-indexed.
template <int FMT, bool WIDE, int MT, int KW, int UNR, typename L>
__device__ __forceinline__ void MmaMainLoop(int warp, int nchunks, float (*acc)[4], L&& load) {
  using Stage = StageT<FMT, WIDE, MT>;
  Stage cur[UNR], nxt[UNR];
  int c0 = warp;
#pragma unroll
  for (int j = 0; j < UNR; ++j)
    if (c0 + j * KW < nchunks) load(c0 + j * KW, cur[j]);
  for (; c0 < nchunks; c0 += KW * UNR) {
#pragma unroll
    for (int j = 0; j < UNR; ++j)
      if (c0 + (UNR + j) * KW < nchunks) load(c0 + (UNR + j) * KW, nxt[j]);
#pragma unroll
    for (int j = 0; j < UNR; ++j)
      if (c0 + j * KW < nchunks) ComputeStage<FMT, WIDE, MT>(cur[j], acc);
#pragma unroll
    for (int j = 0; j < UNR; ++j) cur[j] = nxt[j];
  }
}

}  // namespace onnxruntime::contrib::cuda::packed_projection
