// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// True when the single-kernel tensor-core path can run this call on the current device: a supported shape (rows <= 8,
// k > 0), operands aligned for its vector loads (W to 16 bytes, X to 4 bytes; a tensor can be a view at any 2-byte
// offset into a larger buffer), and a kernel image compiled for sm_90+. A build with only older PTX still loads on an
// sm_90 device through PTX JIT, but its kernel body is empty there.
bool RouterGateUpProjUsesFusedKernel(int sm_major, int64_t rows, int64_t k, int64_t router, int64_t inter,
                                     const void* x, const void* w);

// Fused path: requires RouterGateUpProjUsesFusedKernel. Pointers are bfloat16 or float16 data (`is_bf16`).
Status LaunchRouterGateUpProjFused(cudaStream_t stream, bool use_pdl, bool is_bf16, const void* x, const void* w,
                                   void* router_out, void* act_out, int64_t rows, int64_t k, int64_t n,
                                   int64_t router, int64_t inter, float limit, float alpha, float beta);

// Generic path: y = x * w into `scratch` ([rows, n]) with cuBLAS, then a split + activation kernel. For k == 0 the
// projection is all zeros, as in MatMul, and only the activation runs.
Status LaunchRouterGateUpProjGeneric(cublasHandle_t cublas, cudaStream_t stream, bool is_bf16, const void* x,
                                     const void* w, void* scratch, void* router_out, void* act_out, int64_t rows,
                                     int64_t k, int64_t n, int64_t router, int64_t inter, float limit, float alpha,
                                     float beta);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
