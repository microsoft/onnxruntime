/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Auto-generated MoE GEMM kernel instantiations for SM80.
 * DO NOT EDIT MANUALLY.
 */

#ifndef EXCLUDE_SM_80
#include "contrib_ops/cuda/llm/moe_gemm/launchers/fused_moe_gemm_launcher_sm80.inl"

namespace onnxruntime::llm::kernels::cutlass_kernels {

#ifdef ENABLE_BF16
template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 128, 128, 64, 2, onnxruntime::llm::cutlass_extensions::EpilogueOpDefaultSilu>(
    cutlass::half_t const*, cutlass::half_t const*, cutlass::half_t const*, bool, cutlass::half_t*,
    int64_t const*, int64_t, int64_t, int64_t, int, int, cudaStream_t, int*);

template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 128, 128, 64, 2, onnxruntime::llm::cutlass_extensions::EpilogueOpDefaultFtGelu>(
    cutlass::half_t const*, cutlass::half_t const*, cutlass::half_t const*, bool, cutlass::half_t*,
    int64_t const*, int64_t, int64_t, int64_t, int, int, cudaStream_t, int*);

template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 128, 128, 64, 3, onnxruntime::llm::cutlass_extensions::EpilogueOpDefaultSilu>(
    cutlass::half_t const*, cutlass::half_t const*, cutlass::half_t const*, bool, cutlass::half_t*,
    int64_t const*, int64_t, int64_t, int64_t, int, int, cudaStream_t, int*);

template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 128, 128, 64, 3, onnxruntime::llm::cutlass_extensions::EpilogueOpDefaultFtGelu>(
    cutlass::half_t const*, cutlass::half_t const*, cutlass::half_t const*, bool, cutlass::half_t*,
    int64_t const*, int64_t, int64_t, int64_t, int, int, cudaStream_t, int*);

template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 128, 128, 64, 4, onnxruntime::llm::cutlass_extensions::EpilogueOpDefaultSilu>(
    cutlass::half_t const*, cutlass::half_t const*, cutlass::half_t const*, bool, cutlass::half_t*,
    int64_t const*, int64_t, int64_t, int64_t, int, int, cudaStream_t, int*);

template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 128, 128, 64, 4, onnxruntime::llm::cutlass_extensions::EpilogueOpDefaultFtGelu>(
    cutlass::half_t const*, cutlass::half_t const*, cutlass::half_t const*, bool, cutlass::half_t*,
    int64_t const*, int64_t, int64_t, int64_t, int, int, cudaStream_t, int*);

#else
// BF16 not enabled, only instantiate FP16 variants
template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 128, 128, 64, 2, onnxruntime::llm::cutlass_extensions::EpilogueOpDefaultSilu>(
    cutlass::half_t const*, cutlass::half_t const*, cutlass::half_t const*, bool, cutlass::half_t*,
    int64_t const*, int64_t, int64_t, int64_t, int, int, cudaStream_t, int*);

template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 128, 128, 64, 2, onnxruntime::llm::cutlass_extensions::EpilogueOpDefaultFtGelu>(
    cutlass::half_t const*, cutlass::half_t const*, cutlass::half_t const*, bool, cutlass::half_t*,
    int64_t const*, int64_t, int64_t, int64_t, int, int, cudaStream_t, int*);

template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 128, 128, 64, 3, onnxruntime::llm::cutlass_extensions::EpilogueOpDefaultSilu>(
    cutlass::half_t const*, cutlass::half_t const*, cutlass::half_t const*, bool, cutlass::half_t*,
    int64_t const*, int64_t, int64_t, int64_t, int, int, cudaStream_t, int*);

template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 128, 128, 64, 3, onnxruntime::llm::cutlass_extensions::EpilogueOpDefaultFtGelu>(
    cutlass::half_t const*, cutlass::half_t const*, cutlass::half_t const*, bool, cutlass::half_t*,
    int64_t const*, int64_t, int64_t, int64_t, int, int, cudaStream_t, int*);

template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 128, 128, 64, 4, onnxruntime::llm::cutlass_extensions::EpilogueOpDefaultSilu>(
    cutlass::half_t const*, cutlass::half_t const*, cutlass::half_t const*, bool, cutlass::half_t*,
    int64_t const*, int64_t, int64_t, int64_t, int, int, cudaStream_t, int*);

template void sm80_generic_fused_moe_gemm_kernelLauncher<cutlass::half_t, cutlass::half_t, 128, 128, 64, 4, onnxruntime::llm::cutlass_extensions::EpilogueOpDefaultFtGelu>(
    cutlass::half_t const*, cutlass::half_t const*, cutlass::half_t const*, bool, cutlass::half_t*,
    int64_t const*, int64_t, int64_t, int64_t, int, int, cudaStream_t, int*);

#endif

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
#endif  // EXCLUDE_SM_80
