/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Auto-generated SM90 TMA Warp Specialized Grouped GEMM instantiations for BF16 activations with FP8 weights (W8A16-FP8).
 * DO NOT EDIT MANUALLY.
 */

#ifdef COMPILE_HOPPER_TMA_GROUPED_GEMMS

#include "contrib_ops/cuda/llm/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl"

namespace onnxruntime::llm::kernels::cutlass_kernels {

INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 16, 64, 1, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 16, 64, 1, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 16, 64, 1, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 16, 64, 1, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 16, 64, 2, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 16, 64, 2, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 16, 64, 2, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 16, 64, 2, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 32, 64, 1, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 32, 64, 1, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 32, 64, 1, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 32, 64, 1, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 32, 64, 2, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 32, 64, 2, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 32, 64, 2, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 32, 64, 2, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 64, 64, 1, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 64, 64, 1, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 64, 64, 1, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 64, 64, 1, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 64, 64, 2, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 64, 64, 2, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 64, 64, 2, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 64, 64, 2, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 128, 64, 1, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 128, 64, 1, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 128, 64, 1, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 128, 64, 1, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 128, 64, 2, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 128, 64, 2, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 128, 64, 2, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 128, 64, 2, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 256, 64, 1, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 256, 64, 1, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 256, 64, 1, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 256, 64, 1, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 256, 64, 2, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 256, 64, 2, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 128, 256, 64, 2, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 128, 256, 64, 2, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 256, 128, 64, 1, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 256, 128, 64, 1, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 256, 128, 64, 1, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 256, 128, 64, 1, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 256, 128, 64, 2, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 256, 128, 64, 2, 1, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, NONE, 256, 128, 64, 2, 2, 1, false, false)
INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, SafeBF16, SafeFP8, SafeBF16, EpilogueOpDefault, FINALIZE, 256, 128, 64, 2, 2, 1, false, false)

}  // namespace onnxruntime::llm::kernels::cutlass_kernels

#endif  // COMPILE_HOPPER_TMA_GROUPED_GEMMS
