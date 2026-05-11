/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Stub instantiations for SM90 mixed-input FP4 MoE GEMM launcher. The full
 * implementation in moe_gemm_tma_ws_sm90_mixed_fp4.generated.cu currently
 * triggers an illegal/misaligned memory access inside the CUTLASS 4.4.2
 * PtrArray mixed-input collective on H100/H200 (both FP16 and BF16
 * activations). Until the underlying CUTLASS issue is resolved, this stub
 * is built instead so that constructing a QMoE FP4 session fails at session
 * creation time with a clear error rather than crashing the process at
 * kernel launch.
 *
 * The set of instantiated specializations must mirror those referenced by
 * moe_gemm_template_dispatch_tma_ws_mixed_dtype.h, otherwise the CUDA EP
 * shared library will fail to load with an undefined-symbol error and *all*
 * CUDA sessions (not just QMoE FP4) will fall back to the CPU EP.
 */

#ifdef ENABLE_FP4

#include "contrib_ops/cuda/llm/moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_launcher.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/epilogue_helpers.h"
#include "core/common/common.h"

#include <cute/tensor.hpp>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>

namespace onnxruntime::llm::kernels::cutlass_kernels {

template <typename T, typename WeightType, typename GemmOutputType, typename EpilogueTag, typename CTAShape,
          typename ClusterShape, typename MainloopScheduleType, typename EpilogueScheduleType,
          cutlass::WeightOnlyQuantOp QuantOp>
void sm90_generic_mixed_moe_gemm_kernelLauncher(GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> /*inputs*/,
                                                TmaWarpSpecializedGroupedGemmInput /*hopper_inputs*/, int /*sm_count_*/,
                                                size_t* /*workspace_size*/) {
  ORT_THROW(
      "SM90 mixed-input FP4 (MXFP4) MoE GEMM is not supported in this build. "
      "The PtrArray mixed-input collective in the bundled CUTLASS 4.4.2 currently "
      "fails with an illegal/misaligned memory access for FP4 weights with "
      "FP16/BF16 activations on H100/H200. The launcher has been stubbed out "
      "until the underlying CUTLASS issue is resolved.");
}

using EpiTag = onnxruntime::llm::cutlass_extensions::EpilogueOpDefault;
using EpiSched = cutlass::epilogue::TmaWarpSpecializedCooperative;
using PP = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using COOP = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
static constexpr auto QOP = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;

#define INST_PP(T, M, N, K, CM, CN, CK)                         \
  template void sm90_generic_mixed_moe_gemm_kernelLauncher<     \
      T, __nv_fp4_e2m1, T, EpiTag,                              \
      cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>,    \
      cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>, \
      PP, EpiSched, QOP>(                                       \
      GroupedGemmInput<T, __nv_fp4_e2m1, T, T>, TmaWarpSpecializedGroupedGemmInput, int, size_t*);

#define INST_CO(T, M, N, K, CM, CN, CK)                         \
  template void sm90_generic_mixed_moe_gemm_kernelLauncher<     \
      T, __nv_fp4_e2m1, T, EpiTag,                              \
      cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>,    \
      cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>, \
      COOP, EpiSched, QOP>(                                     \
      GroupedGemmInput<T, __nv_fp4_e2m1, T, T>, TmaWarpSpecializedGroupedGemmInput, int, size_t*);

// The following set mirrors moe_gemm_tma_ws_sm90_mixed_fp4.generated.cu so that
// every dispatch site in moe_gemm_template_dispatch_tma_ws_mixed_dtype.h finds a
// matching symbol at link time. K=256 (= TileK = mxfp4_group_size * PackedScalesNum)
// is the only depth used for FP4.

// half / cluster 1x1x1, PP, M=128
INST_PP(half, 128, 16, 256, 1, 1, 1)
INST_PP(half, 128, 32, 256, 1, 1, 1)
INST_PP(half, 128, 64, 256, 1, 1, 1)
INST_PP(half, 128, 128, 256, 1, 1, 1)

// half / M=64 tiles, PP only, all four cluster shapes
INST_PP(half, 64, 16, 256, 1, 1, 1)
INST_PP(half, 64, 16, 256, 2, 1, 1)
INST_PP(half, 64, 16, 256, 1, 2, 1)
INST_PP(half, 64, 16, 256, 2, 2, 1)
INST_PP(half, 64, 32, 256, 1, 1, 1)
INST_PP(half, 64, 32, 256, 2, 1, 1)
INST_PP(half, 64, 32, 256, 1, 2, 1)
INST_PP(half, 64, 32, 256, 2, 2, 1)
INST_PP(half, 64, 64, 256, 1, 1, 1)
INST_PP(half, 64, 64, 256, 2, 1, 1)
INST_PP(half, 64, 64, 256, 1, 2, 1)
INST_PP(half, 64, 64, 256, 2, 2, 1)

// half / M=128 tiles, PP+COOP, additional cluster shapes
INST_PP(half, 128, 16, 256, 2, 1, 1)
INST_CO(half, 128, 16, 256, 1, 1, 1)
INST_CO(half, 128, 16, 256, 2, 1, 1)
INST_PP(half, 128, 16, 256, 1, 2, 1)
INST_CO(half, 128, 16, 256, 1, 2, 1)
INST_PP(half, 128, 16, 256, 2, 2, 1)
INST_CO(half, 128, 16, 256, 2, 2, 1)
INST_PP(half, 128, 32, 256, 2, 1, 1)
INST_CO(half, 128, 32, 256, 1, 1, 1)
INST_CO(half, 128, 32, 256, 2, 1, 1)
INST_PP(half, 128, 32, 256, 1, 2, 1)
INST_CO(half, 128, 32, 256, 1, 2, 1)
INST_PP(half, 128, 32, 256, 2, 2, 1)
INST_CO(half, 128, 32, 256, 2, 2, 1)
INST_PP(half, 128, 64, 256, 2, 1, 1)
INST_CO(half, 128, 64, 256, 1, 1, 1)
INST_CO(half, 128, 64, 256, 2, 1, 1)
INST_PP(half, 128, 64, 256, 1, 2, 1)
INST_CO(half, 128, 64, 256, 1, 2, 1)
INST_PP(half, 128, 64, 256, 2, 2, 1)
INST_CO(half, 128, 64, 256, 2, 2, 1)
INST_PP(half, 128, 128, 256, 2, 1, 1)
INST_PP(half, 128, 128, 256, 1, 2, 1)
INST_PP(half, 128, 128, 256, 2, 2, 1)

#ifdef ENABLE_BF16
INST_PP(__nv_bfloat16, 64, 16, 256, 1, 1, 1)
INST_PP(__nv_bfloat16, 64, 16, 256, 2, 1, 1)
INST_PP(__nv_bfloat16, 64, 16, 256, 1, 2, 1)
INST_PP(__nv_bfloat16, 64, 16, 256, 2, 2, 1)
INST_PP(__nv_bfloat16, 64, 32, 256, 1, 1, 1)
INST_PP(__nv_bfloat16, 64, 32, 256, 2, 1, 1)
INST_PP(__nv_bfloat16, 64, 32, 256, 1, 2, 1)
INST_PP(__nv_bfloat16, 64, 32, 256, 2, 2, 1)
INST_PP(__nv_bfloat16, 64, 64, 256, 1, 1, 1)
INST_PP(__nv_bfloat16, 64, 64, 256, 2, 1, 1)
INST_PP(__nv_bfloat16, 64, 64, 256, 1, 2, 1)
INST_PP(__nv_bfloat16, 64, 64, 256, 2, 2, 1)

INST_PP(__nv_bfloat16, 128, 16, 256, 1, 1, 1)
INST_CO(__nv_bfloat16, 128, 16, 256, 1, 1, 1)
INST_PP(__nv_bfloat16, 128, 16, 256, 2, 1, 1)
INST_CO(__nv_bfloat16, 128, 16, 256, 2, 1, 1)
INST_PP(__nv_bfloat16, 128, 16, 256, 1, 2, 1)
INST_CO(__nv_bfloat16, 128, 16, 256, 1, 2, 1)
INST_PP(__nv_bfloat16, 128, 16, 256, 2, 2, 1)
INST_CO(__nv_bfloat16, 128, 16, 256, 2, 2, 1)
INST_PP(__nv_bfloat16, 128, 32, 256, 1, 1, 1)
INST_CO(__nv_bfloat16, 128, 32, 256, 1, 1, 1)
INST_PP(__nv_bfloat16, 128, 32, 256, 2, 1, 1)
INST_CO(__nv_bfloat16, 128, 32, 256, 2, 1, 1)
INST_PP(__nv_bfloat16, 128, 32, 256, 1, 2, 1)
INST_CO(__nv_bfloat16, 128, 32, 256, 1, 2, 1)
INST_PP(__nv_bfloat16, 128, 32, 256, 2, 2, 1)
INST_CO(__nv_bfloat16, 128, 32, 256, 2, 2, 1)
INST_PP(__nv_bfloat16, 128, 64, 256, 1, 1, 1)
INST_CO(__nv_bfloat16, 128, 64, 256, 1, 1, 1)
INST_PP(__nv_bfloat16, 128, 64, 256, 2, 1, 1)
INST_CO(__nv_bfloat16, 128, 64, 256, 2, 1, 1)
INST_PP(__nv_bfloat16, 128, 64, 256, 1, 2, 1)
INST_CO(__nv_bfloat16, 128, 64, 256, 1, 2, 1)
INST_PP(__nv_bfloat16, 128, 64, 256, 2, 2, 1)
INST_CO(__nv_bfloat16, 128, 64, 256, 2, 2, 1)
INST_PP(__nv_bfloat16, 128, 128, 256, 1, 1, 1)
INST_PP(__nv_bfloat16, 128, 128, 256, 2, 1, 1)
INST_PP(__nv_bfloat16, 128, 128, 256, 1, 2, 1)
INST_PP(__nv_bfloat16, 128, 128, 256, 2, 2, 1)
#endif  // ENABLE_BF16

#undef INST_PP
#undef INST_CO

}  // namespace onnxruntime::llm::kernels::cutlass_kernels

#endif  // ENABLE_FP4
