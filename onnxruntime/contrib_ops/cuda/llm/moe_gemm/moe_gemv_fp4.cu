// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.h"

#include <cuda_fp16.h>
#include <type_traits>

// Shared INT/FP4 profiled-shape thresholds.
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv.h"
// Shared device-side kernels + launch/dispatch helpers (fpA_intB_gemv namespace).
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv_device.cuh"
#include "core/platform/env_var_utils.h"

namespace onnxruntime::llm {
namespace kernels {
namespace moe_gemv {

namespace fiv = onnxruntime::llm::kernels::fpA_intB_gemv;

// Tiling knobs swept by the FP4 GEMV autotuner. These mirror the INT path's default m-tile
// config (kDefaultCtaN/kDefaultThreads) plus two alternative tilings (kCtaN16/kThreads64).
static constexpr int kDefaultCtaN = 8;
static constexpr int kDefaultThreads = 128;
static constexpr int kCtaN16 = 16;
static constexpr int kThreads64 = 64;

int CtaNForConfig(MoeGemvConfig config) {
  return config == MoeGemvConfig::kCtaN16 ? kCtaN16 : kDefaultCtaN;
}

// Opt-in MXFP4 fc1 split-K + fp32-accumulation SwiGLU GEMV path (env ORT_FP4_GEMV_SPLITK=1).
// Off by default so the shipped single-pass FP4 GEMV stays the default. Parsed once via ORT's
// environment helper for consistent parsing/thread-safety.
bool Fp4MoeGemvUseSplitK() {
  static bool const enabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_GEMV_SPLITK", 0) == 1;
  return enabled;
}

// --- MXFP4 (e2m1) GEMV: non-interleaved ColumnMajor layout (kInterleave = 1) ---
// Weights are the QMoERepackFP4ColToRow output ([experts, n, k/2] row-major, two e2m1
// codes per byte, even-K in the low nibble). Block scales are the
// LaunchQMoECombineFp4ScalesForGemv output ([experts, k/32, n] in TypeA, already folded with
// the per-expert global scale). The kernel decodes e2m1 in-register via Fp4I2FConverter.
template <typename T>
struct Fp4ADetails;
template <>
struct Fp4ADetails<half> {
  using Type = fiv::FP16DetailsA;
};
#ifdef ENABLE_BF16
template <>
struct Fp4ADetails<__nv_bfloat16> {
  using Type = fiv::BF16DetailsA;
};
#endif

// TileSizeK is unused by the ColumnMajor (kInterleave = 1) indexing/reduction beyond the
// shmem-write lane gating (only lane 0 of each warp writes), so 64 matches the INT convention.
static constexpr int kTileSizeKFp4 = 64;
template <typename T>
using Fp4KernelDetails =
    fiv::KernelDetails<typename Fp4ADetails<T>::Type, fiv::Fp4DetailsW, fiv::ColumnMajor, false, kTileSizeKFp4>;

// MXFP4 GEMV shape support. Mirrors is_moe_gemv_supported but for the non-interleaved
// ColumnMajor layout: kInterleave = 1, so n need only be divisible by the CtaN tile width
// selected by `config`, and the per-thread step is StepK = 128 / activation_bits = 8
// (not 128 / weight_bits).
bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                               MoeGemvConfig config) {
  if (sm < 80) {
    return false;
  }
  if (group_size != 32) {  // MXFP4 block size
    return false;
  }
  if (k % group_size != 0) {
    return false;
  }
  if (expanded_num_rows <= 0 || expanded_num_rows > kMaxProfiledExpandedRows) {
    return false;
  }
  if (n < kMinProfiledProblemDim || k < kMinProfiledProblemDim) {
    return false;
  }
  if (expanded_num_rows > kMaxProfiledExpandedRowsForSmallProblemDim &&
      (n < kMinProfiledProblemDimForExpandedRowsAbove4 || k < kMinProfiledProblemDimForExpandedRowsAbove4)) {
    return false;
  }
  if (n % CtaNForConfig(config) != 0) {  // kInterleave = 1
    return false;
  }
  // StepK = 128 / activation_bits = 8; k is a multiple of 32, so k % 8 == 0 always holds.
  if (k % (128 / 16) != 0) {
    return false;
  }
  return true;
}

bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size) {
  return is_moe_gemv_fp4_supported(sm, expanded_num_rows, n, k, group_size, MoeGemvConfig::kDefault);
}

template <typename T>
void launch_moe_gemv_fp4_symmetric(T const* act, uint8_t const* weight, T const* scales, T const* bias, T* out,
                                   int64_t const* expert_first_token_offset, int const* permuted_row_to_expert,
                                   int num_experts, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                                   int sm, MoeGemvConfig config, cudaStream_t stream) {
  ORT_UNUSED_PARAMETER(sm);
  using Details = Fp4KernelDetails<T>;
  // CtaN/Threads are pure parallelization/tiling knobs: the reduction and 16-bit accumulation
  // are identical for every config, so this sweep is numerically bit-exact (no accuracy gate).
  auto launch = [&](auto cta_n, auto threads) {
    fiv::dispatch_moe_gemv_group_size<Details, cta_n(), threads(), T>(
        const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
        expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, n, k, group_size, stream);
  };
  if (config == MoeGemvConfig::kCtaN16) {
    launch([] { return kCtaN16; }, [] { return kDefaultThreads; });
  } else if (config == MoeGemvConfig::kThreads64) {
    launch([] { return kDefaultCtaN; }, [] { return kThreads64; });
  } else {
    launch([] { return kDefaultCtaN; }, [] { return kDefaultThreads; });
  }
}

template <typename T>
void launch_moe_gemv_fp4_symmetric_interleaved_swiglu(
    T const* act, uint8_t const* weight, T const* scales, T const* bias, T* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k, int group_size, int sm,
    cutlass_kernels::ActivationParams activation_params, MoeGemvConfig config, void* splitk_partials,
    cudaStream_t stream) {
  ORT_UNUSED_PARAMETER(sm);
  using Details = Fp4KernelDetails<T>;
  // Opt-in two-pass split-K + fp32-accumulation path. Each split accumulates a shorter K-chain
  // in fp32 and the cross-split reduce sums in fp32 (the partials buffer holds fp32). This both
  // tightens bf16 accuracy and refills the SMs that fp32 accumulation alone leaves under-occupied.
  // The dispatcher auto-degrades to the single-pass interleaved-SwiGLU GEMV when num_iters < 2.
  if (splitk_partials != nullptr && Fp4MoeGemvUseSplitK()) {
    auto launch_splitk = [&](auto cta_n, auto threads) {
      fiv::dispatch_moe_gemv_splitk_twopass_swiglu_group_size<Details, cta_n(), threads(),
                                                              kFp4MoeGemvSplitK, T, float, float>(
          const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
          expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k,
          group_size, activation_params, static_cast<float*>(splitk_partials), stream);
    };
    if (config == MoeGemvConfig::kCtaN16) {
      launch_splitk([] { return kCtaN16; }, [] { return kDefaultThreads; });
    } else if (config == MoeGemvConfig::kThreads64) {
      launch_splitk([] { return kDefaultCtaN; }, [] { return kThreads64; });
    } else {
      launch_splitk([] { return kDefaultCtaN; }, [] { return kDefaultThreads; });
    }
    return;
  }
  // CtaN/Threads are numerically bit-exact across configs (see launch_moe_gemv_fp4_symmetric).
  auto launch = [&](auto cta_n, auto threads) {
    fiv::dispatch_moe_gemv_interleaved_swiglu_group_size<Details, cta_n(), threads(), T>(
        const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
        expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k, group_size,
        activation_params, stream);
  };
  if (config == MoeGemvConfig::kCtaN16) {
    launch([] { return kCtaN16; }, [] { return kDefaultThreads; });
  } else if (config == MoeGemvConfig::kThreads64) {
    launch([] { return kDefaultCtaN; }, [] { return kThreads64; });
  } else {
    launch([] { return kDefaultCtaN; }, [] { return kDefaultThreads; });
  }
}

template void launch_moe_gemv_fp4_symmetric<half>(
    half const*, uint8_t const*, half const*, half const*, half*, int64_t const*, int const*, int,
    int64_t, int64_t, int64_t, int, int, MoeGemvConfig, cudaStream_t);
template void launch_moe_gemv_fp4_symmetric_interleaved_swiglu<half>(
    half const*, uint8_t const*, half const*, half const*, half*, int64_t const*, int const*, int,
    int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams, MoeGemvConfig, void*, cudaStream_t);
#ifdef ENABLE_BF16
template void launch_moe_gemv_fp4_symmetric<__nv_bfloat16>(
    __nv_bfloat16 const*, uint8_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    int64_t const*, int const*, int, int64_t, int64_t, int64_t, int, int, MoeGemvConfig, cudaStream_t);
template void launch_moe_gemv_fp4_symmetric_interleaved_swiglu<__nv_bfloat16>(
    __nv_bfloat16 const*, uint8_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    int64_t const*, int const*, int, int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams,
    MoeGemvConfig, void*, cudaStream_t);
#endif

}  // namespace moe_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
