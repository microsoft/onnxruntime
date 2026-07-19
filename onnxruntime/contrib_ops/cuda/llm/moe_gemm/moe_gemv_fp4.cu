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

// Opt-in interleaved MXFP4 GEMV path (env ORT_FP4_GEMV_INTERLEAVED=1). See moe_gemv_fp4.h.
// Parsed once via ORT's environment helper for consistent parsing/thread-safety. Off by
// default so the shipped single-pass ColumnMajor path stays byte-for-byte unchanged.
bool Fp4MoeGemvUseInterleaved() {
  static bool const enabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_GEMV_INTERLEAVED", 0) == 1;
  return enabled;
}

// Override for the interleaved path (env ORT_FP4_GEMV_INTERLEAVED_HALFACC=1). When set, the
// interleaved GEMV forces 16-bit (AccT=TypeA) accumulation for BOTH fp16 and bf16, overriding the
// default dtype-conditional Fp4GemvAccT policy (fp16->fp16 accum, bf16->fp32 accum). Forcing
// 16-bit on bf16 regresses bf16 accuracy, so this override is off by default.
bool Fp4MoeGemvInterleavedHalfAccum() {
  static bool const enabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_GEMV_INTERLEAVED_HALFACC", 0) == 1;
  return enabled;
}

// Interleaved-path tiling: a smaller CtaN than the default (8) to recover the occupancy the
// interleaved layout + fp32 accumulation cost. CtaN must be even (kernel static_assert) and a
// divisor that keeps n % (CtaN*kInterleave) == 0 for the target shapes. kInterleave = 4 here.
static constexpr int kInterleavedCtaN = 4;
static constexpr int kInterleavedThreads = 128;

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

// Interleaved Details. ColumnMajorInterleaved with TileSizeK = 64 and kElemBits = 4
// gives kInterleave = 128*8/(64*4) = 4, kStepK = 128/4 = 32, kThreadsPerInterleavedTile =
// 64/32 = 2. The linear Fp4I2FConverter is reused (UseInterleavedConverter = false): the
// preprocessor's layout-only steps 1-3 produce exactly the nibble order the linear converter
// expects. Weights for this layout are produced by the interleaved branch of
// QMoE::PrePackRepackFP4Weights (CUTLASS fpA_intB W4_A16 preprocessor, apply_bias_interleave=false).
template <typename T>
using Fp4KernelDetailsInterleaved =
    fiv::KernelDetails<typename Fp4ADetails<T>::Type, fiv::Fp4DetailsW, fiv::ColumnMajorInterleaved, false,
                       kTileSizeKFp4>;

// Interleaved-path accumulation policy (dtype-conditional). fp16 has a 10-bit mantissa, so 16-bit
// (half) accumulation over the interleaved kStepK=32 chains stays within tolerance and keeps
// register use low. bf16 has only 7 mantissa bits, so 16-bit accumulation loses too much precision
// (bf16 fails tolerance) and must accumulate in fp32. The ORT_FP4_GEMV_INTERLEAVED_HALFACC override
// forces 16-bit accum for BOTH dtypes.
template <typename T>
using Fp4GemvAccT = std::conditional_t<std::is_same<T, half>::value, half, float>;

// MXFP4 GEMV shape support. Mirrors is_moe_gemv_supported but for the non-interleaved
// ColumnMajor layout: kInterleave = 1, so n need only be divisible by the CtaN tile width
// selected by `config`, and the per-thread step is StepK = 128 / activation_bits = 8
// (not 128 / weight_bits).
bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                               MoeGemvConfig config) {
  if (sm < 80) {
    return false;
  }
  if (group_size != 16 && group_size != 32) {  // 32 = MXFP4 block size, 16 = NVFP4 block size
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
  if (Fp4MoeGemvUseInterleaved()) {
    // Interleaved path: ColumnMajorInterleaved (kInterleave = 4, kStepK = 32), fixed CtaN =
    // kInterleavedCtaN. Each block covers CtaN*kInterleave columns, and a complete interleaved
    // K-tile is kStepK*kThreadsPerInterleavedTile = 32*2 = 64 wide, so require
    // n % (CtaN*4) == 0 and k % 64 == 0. (gpt-oss-20b fc1 n=5760/k=2880 and fc2 n=2880/k=2880
    // both satisfy this.) `config` is ignored in this mode: CtaN/Threads are pinned to keep the
    // prepacked weight layout and the kernel dispatch in agreement.
    // The interleaved kStepK=32 tile is tied to the MXFP4 block-32 scale layout, so it only supports
    // group_size == 32; NVFP4 (block 16) must use the non-interleaved ColumnMajor path below.
    if (group_size != 32) {
      return false;
    }
    if (n % (kInterleavedCtaN * 4) != 0) {
      return false;
    }
    if (k % 64 != 0) {
      return false;
    }
    return true;
  }
  if (n % CtaNForConfig(config) != 0) {  // kInterleave = 1
    return false;
  }
  // StepK = 128 / activation_bits = 8; is_moe_gemv_fp4_supported requires k % group_size == 0 with
  // group_size >= 16 (16 for NVFP4, 32 for MXFP4), so k % 8 == 0 always holds.
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
  // Interleaved path (opt-in): ColumnMajorInterleaved layout + dtype-conditional accumulation +
  // smaller CtaN. The prepacked fc2 weights are in the interleaved layout, so the kernel must match.
  // CtaN/Threads are pinned (config ignored) so weights and kernel always agree. AccT follows the
  // Fp4GemvAccT policy (fp16->fp16 accum, bf16->fp32 accum); HALFACC forces 16-bit for both.
  if (Fp4MoeGemvUseInterleaved()) {
    using DetailsI = Fp4KernelDetailsInterleaved<T>;
    if (Fp4MoeGemvInterleavedHalfAccum()) {  // override: force 16-bit accum for all dtypes
      fiv::dispatch_moe_gemv_group_size<DetailsI, kInterleavedCtaN, kInterleavedThreads, T, T>(
          const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
          expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, n, k, group_size, stream);
    } else {
      fiv::dispatch_moe_gemv_group_size<DetailsI, kInterleavedCtaN, kInterleavedThreads, T, Fp4GemvAccT<T>>(
          const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
          expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, n, k, group_size, stream);
    }
    return;
  }
  using Details = Fp4KernelDetails<T>;
  // AccT follows the Fp4GemvAccT policy (fp16->fp16 accum, bf16->fp32 accum): bf16 has only 7
  // mantissa bits, so 16-bit accumulation over K loses too much precision and fails tolerance
  // (e.g. NVFP4 block-16 decode at k=512). CtaN/Threads remain pure parallelization/tiling knobs
  // and the accumulation dtype is identical for every config, so this sweep stays bit-exact.
  auto launch = [&](auto cta_n, auto threads) {
    fiv::dispatch_moe_gemv_group_size<Details, cta_n(), threads(), T, Fp4GemvAccT<T>>(
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
    cutlass_kernels::ActivationParams activation_params, MoeGemvConfig config, cudaStream_t stream) {
  ORT_UNUSED_PARAMETER(sm);
  // Interleaved path (opt-in): ColumnMajorInterleaved layout + dtype-conditional accumulation +
  // smaller CtaN, fusing SwiGLU. Takes precedence over the split-K path so the kernel matches the
  // interleaved prepacked fc1 weights. CtaN/Threads pinned (config ignored). AccT follows the
  // Fp4GemvAccT policy: fp16->fp16 accum since fp16's mantissa tolerates it; bf16->fp32 accum
  // since 16-bit accum fails bf16 tolerance. HALFACC forces 16-bit for both dtypes.
  if (Fp4MoeGemvUseInterleaved()) {
    using DetailsI = Fp4KernelDetailsInterleaved<T>;
    if (Fp4MoeGemvInterleavedHalfAccum()) {  // override: force 16-bit accum for all dtypes
      fiv::dispatch_moe_gemv_interleaved_swiglu_group_size<DetailsI, kInterleavedCtaN, kInterleavedThreads, T, T>(
          const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
          expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k, group_size,
          activation_params, stream);
    } else {
      fiv::dispatch_moe_gemv_interleaved_swiglu_group_size<DetailsI, kInterleavedCtaN, kInterleavedThreads, T,
                                                           Fp4GemvAccT<T>>(
          const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
          expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k, group_size,
          activation_params, stream);
    }
    return;
  }
  using Details = Fp4KernelDetails<T>;
  // AccT follows the Fp4GemvAccT policy (fp16->fp16, bf16->fp32); see launch_moe_gemv_fp4_symmetric.
  // The CtaN/Threads sweep stays bit-exact across configs since the accumulation dtype is fixed.
  auto launch = [&](auto cta_n, auto threads) {
    fiv::dispatch_moe_gemv_interleaved_swiglu_group_size<Details, cta_n(), threads(), T, Fp4GemvAccT<T>>(
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
    int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams, MoeGemvConfig, cudaStream_t);
#ifdef ENABLE_BF16
template void launch_moe_gemv_fp4_symmetric<__nv_bfloat16>(
    __nv_bfloat16 const*, uint8_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    int64_t const*, int const*, int, int64_t, int64_t, int64_t, int, int, MoeGemvConfig, cudaStream_t);
template void launch_moe_gemv_fp4_symmetric_interleaved_swiglu<__nv_bfloat16>(
    __nv_bfloat16 const*, uint8_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    int64_t const*, int const*, int, int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams,
    MoeGemvConfig, cudaStream_t);
#endif

}  // namespace moe_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
