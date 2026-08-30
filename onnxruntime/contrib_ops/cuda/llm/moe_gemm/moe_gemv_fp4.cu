// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.h"

#include <cuda_fp16.h>
#include <type_traits>

// Shared INT/FP4 profiled-shape thresholds.
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv.h"
// Shared device-side kernels + launch/dispatch helpers (fpA_intB_gemv namespace).
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv_device.cuh"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
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
  const static bool enabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_GEMV_INTERLEAVED", 0) == 1;
  return enabled;
}

// Override for the interleaved path (env ORT_FP4_GEMV_INTERLEAVED_HALFACC=1). When set, the
// interleaved GEMV forces 16-bit (AccT=TypeA) accumulation for BOTH fp16 and bf16, overriding the
// default dtype-conditional Fp4GemvAccT policy (fp16->fp16 accum, bf16->fp32 accum). Forcing
// 16-bit on bf16 regresses bf16 accuracy, so this override is off by default.
bool Fp4MoeGemvInterleavedHalfAccum() {
  const static bool enabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_GEMV_INTERLEAVED_HALFACC", 0) == 1;
  return enabled;
}

// Interleaved-path tiling: a smaller CtaN than the default (8) to recover the occupancy the
// interleaved layout + fp32 accumulation cost. CtaN must be even (kernel static_assert) and a
// divisor that keeps n % (CtaN*kInterleave) == 0 for the target shapes. kInterleave = 4 here.
static constexpr int kInterleavedCtaN = 4;
static constexpr int kInterleavedThreads = 128;

// Opt-out for the shape-derived default tiling (env ORT_FP4_GEMV_DEFAULT_TILING=0), which
// restores the fixed kDefaultCtaN/kDefaultThreads tiling for every shape.
static bool Fp4MoeGemvUseDefaultTilingHeuristic() {
  static bool const enabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_GEMV_DEFAULT_TILING", 1) == 1;
  return enabled;
}

// Blocks per SM required before a 64-thread block is worth it. The grid is fixed by
// (expanded_num_rows, n / CtaN) and does not depend on Threads, so halving the block size halves
// the threads each block contributes to residency. These kernels are register-limited (about 72
// registers/thread on sm_90, so roughly 900 resident threads/SM), which a 128-thread block reaches
// with ~7 resident blocks and a 64-thread block only with ~14. Requiring 16 blocks/SM therefore
// keeps the SM just as full while halving the epilogue's reduction width.
static constexpr int64_t kMinBlocksPerSmForThreads64 = 16;

MoeGemvConfig Fp4MoeGemvDefaultConfig(int64_t expanded_num_rows, int64_t n, int64_t k,
                                      int multi_processor_count) {
  // The interleaved path pins its own CtaN/Threads and ignores `config`.
  if (Fp4MoeGemvUseInterleaved() || !Fp4MoeGemvUseDefaultTilingHeuristic()) {
    return MoeGemvConfig::kDefault;
  }
  // Each block walks K in strides of CtaK = StepK * Threads, with StepK = 128 / activation_bits.
  constexpr int64_t kStepK = 128 / 16;
  constexpr int64_t kDefaultCtaK = kStepK * kDefaultThreads;

  // (a) Idle threads. When CtaK > k the tail of every block never enters the K loop at all, so a
  //     128-thread block leaves half its threads doing nothing (NVFP4 fc2 has k = inter_size,
  //     e.g. 512 against CtaK = 1024). Narrowing the block is a pure win here.
  if (kDefaultCtaK > k) {
    return MoeGemvConfig::kThreads64;
  }

  // (b) Epilogue cost. The MAC work per block is fixed by (CtaN, k) regardless of Threads, but the
  //     epilogue reduces partial sums across Threads/32 warps through shared memory. A narrower
  //     block does the same math with half the barriers and a shallower reduction tree, so prefer
  //     it whenever the grid is large enough that the SM still fills up (see the constant above).
  const int64_t blocks = expanded_num_rows * (n / kDefaultCtaN);
  if (blocks >= kMinBlocksPerSmForThreads64 * multi_processor_count) {
    return MoeGemvConfig::kThreads64;
  }
  return MoeGemvConfig::kDefault;
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

// Same interleaved Details, but with UseInterleavedConverter = true so the e2m1 decoder inverts
// the `[e0,e2,e4,e6,e1,e3,e5,e7]` nibble pair-interleave on the fly. That is preprocessor step 4
// (interleave_without_bias), i.e. exactly the layout the SM80 grouped GEMM consumes. Selecting
// this variant lets the decode GEMV read the *same* pre-packed buffer the SM80 prefill uses, so
// the MXFP4 expert weights are stored once instead of twice (~9 GiB saved for gpt-oss-20b).
// The un-permutation is a compile-time index remap, so it is free at runtime.
template <typename T>
using Fp4KernelDetailsSm80Pair =
    fiv::KernelDetails<typename Fp4ADetails<T>::Type, fiv::Fp4DetailsW, fiv::ColumnMajorInterleaved, true,
                       kTileSizeKFp4>;

// Carries a Details type into a generic lambda so the interleaved and pair-interleaved launch
// bodies can be shared instead of duplicated.
template <typename U>
struct TypeTag {
  using type = U;
};

// Interleaved-path accumulation policy (dtype-conditional). fp16 has a 10-bit mantissa, so 16-bit
// (half) accumulation over the interleaved kStepK=32 chains stays within tolerance and keeps
// register use low. bf16 has only 7 mantissa bits, so 16-bit accumulation loses too much precision
// (bf16 fails tolerance) and must accumulate in fp32. The ORT_FP4_GEMV_INTERLEAVED_HALFACC override
// forces 16-bit accum for BOTH dtypes.
template <typename T>
using Fp4GemvAccT = std::conditional_t<std::is_same<T, half>::value, half, float>;

__device__ __forceinline__ float DecodeE4M3Fn(uint8_t code) {
  const int sign = code & 0x80;
  const int exponent = (code >> 3) & 0x0f;
  const int mantissa = code & 0x07;
  if ((code & 0x7f) == 0) {
    return sign ? -0.0f : 0.0f;
  }
  if (exponent == 0x0f && mantissa == 0x07) {
    return __int_as_float(0x7fffffff);
  }
  const float value = exponent == 0
                          ? ldexpf(static_cast<float>(mantissa), -9)
                          : ldexpf(1.0f + static_cast<float>(mantissa) * 0.125f, exponent - 7);
  return sign ? -value : value;
}

// NVFP4 schema weights are [E, K, N/2], with adjacent output columns in the two nibbles of
// each byte. A generic remapped ColumnMajor iterator would make every lane gather strided bytes.
// Instead, each warp owns 16 adjacent output columns: its eight N lanes load eight contiguous
// packed bytes for each of four K lanes, then reduce the four partial K streams in-register.
// This keeps the raw initializer directly consumable while giving each K row one coalesced N slice.
template <typename T, bool FusedSwiGlu, bool EnableBias>
__global__ void MoeGemvFp4RawNPackedKernel(
    const T* act, const uint8_t* weight, const uint8_t* block_scales, const float* global_scales,
    const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t weight_expert_stride, int64_t scale_expert_stride, int n, int k,
    cutlass_kernels::ActivationParams activation_params,
    const int* permuted_row_to_source_row, int num_rows) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = 4;
  constexpr int kColsPerWarp = 16;
  constexpr unsigned kFullMask = 0xffffffffu;

  const int row = static_cast<int>(blockIdx.x);
  int expert = permuted_row_to_expert != nullptr ? permuted_row_to_expert[row] : 0;
#pragma unroll 1
  for (int e = 0; e < num_experts && permuted_row_to_expert == nullptr; ++e) {
    if (row >= static_cast<int>(expert_first_token_offset[e + 1])) {
      expert = e + 1;
      continue;
    }
    break;
  }
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int n_pair = lane % (kColsPerWarp / 2);
  const int k_lane = lane / (kColsPerWarp / 2);
  const int n0 = (static_cast<int>(blockIdx.y) * kWarpsPerBlock + warp) * kColsPerWarp + n_pair * 2;
  const bool valid_n = n0 < n;
  const int source_row = permuted_row_to_source_row ? permuted_row_to_source_row[row] % num_rows : row;

  const T* row_act = act + static_cast<int64_t>(source_row) * k;
  const uint8_t* expert_weight = weight + static_cast<int64_t>(expert) * weight_expert_stride;
  const uint8_t* expert_scales = block_scales + static_cast<int64_t>(expert) * scale_expert_stride;
  const float global_scale = global_scales[expert];
  const T* expert_bias = EnableBias ? bias + static_cast<int64_t>(expert) * n : nullptr;

  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int k_idx = k_lane; k_idx < k; k_idx += 4) {
    float a = n_pair == 0 ? static_cast<float>(row_act[k_idx]) : 0.0f;
    a = __shfl_sync(kFullMask, a, k_lane * (kColsPerWarp / 2));
    if (valid_n) {
      const uint8_t packed = expert_weight[static_cast<int64_t>(k_idx) * (n / 2) + n0 / 2];
      const int k_blocks = k / 16;
      const T scale0 = static_cast<T>(
          DecodeE4M3Fn(expert_scales[static_cast<int64_t>(n0) * k_blocks + k_idx / 16]) * global_scale);
      const T scale1 = static_cast<T>(
          DecodeE4M3Fn(expert_scales[static_cast<int64_t>(n0 + 1) * k_blocks + k_idx / 16]) * global_scale);
      const T weight0 = fiv::Fp4I2FConverter<T>::decode(packed & 0x0f);
      const T weight1 = fiv::Fp4I2FConverter<T>::decode(packed >> 4);
      const T scaled_weight0 = static_cast<T>(static_cast<float>(weight0) * static_cast<float>(scale0));
      const T scaled_weight1 = static_cast<T>(static_cast<float>(weight1) * static_cast<float>(scale1));
      acc0 += static_cast<float>(scaled_weight0) * a;
      acc1 += static_cast<float>(scaled_weight1) * a;
    }
  }

  acc0 += __shfl_xor_sync(kFullMask, acc0, 8);
  acc1 += __shfl_xor_sync(kFullMask, acc1, 8);
  acc0 += __shfl_xor_sync(kFullMask, acc0, 16);
  acc1 += __shfl_xor_sync(kFullMask, acc1, 16);
  if (k_lane != 0 || !valid_n) {
    return;
  }

  if constexpr (EnableBias) {
    acc0 += static_cast<float>(expert_bias[n0]);
    acc1 += static_cast<float>(expert_bias[n0 + 1]);
  }
  if constexpr (FusedSwiGlu) {
    const float* alpha = activation_params.swiglu_alpha;
    const float* beta = activation_params.swiglu_beta;
    const float* limit = activation_params.swiglu_limit;
    const float activation_alpha = alpha ? alpha[expert] : activation_params.alpha;
    const float activation_beta = beta ? beta[expert] : activation_params.beta;
    const float activation_limit = limit ? limit[expert] : activation_params.limit;
    if (isfinite(activation_limit)) {
      acc0 = fminf(acc0, activation_limit);
      acc1 = fminf(fmaxf(acc1, -activation_limit), activation_limit);
    }
    acc1 += activation_beta;
    const float sigmoid = 1.0f / (1.0f + expf(-activation_alpha * acc0));
    out[static_cast<int64_t>(row) * (n / 2) + n0 / 2] = static_cast<T>(acc0 * sigmoid * acc1);
  } else {
    out[static_cast<int64_t>(row) * n + n0] = static_cast<T>(acc0);
    out[static_cast<int64_t>(row) * n + n0 + 1] = static_cast<T>(acc1);
  }
#endif
}

template <typename T, bool FusedSwiGlu>
void LaunchMoeGemvFp4RawNPacked(
    const T* act, const uint8_t* weight, const uint8_t* block_scales, const float* global_scales,
    const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t n, int64_t k, cutlass_kernels::ActivationParams activation_params,
    const int* permuted_row_to_source_row, int64_t num_rows, cudaStream_t stream) {
  constexpr int kThreads = 128;
  constexpr int kColsPerBlock = 64;
  const int64_t weight_expert_stride = n * k / 2;
  const int64_t scale_expert_stride = n * (k / 16);
  const dim3 grid(static_cast<unsigned>(expanded_num_rows), static_cast<unsigned>((n + kColsPerBlock - 1) / kColsPerBlock));
  if (bias != nullptr) {
    MoeGemvFp4RawNPackedKernel<T, FusedSwiGlu, true><<<grid, kThreads, 0, stream>>>(
        act, weight, block_scales, global_scales, bias, out,
        expert_first_token_offset, permuted_row_to_expert, num_experts,
        weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k), activation_params,
        permuted_row_to_source_row, static_cast<int>(num_rows));
  } else {
    MoeGemvFp4RawNPackedKernel<T, FusedSwiGlu, false><<<grid, kThreads, 0, stream>>>(
        act, weight, block_scales, global_scales, bias, out,
        expert_first_token_offset, permuted_row_to_expert, num_experts,
        weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k), activation_params,
        permuted_row_to_source_row, static_cast<int>(num_rows));
  }
}

// MXFP4 GEMV shape support. Mirrors is_moe_gemv_supported but for the non-interleaved
// ColumnMajor layout: kInterleave = 1, so n need only be divisible by the CtaN tile width
// selected by `config`, and the per-thread step is StepK = 128 / activation_bits = 8
// (not 128 / weight_bits).
bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                               MoeGemvConfig config, bool sm80_pair_interleaved, bool raw_n_packed) {
  if (sm < 80) {
    return false;
  }
  if (group_size != 16 && group_size != 32) {  // 32 = MXFP4 block size, 16 = NVFP4 block size
    return false;
  }
  if (k % group_size != 0) {
    return false;
  }
  if (expanded_num_rows <= 0 || expanded_num_rows > kMaxProfiledExpandedRowsFp4) {
    return false;
  }
  if (n < kMinProfiledProblemDim || k < kMinProfiledProblemDim) {
    return false;
  }
  if (expanded_num_rows > kMaxProfiledExpandedRowsForSmallProblemDim &&
      (n < kMinProfiledProblemDimForExpandedRowsAbove4 || k < kMinProfiledProblemDimForExpandedRowsAbove4)) {
    return false;
  }
  if (raw_n_packed) {
    return group_size == 16 && n % 2 == 0;
  }
  if (sm80_pair_interleaved || Fp4MoeGemvUseInterleaved()) {
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

bool is_moe_gemv_fp4_sm80_layout_supported(int64_t n, int64_t k, int group_size) {
  // Same constraints the interleaved branch of is_moe_gemv_fp4_supported enforces, minus the
  // row-count bounds (which PrePack cannot know): the interleaved kStepK=32 tile is tied to the
  // MXFP4 block-32 scale layout, each block covers kInterleavedCtaN*kInterleave columns, and a
  // complete interleaved K-tile is kStepK*kThreadsPerInterleavedTile = 64 wide.
  return group_size == 32 && n % (kInterleavedCtaN * 4) == 0 && k % 64 == 0;
}

bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                               MoeGemvConfig config) {
  return is_moe_gemv_fp4_supported(sm, expanded_num_rows, n, k, group_size, config,
                                   /*sm80_pair_interleaved=*/false, /*raw_n_packed=*/false);
}

bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size) {
  return is_moe_gemv_fp4_supported(sm, expanded_num_rows, n, k, group_size, MoeGemvConfig::kDefault);
}

template <typename T>
void launch_moe_gemv_fp4_symmetric(const T* act, const uint8_t* weight, const T* scales,
                                   const uint8_t* raw_block_scales, const float* raw_global_scales,
                                   const T* bias, T* out,
                                   const int64_t* expert_first_token_offset, const int* permuted_row_to_expert,
                                   int num_experts, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                                   int sm, MoeGemvConfig config, bool sm80_pair_interleaved, bool raw_n_packed,
                                   cudaStream_t stream) {
  ORT_UNUSED_PARAMETER(sm);
  if (raw_n_packed) {
    ORT_ENFORCE(group_size == 16, "Raw N-packed FP4 GEMV is NVFP4-only.");
    ORT_ENFORCE(raw_block_scales != nullptr && raw_global_scales != nullptr,
                "Raw N-packed NVFP4 GEMV requires block and global scales.");
    LaunchMoeGemvFp4RawNPacked<T, false>(
        act, weight, raw_block_scales, raw_global_scales, bias, out,
        expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, n, k, cutlass_kernels::ActivationParams{}, nullptr, expanded_num_rows, stream);
    return;
  }
  // Interleaved path: ColumnMajorInterleaved layout + dtype-conditional accumulation + smaller
  // CtaN. Taken either via the opt-in env knob or because the caller pre-packed a single
  // SM80-grouped-GEMM buffer (sm80_pair_interleaved) that this kernel un-permutes while decoding.
  // The prepacked fc2 weights are in the interleaved layout, so the kernel must match.
  // CtaN/Threads are pinned (config ignored) so weights and kernel always agree. AccT follows the
  // Fp4GemvAccT policy (fp16->fp16 accum, bf16->fp32 accum); HALFACC forces 16-bit for both.
  if (sm80_pair_interleaved || Fp4MoeGemvUseInterleaved()) {
    auto launch_interleaved = [&](auto details_tag) {
      using DetailsI = typename decltype(details_tag)::type;
      if (Fp4MoeGemvInterleavedHalfAccum()) {  // override: force 16-bit accum for all dtypes
        fiv::dispatch_moe_gemv_group_size<DetailsI, kInterleavedCtaN, kInterleavedThreads, T, T>(
            const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
            expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, n, k, group_size,
            stream);
      } else {
        fiv::dispatch_moe_gemv_group_size<DetailsI, kInterleavedCtaN, kInterleavedThreads, T, Fp4GemvAccT<T>>(
            const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
            expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, n, k, group_size,
            stream);
      }
    };
    if (sm80_pair_interleaved) {
      launch_interleaved(TypeTag<Fp4KernelDetailsSm80Pair<T>>{});
    } else {
      launch_interleaved(TypeTag<Fp4KernelDetailsInterleaved<T>>{});
    }
    return;
  }
  using Details = Fp4KernelDetails<T>;
  // AccT follows the Fp4GemvAccT policy (fp16->fp16 accum, bf16->fp32 accum): bf16 has only 7
  // mantissa bits, so 16-bit accumulation over K loses too much precision and fails tolerance
  // (e.g. NVFP4 block-16 decode at k=512). CtaN/Threads remain pure parallelization/tiling knobs
  // and the accumulation dtype is identical for every config, so every config computes the same
  // dot products; Threads additionally sets the K partition, so it perturbs the summation order.
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
    const T* act, const uint8_t* weight, const T* scales, const uint8_t* raw_block_scales,
    const float* raw_global_scales, const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k, int group_size, int sm,
    cutlass_kernels::ActivationParams activation_params, MoeGemvConfig config, bool sm80_pair_interleaved,
    bool raw_n_packed,
    const int* permuted_row_to_source_row, int64_t num_rows, cudaStream_t stream) {
  ORT_UNUSED_PARAMETER(sm);
  if (raw_n_packed) {
    ORT_ENFORCE(group_size == 16, "Raw N-packed FP4 GEMV is NVFP4-only.");
    ORT_ENFORCE(raw_block_scales != nullptr && raw_global_scales != nullptr,
                "Raw N-packed NVFP4 GEMV requires block and global scales.");
    LaunchMoeGemvFp4RawNPacked<T, true>(
        act, weight, raw_block_scales, raw_global_scales, bias, out,
        expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size * 2, k, activation_params, permuted_row_to_source_row, num_rows, stream);
    return;
  }
  // Interleaved path: ColumnMajorInterleaved layout + dtype-conditional accumulation + smaller
  // CtaN, fusing SwiGLU. Taken either via the opt-in env knob or because the caller pre-packed a
  // single SM80-grouped-GEMM buffer (sm80_pair_interleaved). SwiGLU fusion is orthogonal to the
  // weight layout: it only concerns the fc1 output-column (gate/value) order, so both variants
  // fuse it identically. Takes precedence over the split-K path so the kernel matches the
  // prepacked fc1 weights. CtaN/Threads pinned (config ignored). AccT follows the Fp4GemvAccT
  // policy: fp16->fp16 accum since fp16's mantissa tolerates it; bf16->fp32 accum since 16-bit
  // accum fails bf16 tolerance. HALFACC forces 16-bit for both dtypes.
  if (sm80_pair_interleaved || Fp4MoeGemvUseInterleaved()) {
    auto launch_interleaved = [&](auto details_tag) {
      using DetailsI = typename decltype(details_tag)::type;
      if (Fp4MoeGemvInterleavedHalfAccum()) {  // override: force 16-bit accum for all dtypes
        fiv::dispatch_moe_gemv_interleaved_swiglu_group_size<DetailsI, kInterleavedCtaN, kInterleavedThreads, T, T>(
            const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
            expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k,
            group_size, activation_params, permuted_row_to_source_row, static_cast<int>(num_rows), stream);
      } else {
        fiv::dispatch_moe_gemv_interleaved_swiglu_group_size<DetailsI, kInterleavedCtaN, kInterleavedThreads, T,
                                                             Fp4GemvAccT<T>>(
            const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
            expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k,
            group_size, activation_params, permuted_row_to_source_row, static_cast<int>(num_rows), stream);
      }
    };
    if (sm80_pair_interleaved) {
      launch_interleaved(TypeTag<Fp4KernelDetailsSm80Pair<T>>{});
    } else {
      launch_interleaved(TypeTag<Fp4KernelDetailsInterleaved<T>>{});
    }
    return;
  }
  using Details = Fp4KernelDetails<T>;
  // AccT follows the Fp4GemvAccT policy (fp16->fp16, bf16->fp32); see launch_moe_gemv_fp4_symmetric.
  // The accumulation dtype is fixed across the CtaN/Threads sweep; Threads still changes the K
  // partition, so it is a tiling knob, not a bit-exact one.
  auto launch = [&](auto cta_n, auto threads) {
    fiv::dispatch_moe_gemv_interleaved_swiglu_group_size<Details, cta_n(), threads(), T, Fp4GemvAccT<T>>(
        const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
        expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k, group_size,
        activation_params, permuted_row_to_source_row, static_cast<int>(num_rows), stream);
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
    const half*, const uint8_t*, const half*, const uint8_t*, const float*, const half*, half*,
    const int64_t*, const int*, int,
    int64_t, int64_t, int64_t, int, int, MoeGemvConfig, bool, bool, cudaStream_t);
template void launch_moe_gemv_fp4_symmetric_interleaved_swiglu<half>(
    const half*, const uint8_t*, const half*, const uint8_t*, const float*, const half*, half*,
    const int64_t*, const int*, int,
    int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams, MoeGemvConfig, bool, bool,
    const int*, int64_t, cudaStream_t);
#ifdef ENABLE_BF16
template void launch_moe_gemv_fp4_symmetric<__nv_bfloat16>(
    const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const uint8_t*, const float*,
    const __nv_bfloat16*, __nv_bfloat16*,
    const int64_t*, const int*, int, int64_t, int64_t, int64_t, int, int, MoeGemvConfig, bool, bool, cudaStream_t);
template void launch_moe_gemv_fp4_symmetric_interleaved_swiglu<__nv_bfloat16>(
    const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const uint8_t*, const float*,
    const __nv_bfloat16*, __nv_bfloat16*,
    const int64_t*, const int*, int, int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams,
    MoeGemvConfig, bool, bool, const int*, int64_t, cudaStream_t);
#endif

}  // namespace moe_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
