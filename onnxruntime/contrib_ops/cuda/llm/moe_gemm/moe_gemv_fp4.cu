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
// Each word is un-permuted with a few register ops before the shared packed decode.
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
                          ? static_cast<float>(mantissa) * 0.001953125f
                          : __int_as_float(((exponent + 120) << 23) | (mantissa << 20));
  return sign ? -value : value;
}

// NVFP4 schema weights are [E, K, N/2], with adjacent output columns in the two nibbles of
// each byte. Transpose an N16/K1024 tile in shared memory so 64 independent K groups can
// reuse each block scale for 16 values. Padding distributes compute-time reads across banks.
template <typename T, bool FusedSwiGlu, bool EnableBias, bool WeightsRowMajor>
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
  constexpr int kColsPerThread = 8;
  constexpr int kNLanes = 2;
  constexpr int kKLanes = kWarpsPerBlock * kWarpSize / kNLanes;
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
  const int n_lane = lane % kNLanes;
  const int k_lane = threadIdx.x / kNLanes;
  const int n_base = (static_cast<int>(blockIdx.y) * kNLanes + n_lane) * kColsPerThread;
  const int source_row = permuted_row_to_source_row ? permuted_row_to_source_row[row] % num_rows : row;

  const T* row_act = act + static_cast<int64_t>(source_row) * k;
  const uint8_t* expert_weight = weight + static_cast<int64_t>(expert) * weight_expert_stride;
  const uint8_t* expert_scales = block_scales + static_cast<int64_t>(expert) * scale_expert_stride;
  const float global_scale = global_scales[expert];
  const T* expert_bias = EnableBias ? bias + static_cast<int64_t>(expert) * n : nullptr;

  __shared__ uint32_t weight_tile[kKLanes][kNLanes][17];
  float accumulators[kColsPerThread] = {};
  for (int k_tile = 0; k_tile < k; k_tile += kKLanes * 16) {
    if constexpr (WeightsRowMajor) {
      constexpr int kPackedRowBytes = kKLanes * 16 / 2;
      constexpr int kVectorsPerRow = kPackedRowBytes / sizeof(uint4);
      __shared__ uint4 packed_tile[kNLanes * kColsPerThread][kVectorsPerRow];
      const int64_t tile_n = static_cast<int64_t>(blockIdx.y) * kNLanes * kColsPerThread;
      for (int chunk = threadIdx.x; chunk < kNLanes * kColsPerThread * kVectorsPerRow;
           chunk += kWarpSize * kWarpsPerBlock) {
        const int column_offset = chunk / kVectorsPerRow;
        const int vector_offset = chunk % kVectorsPerRow;
        const int byte_offset = vector_offset * sizeof(uint4);
        const int remaining = k / 2 - k_tile / 2 - byte_offset;
        uint4 packed_bytes = {};
        if (tile_n + column_offset < n && remaining > 0) {
          const uint8_t* source = expert_weight + (tile_n + column_offset) * (k / 2) +
                                  k_tile / 2 + byte_offset;
          if (remaining >= static_cast<int>(sizeof(uint4)) &&
              (reinterpret_cast<uintptr_t>(source) % alignof(uint4)) == 0) {
            packed_bytes = *reinterpret_cast<const uint4*>(source);
          } else {
#pragma unroll
            for (int byte = 0; byte < static_cast<int>(sizeof(uint4)); ++byte) {
              if (byte < remaining) {
                reinterpret_cast<uint8_t*>(&packed_bytes)[byte] = source[byte];
              }
            }
          }
        }
        packed_tile[column_offset][vector_offset] = packed_bytes;
      }
      __syncthreads();
      for (int tile_row = threadIdx.x; tile_row < kKLanes * 16;
           tile_row += kWarpSize * kWarpsPerBlock) {
        uint2 packed = {};
#pragma unroll
        for (int vector = 0; vector < kNLanes; ++vector) {
          uint32_t word = 0;
#pragma unroll
          for (int column_offset = 0; column_offset < kColsPerThread; ++column_offset) {
            const int column = vector * kColsPerThread + column_offset;
            const uint8_t code_pair = reinterpret_cast<const uint8_t*>(packed_tile[column])[tile_row / 2];
            const uint32_t code = (code_pair >> ((tile_row & 1) * 4)) & 15;
            word |= code << (column_offset * 4);
          }
          reinterpret_cast<uint32_t*>(&packed)[vector] = word;
        }
        weight_tile[tile_row / 16][0][tile_row % 16] = packed.x;
        weight_tile[tile_row / 16][1][tile_row % 16] = packed.y;
      }
    } else {
      for (int tile_row = threadIdx.x; tile_row < kKLanes * 16; tile_row += kWarpSize * kWarpsPerBlock) {
        const int tile_n = static_cast<int>(blockIdx.y) * kNLanes * kColsPerThread;
        uint2 packed = {};
        if (k_tile + tile_row < k) {
          const int reduction_column = k_tile + tile_row;
          const uint8_t* weights = expert_weight + static_cast<int64_t>(reduction_column) * (n / 2);
          if (n % (kNLanes * kColsPerThread) == 0) {
            packed = *reinterpret_cast<const uint2*>(weights + tile_n / 2);
          } else {
#pragma unroll
            for (int vector = 0; vector < kNLanes; ++vector) {
              uint32_t word = 0;
#pragma unroll
              for (int pair = 0; pair < kColsPerThread / 2; ++pair) {
                const int column = tile_n + vector * kColsPerThread + pair * 2;
                if (column < n) {
                  word |= static_cast<uint32_t>(weights[column / 2]) << (pair * 8);
                }
              }
              reinterpret_cast<uint32_t*>(&packed)[vector] = word;
            }
          }
        }
        weight_tile[tile_row / 16][0][tile_row % 16] = packed.x;
        weight_tile[tile_row / 16][1][tile_row % 16] = packed.y;
      }
    }
    __syncthreads();
    const int k_base = k_tile + k_lane * 16;
    if (k_base < k) {
      alignas(4) T scales[kColsPerThread];
#pragma unroll
      for (int col = 0; col < kColsPerThread; ++col) {
        scales[col] = n_base + col < n
                          ? static_cast<T>(DecodeE4M3Fn(expert_scales[static_cast<int64_t>(n_base + col) *
                                                                          (k / 16) +
                                                                      k_base / 16]) *
                                           global_scale)
                          : static_cast<T>(0.0f);
      }
      alignas(16) T activations[16];
      reinterpret_cast<uint4*>(activations)[0] = reinterpret_cast<const uint4*>(row_act + k_base)[0];
      reinterpret_cast<uint4*>(activations)[1] = reinterpret_cast<const uint4*>(row_act + k_base)[1];
#pragma unroll
      for (int element = 0; element < 16; ++element) {
        uint32_t packed = weight_tile[k_lane][n_lane][element];
        const float activation = static_cast<float>(activations[element]);
        alignas(4) T decoded[kColsPerThread];
        fiv::Fp4I2FConverter<T>::template convert<kColsPerThread>(&packed, decoded);
        using PackedT = std::conditional_t<std::is_same_v<T, half>, half2, __nv_bfloat162>;
#pragma unroll
        for (int pair = 0; pair < kColsPerThread / 2; ++pair) {
          reinterpret_cast<PackedT*>(decoded)[pair] =
              __hmul2(reinterpret_cast<const PackedT*>(decoded)[pair],
                      reinterpret_cast<const PackedT*>(scales)[pair]);
        }
#pragma unroll
        for (int col = 0; col < kColsPerThread; ++col) {
          accumulators[col] += static_cast<float>(decoded[col]) * activation;
        }
      }
    }
    __syncthreads();
  }

  __shared__ float partials[kWarpsPerBlock][kColsPerThread][kNLanes];
#pragma unroll
  for (int col = 0; col < kColsPerThread; ++col) {
    accumulators[col] += __shfl_xor_sync(kFullMask, accumulators[col], 2);
    accumulators[col] += __shfl_xor_sync(kFullMask, accumulators[col], 4);
    accumulators[col] += __shfl_xor_sync(kFullMask, accumulators[col], 8);
    accumulators[col] += __shfl_xor_sync(kFullMask, accumulators[col], 16);
    if (lane < kNLanes) {
      partials[warp][col][n_lane] = accumulators[col];
    }
  }
  __syncthreads();
  if (k_lane != 0 || n_base >= n) {
    return;
  }

#pragma unroll
  for (int col = 0; col < kColsPerThread; col += 2) {
    const int n0 = n_base + col;
    if (n0 >= n) {
      continue;
    }
    float acc0 = 0.0f;
    float acc1 = 0.0f;
#pragma unroll
    for (int partial = 0; partial < kWarpsPerBlock; ++partial) {
      acc0 += partials[partial][col][n_lane];
      acc1 += partials[partial][col + 1][n_lane];
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
  }
#endif
}

// ---------------------------------------------------------------------------------------------
// Decode-specialised NVFP4 GEMV for row-major [E, N, K/2] weights (S=1 / small expanded row count).
//
// The baseline kernel above stages a 16-column x 1024-K tile through shared memory with a
// load -> barrier -> repack -> barrier -> compute cycle and no prefetch, so each CTA is latency bound
// (about 3x off the HBM roofline for the Qwen3.8 FC1/FC2 shapes). This kernel streams the weights
// straight from global memory into registers with independent 128-bit loads, decodes FP4 in
// registers, and keeps the activation row in shared memory as FP32.
//
// It reproduces the baseline floating-point summation order exactly so results are bit-identical:
// the baseline reduces over K as 64 "lanes" L = (k / 16) % 64, each lane accumulating its 16-element
// groups sequentially across 1024-wide K tiles, then a butterfly over the 16 lanes of a warp and a
// sequential sum over the 4 warps. Here a thread owns one output column and the lane pair
// L = 16w + 2j, 2j + 1 (one 16-byte load per K tile); adding the two chains is the baseline's first
// butterfly step, shuffles over j are the remaining three, and the four partials w = 0..3 are summed
// in order by the same thread, so no cross-warp reduction or barrier is needed after the main loop.
template <typename T>
__device__ __forceinline__ float PackedLowToFloat(uint32_t word) {
  if constexpr (std::is_same_v<T, half>) {
    return __half2float(__ushort_as_half(static_cast<uint16_t>(word & 0xffffu)));
  } else {
    return __uint_as_float(word << 16);
  }
}

template <typename T>
__device__ __forceinline__ float PackedHighToFloat(uint32_t word) {
  if constexpr (std::is_same_v<T, half>) {
    return __half2float(__ushort_as_half(static_cast<uint16_t>(word >> 16)));
  } else {
    return __uint_as_float(word & 0xffff0000u);
  }
}

__device__ __forceinline__ uint4 LoadWeightStreaming(const void* ptr) {
  uint4 value;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
      : "l"(ptr));
  return value;
}

static constexpr int kDecodeMaxK = 3072;  // three K tiles in flight per lane pair

// Tiling of the decode kernels: warps per CTA, output columns per thread and the min-blocks-per-SM hint that
// caps registers. A CTA covers 4 * Warps * Cols columns of one expanded row. FC1 (fused SwiGLU) is ALU bound,
// FC2 is latency bound and wants one resident wave of 2-column threads. (Measured on H200 for N=1280/K=2560
// and N=2560/K=640 at 10 expanded rows.)
#ifndef GEMV_FC1_WARPS
#define GEMV_FC1_WARPS 4
#define GEMV_FC1_COLS 2
#define GEMV_FC1_MINB 4
#define GEMV_FC2_WARPS 2
#define GEMV_FC2_COLS 2
#define GEMV_FC2_MINB 8
#endif

// Opt-out for the decode-specialised row-major NVFP4 GEMV (env ORT_FP4_GEMV_DECODE_KERNEL=0).
static bool Fp4MoeGemvUseDecodeKernel() {
  static const bool enabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_FP4_GEMV_DECODE_KERNEL", 1) != 0;
  return enabled;
}

// bf16 decode without lookup tables: place the three e2m1 magnitude bits (e1 e0 m) at bf16 bits 8..6
// and the sign at bit 15. The result is fp4_value * 2^-126 (denormal for e == 0), so multiplying by
// block_scale * 2^126 gives exactly the baseline's bf16(fp4_value * block_scale) rounding. Returns
// pairs {nibble j, nibble j + 4} for j = 0..3.
__device__ __forceinline__ void Fp4WordToRawBf16(uint32_t word, uint32_t (&raw)[4]) {
  const uint32_t upper = word >> 8;
  raw[0] = ((word & 0x000F000Fu) * 0x1040u) & 0x81C081C0u;
  raw[1] = ((word & 0x00F000F0u) * 0x0104u) & 0x81C081C0u;
  raw[2] = ((upper & 0x000F000Fu) * 0x1040u) & 0x81C081C0u;
  raw[3] = ((upper & 0x00F000F0u) * 0x0104u) & 0x81C081C0u;
}

static __device__ constexpr uint32_t kBf16Pow2_126 = 0x7E807E80u;  // {2^126, 2^126}

// Activation unit (32 elements = one lane pair's K slice) stride in shared memory. 36 floats = 9 float4
// chunks, so the eight lanes j of a warp read distinct banks without any per-chunk index arithmetic.
static constexpr int kDecodeActUnitStride = 36;

template <typename T>
using PackedPairT = std::conditional_t<std::is_same_v<T, half>, half2, __nv_bfloat162>;

// Accumulates the 32 products of one weight vector per column into the two lane-pair chains of each
// column. `decode(c, w, pairs)` yields the four packed pairs {element j, element j + 4} of 32-bit weight
// word w of column c; each activation chunk is read from shared memory once for all columns.
template <typename T, int Cols, typename DecodeFn>
__device__ __forceinline__ void FmaWeightVector(const float* act, float (&chain_a)[Cols], float (&chain_b)[Cols],
                                                DecodeFn decode) {
#pragma unroll
  for (int w = 0; w < 4; ++w) {
    uint32_t decoded_pairs[Cols][4];
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
      decode(c, w, decoded_pairs[c]);
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      // Elements 8w + 4h .. 8w + 4h + 3 are float4 chunk 2w + h of the unit.
      const float4 a = *reinterpret_cast<const float4*>(act + (2 * w + h) * 4);
      const float activations[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
      for (int c = 0; c < Cols; ++c) {
        float& chain = (w / 2) ? chain_b[c] : chain_a[c];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float decoded = h == 0 ? PackedLowToFloat<T>(decoded_pairs[c][i]) : PackedHighToFloat<T>(decoded_pairs[c][i]);
          chain += decoded * activations[i];
        }
      }
    }
  }
}

// Block scales of the two groups of one vector: scale_rounded = T(e4m3 * global), exactly as the baseline,
// packed {group A, group B} in T.
template <typename T>
__device__ __forceinline__ uint32_t RoundedScalePair(uint32_t scale_code, float global_scale) {
  uint32_t half_pair;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(half_pair) : "h"(static_cast<uint16_t>(scale_code)));
  const float scale_a = __half2float(__ushort_as_half(static_cast<uint16_t>(half_pair & 0xffffu))) * global_scale;
  const float scale_b = __half2float(__ushort_as_half(static_cast<uint16_t>(half_pair >> 16))) * global_scale;
  if constexpr (std::is_same_v<T, half>) {
    const half2 rounded = __floats2half2_rn(scale_a, scale_b);
    return *reinterpret_cast<const uint32_t*>(&rounded);
  } else {
    const __nv_bfloat162 rounded = __floats2bfloat162_rn(scale_a, scale_b);
    return *reinterpret_cast<const uint32_t*>(&rounded);
  }
}

// Operands of one weight vector per column, passed by value to the out-of-line generic decode so the
// caller's register-resident arrays are never address-taken.
template <int Cols>
struct DecodeVectorState {
  uint4 packed[Cols];
  uint32_t scale_codes[Cols];
  float chain_a[Cols];
  float chain_b[Cols];
};

// Generic decode (any T, any block scale): the Fp4I2FConverter path of the baseline kernel.
template <typename T, int Cols>
__device__ __noinline__ DecodeVectorState<Cols> AccumulateWeightVectorsGeneric(
    DecodeVectorState<Cols> state, float global_scale, const float* act) {
  uint32_t scale_rounded[Cols];
#pragma unroll
  for (int c = 0; c < Cols; ++c) {
    scale_rounded[c] = RoundedScalePair<T>(state.scale_codes[c], global_scale);
  }
  FmaWeightVector<T, Cols>(act, state.chain_a, state.chain_b, [&](int c, int w, uint32_t (&decoded_pairs)[4]) {
    uint32_t word_copy = (&state.packed[c].x)[w];
    alignas(4) T decoded[8];
    fiv::Fp4I2FConverter<T>::template convert<8>(&word_copy, decoded);
    const uint32_t scale_splat = __byte_perm(scale_rounded[c], scale_rounded[c], (w / 2) ? 0x3232 : 0x1010);
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      reinterpret_cast<PackedPairT<T>*>(decoded)[p] = __hmul2(reinterpret_cast<const PackedPairT<T>*>(decoded)[p],
                                                              *reinterpret_cast<const PackedPairT<T>*>(&scale_splat));
    }
    const uint16_t* bits = reinterpret_cast<const uint16_t*>(decoded);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      decoded_pairs[j] = static_cast<uint32_t>(bits[j]) | (static_cast<uint32_t>(bits[j + 4]) << 16);
    }
  });
  return state;
}

// bf16 fast path: valid when every block scale is below 4 (global_scale * 448 < 3.98), so that
// block_scale * 2^126 stays finite in bf16.
template <typename T, int Cols>
__device__ __forceinline__ void AccumulateWeightVectors(
    const uint4 (&packed)[Cols], const uint32_t (&scale_codes)[Cols], float global_scale, bool fast_scales,
    const float* act, float (&chain_a)[Cols], float (&chain_b)[Cols]) {
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    if (fast_scales) {
      uint32_t scale_raw[Cols][2];
#pragma unroll
      for (int c = 0; c < Cols; ++c) {
        const uint32_t rounded_bits = RoundedScalePair<T>(scale_codes[c], global_scale);
        const __nv_bfloat162 shifted = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&rounded_bits),
                                               *reinterpret_cast<const __nv_bfloat162*>(&kBf16Pow2_126));
        const uint32_t shifted_bits = *reinterpret_cast<const uint32_t*>(&shifted);
        scale_raw[c][0] = __byte_perm(shifted_bits, shifted_bits, 0x1010);
        scale_raw[c][1] = __byte_perm(shifted_bits, shifted_bits, 0x3232);
      }
      FmaWeightVector<T, Cols>(act, chain_a, chain_b, [&](int c, int w, uint32_t (&decoded_pairs)[4]) {
        uint32_t raw[4];
        Fp4WordToRawBf16((&packed[c].x)[w], raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const __nv_bfloat162 product = __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&raw[j]),
                                                 *reinterpret_cast<const __nv_bfloat162*>(&scale_raw[c][w / 2]));
          decoded_pairs[j] = *reinterpret_cast<const uint32_t*>(&product);
        }
      });
      return;
    }
  }
  DecodeVectorState<Cols> state;
#pragma unroll
  for (int c = 0; c < Cols; ++c) {
    state.packed[c] = packed[c];
    state.scale_codes[c] = scale_codes[c];
    state.chain_a[c] = chain_a[c];
    state.chain_b[c] = chain_b[c];
  }
  state = AccumulateWeightVectorsGeneric<T, Cols>(state, global_scale, act);
#pragma unroll
  for (int c = 0; c < Cols; ++c) {
    chain_a[c] = state.chain_a[c];
    chain_b[c] = state.chain_b[c];
  }
}

template <typename T, bool FusedSwiGlu, bool EnableBias, int Warps, int Cols, int MinBlocks>
__global__ void __launch_bounds__(Warps * 32, MinBlocks) MoeGemvFp4DecodeKernel(
    const T* act, const uint8_t* weight, const uint8_t* block_scales, const float* global_scales,
    const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t weight_expert_stride, int64_t scale_expert_stride, int n, int k,
    cutlass_kernels::ActivationParams activation_params,
    const int* permuted_row_to_source_row, int num_rows) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  constexpr int kTilesInFlight = 3;  // K tiles whose loads are issued together
  constexpr int kThreads = Warps * 32;
  constexpr unsigned kFullMask = 0xffffffffu;
  static_assert(Cols == 1 || Cols == 2, "SwiGLU pairs are either inside a thread or across adjacent slots");

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

  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int j = lane % 8;
  const int slot = lane / 8;
  const int first_column = (static_cast<int>(blockIdx.y) * Warps + warp) * (4 * Cols) + slot * Cols;
  const int source_row = permuted_row_to_source_row ? permuted_row_to_source_row[row] % num_rows : row;
  const int groups = k / 16;
  const int w_blocks = min(4, (groups + 15) / 16);

  const uint8_t* weight_row = weight + static_cast<int64_t>(expert) * weight_expert_stride +
                              static_cast<int64_t>(first_column) * (k / 2) + j * 16;
  const uint8_t* scale_row = block_scales + static_cast<int64_t>(expert) * scale_expert_stride +
                             static_cast<int64_t>(first_column) * groups + j * 2;

  uint4 packed[kTilesInFlight][Cols];
  uint32_t scale_codes[kTilesInFlight][Cols];
  const auto load_block = [&](int w) {
#pragma unroll
    for (int u = 0; u < kTilesInFlight; ++u) {
      // Lane pairs past the end of K load nothing and contribute exact zeros.
      const bool valid = u * 64 + w * 16 + j * 2 < groups;
#pragma unroll
      for (int c = 0; c < Cols; ++c) {
        packed[u][c] = valid ? LoadWeightStreaming(weight_row + static_cast<int64_t>(c) * (k / 2) + w * 128 + u * 512)
                             : make_uint4(0, 0, 0, 0);
        scale_codes[u][c] = valid ? *reinterpret_cast<const uint16_t*>(scale_row + c * groups + w * 16 + u * 64) : 0u;
      }
    }
  };
  load_block(0);

  // Stage the activation row as FP32, one padded 32-element unit per lane pair.
  extern __shared__ __align__(16) float act_smem[];
  {
    const uint4* source = reinterpret_cast<const uint4*>(act + static_cast<int64_t>(source_row) * k);
    for (int index = threadIdx.x; index < k / 8; index += kThreads) {
      const uint4 raw = source[index];
      const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
      float values[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        values[2 * i] = PackedLowToFloat<T>(words[i]);
        values[2 * i + 1] = PackedHighToFloat<T>(words[i]);
      }
      float* destination = act_smem + (index / 4) * kDecodeActUnitStride + (index % 4) * 8;
      *reinterpret_cast<float4*>(destination) = make_float4(values[0], values[1], values[2], values[3]);
      *reinterpret_cast<float4*>(destination + 4) = make_float4(values[4], values[5], values[6], values[7]);
    }
  }
  const float global_scale = global_scales[expert];
  const bool fast_scales = fabsf(global_scale) * 448.0f < 3.98f;
  __syncthreads();

  float total[Cols];
#pragma unroll
  for (int c = 0; c < Cols; ++c) {
    total[c] = 0.0f;
  }
#pragma unroll 1
  for (int w = 0; w < w_blocks; ++w) {
    float chain_a[Cols];
    float chain_b[Cols];
    uint4 current_packed[kTilesInFlight][Cols];
    uint32_t current_codes[kTilesInFlight][Cols];
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
      chain_a[c] = 0.0f;
      chain_b[c] = 0.0f;
    }
#pragma unroll
    for (int u = 0; u < kTilesInFlight; ++u) {
#pragma unroll
      for (int c = 0; c < Cols; ++c) {
        current_packed[u][c] = packed[u][c];
        current_codes[u][c] = scale_codes[u][c];
      }
    }
    if (w + 1 < w_blocks) {
      load_block(w + 1);
    }
#pragma unroll
    for (int u = 0; u < kTilesInFlight; ++u) {
      // Warp-uniform: lane j == 0 has the smallest group of the warp.
      if (u * 64 + w * 16 < groups) {
        const int group = min(u * 64 + w * 16 + j * 2, groups - 2);
        const float* unit_act = act_smem + (group / 2) * kDecodeActUnitStride;
        AccumulateWeightVectors<T, Cols>(current_packed[u], current_codes[u], global_scale, fast_scales, unit_act,
                                         chain_a, chain_b);
      }
    }
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
      float partial = chain_a[c] + chain_b[c];
      partial += __shfl_xor_sync(kFullMask, partial, 1);
      partial += __shfl_xor_sync(kFullMask, partial, 2);
      partial += __shfl_xor_sync(kFullMask, partial, 4);
      total[c] += partial;
    }
  }

  // SwiGLU pairs columns (n, n + 1): inside the thread for Cols == 2, else in adjacent slots.
  float acc0 = total[0];
  float acc1 = total[Cols - 1];
  if constexpr (FusedSwiGlu && Cols == 1) {
    acc1 = __shfl_xor_sync(kFullMask, total[0], 8);
  }
  if (j != 0) {
    return;
  }
  if constexpr (FusedSwiGlu) {
    if (Cols == 1 && (slot & 1)) {
      return;
    }
    if constexpr (EnableBias) {
      const T* expert_bias = bias + static_cast<int64_t>(expert) * n;
      acc0 += static_cast<float>(expert_bias[first_column]);
      acc1 += static_cast<float>(expert_bias[first_column + 1]);
    }
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
    out[static_cast<int64_t>(row) * (n / 2) + first_column / 2] = static_cast<T>(acc0 * sigmoid * acc1);
  } else {
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
      float value = total[c];
      if constexpr (EnableBias) {
        value += static_cast<float>(bias[static_cast<int64_t>(expert) * n + first_column + c]);
      }
      out[static_cast<int64_t>(row) * n + first_column + c] = static_cast<T>(value);
    }
  }
#endif
}

template <typename T, bool FusedSwiGlu>
void LaunchMoeGemvFp4RawNPacked(
    const T* act, const uint8_t* weight, const uint8_t* block_scales, const float* global_scales,
    const T* bias, T* out,
    const int64_t* expert_first_token_offset, const int* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t n, int64_t k, cutlass_kernels::ActivationParams activation_params,
    const int* permuted_row_to_source_row, int64_t num_rows, cudaStream_t stream, bool weights_row_major) {
  constexpr int kThreads = 128;
  constexpr int kColsPerBlock = 16;
  const int64_t weight_expert_stride = n * k / 2;
  const int64_t scale_expert_stride = n * (k / 16);
  if (weights_row_major && Fp4MoeGemvUseDecodeKernel() && k % 32 == 0 && k <= kDecodeMaxK &&
      reinterpret_cast<uintptr_t>(weight) % 16 == 0 && reinterpret_cast<uintptr_t>(block_scales) % 2 == 0 &&
      reinterpret_cast<uintptr_t>(act) % 16 == 0) {
    const size_t decode_smem = static_cast<size_t>(k / 32) * kDecodeActUnitStride * sizeof(float);
    // Columns per CTA = 4 slots x Cols x Warps.
    const auto launch_decode = [&](auto warps_tag, auto cols_tag, auto min_blocks_tag, auto bias_tag) {
      constexpr int kWarps = decltype(warps_tag)::value;
      constexpr int kCols = decltype(cols_tag)::value;
      constexpr int kMinBlocks = decltype(min_blocks_tag)::value;
      constexpr bool kEnableBias = decltype(bias_tag)::value;
      constexpr int kBlockCols = 4 * kWarps * kCols;
      const dim3 decode_grid(static_cast<unsigned>(expanded_num_rows), static_cast<unsigned>(n / kBlockCols));
      MoeGemvFp4DecodeKernel<T, FusedSwiGlu, kEnableBias, kWarps, kCols, kMinBlocks>
          <<<decode_grid, kWarps * 32, decode_smem, stream>>>(
              act, weight, block_scales, global_scales, bias, out,
              expert_first_token_offset, permuted_row_to_expert, num_experts,
              weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k), activation_params,
              permuted_row_to_source_row, static_cast<int>(num_rows));
    };
    const auto dispatch = [&](auto bias_tag) {
      if constexpr (FusedSwiGlu) {
        if (n % (4 * GEMV_FC1_WARPS * GEMV_FC1_COLS) == 0) {
          launch_decode(std::integral_constant<int, GEMV_FC1_WARPS>{}, std::integral_constant<int, GEMV_FC1_COLS>{},
                        std::integral_constant<int, GEMV_FC1_MINB>{}, bias_tag);
          return true;
        }
      } else {
        if (n % (4 * GEMV_FC2_WARPS * GEMV_FC2_COLS) == 0) {
          launch_decode(std::integral_constant<int, GEMV_FC2_WARPS>{}, std::integral_constant<int, GEMV_FC2_COLS>{},
                        std::integral_constant<int, GEMV_FC2_MINB>{}, bias_tag);
          return true;
        }
      }
      return false;
    };
    const bool launched = bias != nullptr ? dispatch(std::true_type{}) : dispatch(std::false_type{});
    if (launched) {
      return;
    }
  }
  const dim3 grid(static_cast<unsigned>(expanded_num_rows), static_cast<unsigned>((n + kColsPerBlock - 1) / kColsPerBlock));
  const auto launch = [&](auto row_major_tag) {
    constexpr bool kWeightsRowMajor = decltype(row_major_tag)::value;
    if (bias != nullptr) {
      MoeGemvFp4RawNPackedKernel<T, FusedSwiGlu, true, kWeightsRowMajor><<<grid, kThreads, 0, stream>>>(
        act, weight, block_scales, global_scales, bias, out,
        expert_first_token_offset, permuted_row_to_expert, num_experts,
        weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k), activation_params,
        permuted_row_to_source_row, static_cast<int>(num_rows));
    } else {
      MoeGemvFp4RawNPackedKernel<T, FusedSwiGlu, false, kWeightsRowMajor><<<grid, kThreads, 0, stream>>>(
        act, weight, block_scales, global_scales, bias, out,
        expert_first_token_offset, permuted_row_to_expert, num_experts,
        weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k), activation_params,
        permuted_row_to_source_row, static_cast<int>(num_rows));
    }
  };
  if (weights_row_major) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
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
                                   cudaStream_t stream, bool weights_row_major) {
  ORT_UNUSED_PARAMETER(sm);
  if (raw_n_packed) {
    ORT_ENFORCE(group_size == 16, "Raw N-packed FP4 GEMV is NVFP4-only.");
    ORT_ENFORCE(raw_block_scales != nullptr && raw_global_scales != nullptr,
                "Raw N-packed NVFP4 GEMV requires block and global scales.");
    LaunchMoeGemvFp4RawNPacked<T, false>(
        act, weight, raw_block_scales, raw_global_scales, bias, out,
        expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, n, k, cutlass_kernels::ActivationParams{}, nullptr, expanded_num_rows, stream,
        weights_row_major);
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
    const int* permuted_row_to_source_row, int64_t num_rows, cudaStream_t stream, bool weights_row_major) {
  ORT_UNUSED_PARAMETER(sm);
  if (raw_n_packed) {
    ORT_ENFORCE(group_size == 16, "Raw N-packed FP4 GEMV is NVFP4-only.");
    ORT_ENFORCE(raw_block_scales != nullptr && raw_global_scales != nullptr,
                "Raw N-packed NVFP4 GEMV requires block and global scales.");
    LaunchMoeGemvFp4RawNPacked<T, true>(
        act, weight, raw_block_scales, raw_global_scales, bias, out,
        expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size * 2, k, activation_params, permuted_row_to_source_row, num_rows, stream,
        weights_row_major);
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
    int64_t, int64_t, int64_t, int, int, MoeGemvConfig, bool, bool, cudaStream_t, bool);
template void launch_moe_gemv_fp4_symmetric_interleaved_swiglu<half>(
    const half*, const uint8_t*, const half*, const uint8_t*, const float*, const half*, half*,
    const int64_t*, const int*, int,
    int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams, MoeGemvConfig, bool, bool,
    const int*, int64_t, cudaStream_t, bool);
#ifdef ENABLE_BF16
template void launch_moe_gemv_fp4_symmetric<__nv_bfloat16>(
    const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const uint8_t*, const float*,
    const __nv_bfloat16*, __nv_bfloat16*,
    const int64_t*, const int*, int, int64_t, int64_t, int64_t, int, int, MoeGemvConfig, bool, bool, cudaStream_t, bool);
template void launch_moe_gemv_fp4_symmetric_interleaved_swiglu<__nv_bfloat16>(
    const __nv_bfloat16*, const uint8_t*, const __nv_bfloat16*, const uint8_t*, const float*,
    const __nv_bfloat16*, __nv_bfloat16*,
    const int64_t*, const int*, int, int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams,
    MoeGemvConfig, bool, bool, const int*, int64_t, cudaStream_t, bool);
#endif

}  // namespace moe_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
