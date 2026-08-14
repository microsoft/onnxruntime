// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <map>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/quantization/matmul_4bits_common.cuh"
#include "contrib_ops/cuda/quantization/matmul_4bits_m1.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// VALIDATION-ONLY (never merged into #31988 or main): lets a throwaway A/B benchmark force
// the pre-PR baseline launch geometry (cols_per_block == kColsPerThreadBlock unconditionally)
// so head-vs-base timing can be compared on the same binary/process/GPU without a second
// build or a second CI run -- eliminating build-to-build and machine-state (thermal, clock,
// driver cache) confounds from the comparison. Gated behind an environment variable (read
// once, lazily, and cached) rather than a settable global so it can be toggled from a
// separate benchmark *process* (e.g. onnxruntime_perf_test.exe) without needing to link
// against or mutate symbols inside this translation unit. Unset/0 (the default) means
// "no override": behavior is byte-for-byte identical to #31988's head.
inline int MatMulNBitsM1PerfValidationForceColsPerBlock() {
  static const int forced = [] {
    const char* env = std::getenv("ORT_MATMULNBITS_M1_PERF_VALIDATION_FORCE_COLS_PER_BLOCK");
    if (env == nullptr || env[0] == '\0') {
      return 0;
    }
    return std::atoi(env);
  }();
  return forced;
}

// ---------------------------------------------------------------------------------------
// VALIDATION-ONLY (never merged into #31988 or main): in-process, kernel-level A/B
// benchmark instrumentation.
//
// The session-level benchmark (onnxruntime_perf_test.exe wall-clock latency, see PR #31988's
// Performance section / validation PR #32073/#32073-successor) could not separate the M=1
// kernel's own execution time from ORT Session/graph/IOBinding overhead, and the resulting
// A/B signal was indistinguishable from A/A run-to-run noise (geomean 1.006 in both groups).
//
// This block goes one level deeper: it times *only* the kernel launch (cudaEvent_t pair
// recorded immediately around the dispatch macro below, elapsed time read back with
// cudaEventElapsedTime) and alternates which cols_per_block value is used on every call via a
// fixed-seed PRNG, so a single onnxruntime_perf_test.exe process -- driven by repeated
// Session::Run() calls against a fixed-shape MatMulNBits model -- interleaves the two arms
// without a second process or a second build. Reporting happens by printing a summary line
// directly to stderr from *inside* this same translation unit/module once enough samples are
// collected; no cross-module (EXE-vs-provider-DLL) call is required in either direction,
// which matters because onnxruntime_providers_cuda is loaded as a runtime plugin and is not a
// build-time link dependency of any CI-executed test executable.
//
// Entirely gated behind ORT_MATMULNBITS_M1_KERNEL_AB_BENCH=1 (unset by default): with the
// switch off, every function below is a cheap no-op/short-circuit and #31988's reviewed head
// behavior (including performance) is unchanged.
struct MatMulNBitsM1KernelAbBenchSample {
  int cols_used;
  float elapsed_ms;
};

inline int MatMulNBitsM1KernelAbBenchEnvInt(const char* name, int default_value) {
  const char* env = std::getenv(name);
  if (env == nullptr || env[0] == '\0') {
    return default_value;
  }
  return std::atoi(env);
}

inline bool MatMulNBitsM1KernelAbBenchEnabled() {
  const char* env = std::getenv("ORT_MATMULNBITS_M1_KERNEL_AB_BENCH");
  return env != nullptr && env[0] == '1';
}

// Picks this call's cols_per_block (arm A or arm B, chosen by a fixed-seed coin flip so the
// interleave order is reproducible across runs) and returns true (the call should be timed).
// Kept simple/always-true rather than tracking a shared "done" flag with the per-dtype
// MatMulNBitsM1KernelAbBenchRecordSample<T> below (which independently stops accumulating and
// prints once, per dtype, after enough samples) -- calls after that point still get a valid
// (if unused) arm assignment, so this never produces an invalid geometry.
inline bool MatMulNBitsM1KernelAbBenchPrepareCall(int n, int natural_cols_per_block,
                                                   int* cols_per_block_in_out) {
  static bool header_printed = false;

  const int arm_a_env = MatMulNBitsM1KernelAbBenchEnvInt("ORT_MATMULNBITS_M1_KERNEL_AB_ARM_A", 0);
  const int arm_b_env = MatMulNBitsM1KernelAbBenchEnvInt("ORT_MATMULNBITS_M1_KERNEL_AB_ARM_B", 0);
  const int arm_a = (arm_a_env != 0 && n % arm_a_env == 0) ? arm_a_env : natural_cols_per_block;
  const int arm_b = (arm_b_env != 0 && n % arm_b_env == 0) ? arm_b_env : natural_cols_per_block;

  if (!header_printed) {
    header_printed = true;
    fprintf(stderr,
            "MATMULNBITS_M1_KERNEL_AB_HEADER n=%d arm_a_requested=%d arm_a_effective=%d "
            "arm_b_requested=%d arm_b_effective=%d target_samples=%d\n",
            n, arm_a_env, arm_a, arm_b_env, arm_b,
            MatMulNBitsM1KernelAbBenchEnvInt("ORT_MATMULNBITS_M1_KERNEL_AB_TARGET_SAMPLES", 2000));
  }

  // Fixed seed: reproducible interleave order across repeated invocations/CI runs.
  static std::mt19937 rng(0xC0FFEEu);
  const bool pick_b = (rng() & 1u) != 0;
  *cols_per_block_in_out = pick_b ? arm_b : arm_a;
  return true;
}

// Non-template: same definition folded (COMDAT) across the float/half/bfloat16 translation
// units, so the lazily-created events (and MatMulNBitsM1KernelAbBenchPrepareCall's RNG/done
// state above) are genuinely shared process-wide rather than duplicated per dtype.
inline void MatMulNBitsM1KernelAbBenchGetEvents(cudaEvent_t* start, cudaEvent_t* stop) {
  static cudaEvent_t s_start = [] {
    cudaEvent_t e{};
    cudaEventCreate(&e);
    return e;
  }();
  static cudaEvent_t s_stop = [] {
    cudaEvent_t e{};
    cudaEventCreate(&e);
    return e;
  }();
  *start = s_start;
  *stop = s_stop;
}

inline float MatMulNBitsM1KernelAbBenchPercentile(std::vector<float> v, double p) {
  if (v.empty()) {
    return 0.0f;
  }
  std::sort(v.begin(), v.end());
  const size_t idx = static_cast<size_t>(p * static_cast<double>(v.size() - 1));
  return v[idx];
}

// Per-dtype (T-templated, deliberately -- each dtype gets its own independent sample buffer)
// accumulator/printer. Called once per timed kernel launch; synchronizes on `stop` (required
// for cudaEventElapsedTime to be valid) and, once ORT_MATMULNBITS_M1_KERNEL_AB_TARGET_SAMPLES
// timed calls have been collected for this dtype, prints one summary line per distinct
// cols_per_block bucket observed (almost always exactly the two arms) to stderr and stops
// collecting further samples.
template <class T>
inline void MatMulNBitsM1KernelAbBenchRecordSample(int n, int k, int block_size, int cols_used,
                                                    cudaEvent_t start, cudaEvent_t stop, int sm_count) {
  static std::vector<MatMulNBitsM1KernelAbBenchSample> samples;
  static bool printed = false;
  static int clock_khz_start = -1;
  if (printed) {
    return;
  }

  cudaEventSynchronize(stop);
  float elapsed_ms = 0.0f;
  cudaEventElapsedTime(&elapsed_ms, start, stop);

  if (clock_khz_start < 0) {
    cudaDeviceGetAttribute(&clock_khz_start, cudaDevAttrClockRate, 0);
  }

  samples.push_back({cols_used, elapsed_ms});

  const int target = MatMulNBitsM1KernelAbBenchEnvInt("ORT_MATMULNBITS_M1_KERNEL_AB_TARGET_SAMPLES", 2000);
  if (static_cast<int>(samples.size()) < target) {
    return;
  }
  printed = true;

  int clock_khz_end = -1;
  cudaDeviceGetAttribute(&clock_khz_end, cudaDevAttrClockRate, 0);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);

  // Discard the first samples (regardless of arm) as warmup.
  constexpr size_t kWarmup = 50;
  std::map<int, std::vector<float>> buckets_us;
  for (size_t i = kWarmup; i < samples.size(); ++i) {
    buckets_us[samples[i].cols_used].push_back(samples[i].elapsed_ms * 1000.0f);
  }

  for (auto& kv : buckets_us) {
    const int bucket_cols = kv.first;
    std::vector<float>& us = kv.second;
    float mean = 0.0f;
    for (float v : us) mean += v;
    mean /= static_cast<float>(us.size());
    const float p50 = MatMulNBitsM1KernelAbBenchPercentile(us, 0.50);
    const float p90 = MatMulNBitsM1KernelAbBenchPercentile(us, 0.90);
    const float p95 = MatMulNBitsM1KernelAbBenchPercentile(us, 0.95);
    const float vmin = *std::min_element(us.begin(), us.end());
    const float vmax = *std::max_element(us.begin(), us.end());
    fprintf(stderr,
            "MATMULNBITS_M1_KERNEL_AB_RESULT n=%d k=%d block_size=%d sm_count=%d gpu=\"%s\" "
            "cc=%d.%d clock_khz_start=%d clock_khz_end=%d bucket_cols=%d count=%zu "
            "mean_us=%.3f p50_us=%.3f p90_us=%.3f p95_us=%.3f min_us=%.3f max_us=%.3f\n",
            n, k, block_size, sm_count, prop.name, prop.major, prop.minor, clock_khz_start,
            clock_khz_end, bucket_cols, us.size(), mean, p50, p90, p95, vmin, vmax);
  }
}
// ---------------------------------------------------------------------------------------

template <class T, int block_size, bool has_zero_point, int cols_per_block = kColsPerThreadBlock>
__global__ void __launch_bounds__(kWarpSize* cols_per_block) MatMulFloat4BitsKernelM1(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int blocks_per_K) {
  const int n_block_id = blockIdx.x;
  const int m_id = blockIdx.y;
  const int lane_id = threadIdx.x;
  const int warp_id = WarpUniform(threadIdx.y);
  const int n_id = n_block_id * cols_per_block + warp_id;
  constexpr int k_per_iter = kWarpSize * kElementsPerThreadPerIteration;

  extern __shared__ char shared_buffer[];
  T* b_scale_vec = (T*)shared_buffer;
  int offset = n_block_id * cols_per_block * blocks_per_K;
  for (int i = warp_id * kWarpSize + lane_id; i < cols_per_block * blocks_per_K;
       i += cols_per_block * kWarpSize) {
    b_scale_vec[i] = scales_data[offset + i];
  }

  uint8_t* b_zp_vec;
  (void)b_zp_vec;
  if constexpr (has_zero_point) {
    b_zp_vec = reinterpret_cast<uint8_t*>(b_scale_vec + cols_per_block * blocks_per_K);
    const int b_zp_k = (blocks_per_K + 1) / 2;
    int zp_offset = n_block_id * cols_per_block * b_zp_k;
    for (int i = warp_id * kWarpSize + lane_id; i < cols_per_block * b_zp_k;
         i += cols_per_block * kWarpSize) {
      b_zp_vec[2 * i] = (zero_points[zp_offset + i] & 0x0f);
      b_zp_vec[2 * i + 1] = (zero_points[zp_offset + i] >> 4);
    }
    b_zp_vec += warp_id * b_zp_k * 2;
  }
  __syncthreads();

  a_data += m_id * k + (lane_id << 3);
  b_scale_vec += warp_id * blocks_per_K;

  T sums[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  int k_id = 0;
  int t_meta_k = lane_id * 8 / block_size;
  b_data_quant += n_id * blocks_per_K * (block_size / 2) + lane_id * 4;

#define UnRollReduction(unroll_size)                                                              \
  do {                                                                                            \
    constexpr int kUnroll = unroll_size;                                                          \
    constexpr int kUnrollStep = kUnroll * k_per_iter;                                             \
    const int k_unroll_bound = k - k % kUnrollStep;                                               \
    for (; k_id < k_unroll_bound; k_id += kUnrollStep) {                                          \
      _Pragma("unroll") for (int i = 0; i < kUnroll; i++) {                                       \
        uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant + k_per_iter / 2 * i)); \
        T scale = b_scale_vec[t_meta_k + k_per_iter / block_size * i];                            \
        uint8_t zp = 8;                                                                           \
        if constexpr (has_zero_point) {                                                           \
          zp = b_zp_vec[t_meta_k + k_per_iter / block_size * i];                                  \
        }                                                                                         \
        AccumulateEightElements4b(value, scale, zp, a_data + k_id + i * k_per_iter, sums);        \
      }                                                                                           \
      b_data_quant += k_per_iter / 2 * kUnroll;                                                   \
      t_meta_k += k_per_iter / block_size * kUnroll;                                              \
    }                                                                                             \
  } while (false)

  UnRollReduction(16);
  UnRollReduction(4);
  UnRollReduction(1);
#undef UnRollReduction

  if (k_id + lane_id * 8 < k) {
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant));
    T scale = b_scale_vec[t_meta_k];
    uint8_t zp = 8;
    if constexpr (has_zero_point) {
      zp = b_zp_vec[t_meta_k];
    }
    AccumulateEightElements4b(value, scale, zp, a_data + k_id, sums);
  }

  float sum = (float)(sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7]);
  for (int i = kWarpSize / 2; i > 0; i = i / 2) {
    sum += onnxruntime::cuda::WARP_SHFL_DOWN(sum, i);
  }

  if (lane_id == 0) {
    output[m_id * n + n_id] = sum;
  }
}

template <class T>
bool TryMatMul4BitsM1(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    int sm_count,
    cudaStream_t stream) {
  // Admission: always gate on the *original* cols=8 shared-memory footprint so that the
  // accepted-shape set is bit-for-bit identical to upstream. Shapes with huge K that would
  // have been declined (shared mem too large with 8 cols/CTA) must NOT be silently admitted
  // just because a smaller cols_per_block would fit. Those shapes fall through to cuBLAS
  // with fp32 accumulation — admitting them here would be a silent accuracy regression.
  if (n % kColsPerThreadBlock != 0 || k % kElementsPerThreadPerIteration != 0) {
    return false;
  }

  const int blocks_per_K = (k + block_size - 1) / block_size;
  const size_t admission_shared_mem =
      sizeof(T) * blocks_per_K * kColsPerThreadBlock +
      static_cast<size_t>(zero_points != nullptr ? (blocks_per_K + 1) / 2 * kColsPerThreadBlock * 2 : 0);
  if (admission_shared_mem > shared_mem_per_block) {
    return false;
  }

  // Launch: select cols_per_block using the CUDA occupancy API rather than a magic constant.
  // For each candidate (8, 4, 2) we query cudaOccupancyMaxActiveBlocksPerMultiprocessor for
  // the actual kernel instantiation (register and shared-memory footprints differ per
  // cols_per_block) and hand the results to ChooseColsPerBlockFromOccupancy, which applies
  // the lexicographic objective (SMs busy, then resident warps, then keep the largest
  // candidate). The decision function is deliberately a pure host function in
  // matmul_4bits_cols_per_block.h so it can be unit-tested without a GPU.
  //
  // This replaces the old kTargetCtasPerSm = 12 magic number, which was tuned for
  // datacenter parts (A100/H100) and regressed CC 8.6/8.9 consumer GPUs where per-SM
  // block limits differ.
  //
  // NOTE: This path cannot be validated without a GPU. Consumer-GPU measurements
  // (CC 8.6 RTX 3060, CC 8.9 RTX 4090) are required before this PR can leave draft.

  // We need the kernel pointer to query occupancy. To avoid combinatorial explosion of
  // block_size × has_zp × cols queries, we pick a representative kernel (block_size=32,
  // no zero points) — the register/shared-memory profile is dominated by cols_per_block,
  // not by block_size or the zero-point branch.
  auto query_max_active_blocks = [&](int cpb, size_t smem) -> int {
    int max_blocks = 0;
    const int threads_per_block = onnxruntime::cuda::GPU_WARP_SIZE_HOST * cpb;
    cudaError_t err = cudaSuccess;
    // Use block_size=32 as a representative instantiation for occupancy query.
    // The register pressure difference between block_size variants is negligible
    // (same unrolled loop body, only the iteration bound changes).
    switch (cpb) {
      case 8:
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks, MatMulFloat4BitsKernelM1<T, 32, false, 8>, threads_per_block, smem);
        break;
      case 4:
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks, MatMulFloat4BitsKernelM1<T, 32, false, 4>, threads_per_block, smem);
        break;
      case 2:
        err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks, MatMulFloat4BitsKernelM1<T, 32, false, 2>, threads_per_block, smem);
        break;
      default:
        break;
    }
    return (err == cudaSuccess) ? max_blocks : 0;
  };

  int max_blocks_per_sm[kNumColsPerBlockCandidates] = {0, 0, 0};
  for (int i = 0; i < kNumColsPerBlockCandidates; ++i) {
    const int cpb = kColsPerBlockCandidates[i];
    if (n % cpb != 0) {
      continue;
    }
    const size_t smem = sizeof(T) * blocks_per_K * cpb +
                        static_cast<size_t>(zero_points != nullptr ? (blocks_per_K + 1) / 2 * cpb * 2 : 0);
    max_blocks_per_sm[i] = query_max_active_blocks(cpb, smem);
  }

  // ChooseColsPerBlockFromOccupancy itself falls back to SelectColsPerBlock(n, sm_count) when
  // every query failed or sm_count is unavailable, and only ever returns a divisor of n.
  int cols_per_block = ChooseColsPerBlockFromOccupancy(n, sm_count, max_blocks_per_sm);
  // VALIDATION-ONLY: see MatMulNBitsM1PerfValidationForceColsPerBlock() above. No-op (0)
  // outside the throwaway perf-validation benchmark.
  const int forced_cols_per_block = MatMulNBitsM1PerfValidationForceColsPerBlock();
  if (forced_cols_per_block != 0) {
    cols_per_block = forced_cols_per_block;
  }

  // VALIDATION-ONLY: see MatMulNBitsM1KernelAbBenchPrepareCall() above. No-op outside the
  // throwaway kernel-level A/B benchmark; overrides cols_per_block for *this call only* with
  // a pseudo-randomly chosen arm so a single process can interleave A/B at the kernel level.
  const bool mm_ab_bench_timed_this_call =
      MatMulNBitsM1KernelAbBenchEnabled() &&
      MatMulNBitsM1KernelAbBenchPrepareCall(n, cols_per_block, &cols_per_block);

  const size_t launch_shared_mem =
      sizeof(T) * blocks_per_K * cols_per_block +
      static_cast<size_t>(zero_points != nullptr ? (blocks_per_K + 1) / 2 * cols_per_block * 2 : 0);

  dim3 blocks((n + cols_per_block - 1) / cols_per_block, 1);
  dim3 threads(onnxruntime::cuda::GPU_WARP_SIZE_HOST, cols_per_block);

  // Template instantiation cost: the kernel is templated on <T, block_size, has_zero_point, cols_per_block>.
  // Adding cols_per_block ∈ {8, 4, 2} triples the per-type instantiation count:
  //   Before: 4 block_sizes × 2 zp variants × 1 cols = 8 per type, 24 total (3 types).
  //   After:  4 block_sizes × 2 zp variants × 3 cols = 24 per type, 72 total.
  // Practical cost: each instantiation is ~0.8 KB PTX (simple warp-reduction kernel with no
  // unrolled loops beyond the 8-element vectorized load). Incremental binary growth is bounded
  // at ~38 KB across all three type CUs — modest relative to ORT's 40+ MB provider .so.
  // Compile-time impact is similarly bounded: this kernel is a leaf template in three separate
  // .cu files (float, half, bf16) that already compile independently.
  //
  // All three cols_per_block values are reachable in production. Under
  // ChooseColsPerBlockFromOccupancy the selector is driven by how many SMs the grid covers:
  //   - 8: n/8 >= sm_count, i.e. the default grid already busies every SM (n >= 576 on a
  //        72-SM A10, n >= 1056 on a 132-SM H100) — the common decode case for wide n.
  //   - 4: n/8 < sm_count <= n/4 and n % 4 == 0 (e.g. n = 384 on a 72-SM A10).
  //   - 2: n/4 < sm_count and n is even (e.g. n = 256 on a 72-SM A10: 32 CTAs at cols=8
  //        would leave 40 SMs idle, 128 CTAs at cols=2 busy all 72).
  // The host-only fallback SelectColsPerBlock, used only when every occupancy query fails,
  // reaches the same three values through its own 8-waves-per-SM target.
  // No instantiation is dead code.

#define MATMUL_FLOAT4B_M1_DISPATCH_COLS(bs, cpb)                                                 \
  if (zero_points != nullptr) {                                                                  \
    MatMulFloat4BitsKernelM1<T, bs, true, cpb><<<blocks, threads, launch_shared_mem, stream>>>(  \
        output, a_data, b_data_quant, scales_data, zero_points, 1, n, k, blocks_per_K);          \
  } else {                                                                                       \
    MatMulFloat4BitsKernelM1<T, bs, false, cpb><<<blocks, threads, launch_shared_mem, stream>>>( \
        output, a_data, b_data_quant, scales_data, nullptr, 1, n, k, blocks_per_K);              \
  }

#define MATMUL_FLOAT4B_M1_DISPATCH(bs)                                         \
  if (cols_per_block == 8) {                                                   \
    MATMUL_FLOAT4B_M1_DISPATCH_COLS(bs, 8)                                     \
  } else if (cols_per_block == 4) {                                            \
    MATMUL_FLOAT4B_M1_DISPATCH_COLS(bs, 4)                                     \
  } else if (cols_per_block == 2) {                                            \
    MATMUL_FLOAT4B_M1_DISPATCH_COLS(bs, 2)                                     \
  } else {                                                                     \
    ORT_THROW("cols_per_block ", cols_per_block, " is not supported (8/4/2)"); \
  }

  cudaEvent_t mm_ab_bench_start = nullptr, mm_ab_bench_stop = nullptr;
  if (mm_ab_bench_timed_this_call) {
    MatMulNBitsM1KernelAbBenchGetEvents(&mm_ab_bench_start, &mm_ab_bench_stop);
    cudaEventRecord(mm_ab_bench_start, stream);
  }

  if (block_size == 16) {
    MATMUL_FLOAT4B_M1_DISPATCH(16)
  } else if (block_size == 32) {
    MATMUL_FLOAT4B_M1_DISPATCH(32)
  } else if (block_size == 64) {
    MATMUL_FLOAT4B_M1_DISPATCH(64)
  } else if (block_size == 128) {
    MATMUL_FLOAT4B_M1_DISPATCH(128)
  } else {
    ORT_THROW("block size ", block_size, " is not supported");
  }
#undef MATMUL_FLOAT4B_M1_DISPATCH
#undef MATMUL_FLOAT4B_M1_DISPATCH_COLS

  if (mm_ab_bench_timed_this_call) {
    cudaEventRecord(mm_ab_bench_stop, stream);
    MatMulNBitsM1KernelAbBenchRecordSample<T>(n, k, block_size, cols_per_block, mm_ab_bench_start,
                                               mm_ab_bench_stop, sm_count);
  }

  return true;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
