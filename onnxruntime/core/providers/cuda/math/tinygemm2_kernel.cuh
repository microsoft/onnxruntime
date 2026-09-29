/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Adapted from TensorRT-LLM cpp/tensorrt_llm/kernels/tinygemm2. Changes: B is read in the ONNX MatMul
// [K, N] layout (transposed ldmatrix on a 32B-swizzled tile) instead of [N, K], fp16 support, no bias,
// raw mbarriers instead of cuda::barrier, and a named barrier for the compute-warp reduction.
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace onnxruntime {
namespace cuda {
namespace tinygemm2 {

// A CTA computes kTileN output columns x kTileM rows of C[M, N] = A[M, K] * B[K, N]. The output
// columns are the MMA "M" dimension (A operand, from B) and the rows are the MMA "N" dimension.
constexpr int kTileN = 16;
constexpr int kTileM = 8;
constexpr int kTileK = 64;
constexpr int kStages = 16;
constexpr int kStageUnroll = 4;
// Warps 0-3 compute, 4-7 load B, 8-11 load A.
constexpr int kThreads = 384;
constexpr int kComputeThreads = 128;
constexpr int kWeightTileBytes = kTileK * kTileN * 2;
constexpr int kActivationTileBytes = kTileM * kTileK * 2;
constexpr int kDynamicSmemBytes = kStages * kStageUnroll * (kWeightTileBytes + kActivationTileBytes);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

__device__ __forceinline__ uint32_t SmemAddr(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void BarrierInit(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}

__device__ __forceinline__ void BarrierWait(uint32_t bar, int phase) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra.uni DONE;\n"
      "bra.uni LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(bar),
      "r"(phase));
}

__device__ __forceinline__ bool BarrierTryWait(uint32_t bar, int phase) {
  uint32_t success;
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
      "selp.b32 %0, 1, 0, P1;\n"
      "}"
      : "=r"(success)
      : "r"(bar), "r"(phase));
  return success != 0;
}

__device__ __forceinline__ bool ElectOneSync() {
  uint32_t pred = 0;
  uint32_t lane = 0;
  asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "elect.sync %%rx|%%px, %2;\n"
      "@%%px mov.s32 %1, 1;\n"
      "mov.s32 %0, %%rx;\n"
      "}\n"
      : "+r"(lane), "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred != 0;
}

__device__ __forceinline__ void LoadTile2d(uint32_t smem, const CUtensorMap* map, uint32_t bar, int x, int y) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(
          smem),
      "l"(reinterpret_cast<uint64_t>(map)), "r"(bar), "r"(x), "r"(y)
      : "memory");
}

template <typename T>
__device__ __forceinline__ void Mma16816(float d[4], const uint32_t a[4], const uint32_t b[2]);

template <>
__device__ __forceinline__ void Mma16816<__nv_bfloat16>(float d[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <>
__device__ __forceinline__ void Mma16816<__half>(float d[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <typename T>
__device__ __forceinline__ T FromFloat(float v);
template <>
__device__ __forceinline__ __nv_bfloat16 FromFloat<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <>
__device__ __forceinline__ __half FromFloat<__half>(float v) { return __float2half(v); }

#endif  // __CUDA_ARCH__ >= 900

// B tiles are [kTileK rows of K][kTileN columns] (32-byte rows, 32B swizzle); A tiles are
// [kTileM rows][kTileK] (128-byte rows, 128B swizzle). Both swizzles XOR address bits of the absolute
// shared-memory address, which the ldmatrix addressing below reproduces.
template <typename T>
__global__ void __launch_bounds__(kThreads, 1)
    TinyGemm2Kernel(T* __restrict__ output, int m, int n, int k, const __grid_constant__ CUtensorMap weight_map,
                    const __grid_constant__ CUtensorMap activation_map) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  extern __shared__ __align__(128) char smem[];
  char* sh_weights = smem;
  char* sh_activations = smem + kStages * kStageUnroll * kWeightTileBytes;

  __shared__ __align__(8) uint64_t bar_wt_ready[kStages];
  __shared__ __align__(8) uint64_t bar_act_ready[kStages];
  __shared__ __align__(8) uint64_t bar_consumed[kStages];
  __shared__ float4 reduction_buffer[kComputeThreads];

  if (threadIdx.x == 0) {
    for (int i = 0; i < kStages; ++i) {
      BarrierInit(SmemAddr(&bar_wt_ready[i]), 1);
      BarrierInit(SmemAddr(&bar_act_ready[i]), 1);
      BarrierInit(SmemAddr(&bar_consumed[i]), 32);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&weight_map)) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&activation_map)) : "memory");
  }
  __syncthreads();

  const int warp_id = static_cast<int>(threadIdx.x) / 32;
  const int lane_id = static_cast<int>(threadIdx.x) % 32;
  const int col_base = static_cast<int>(blockIdx.x) * kTileN;
  const int row_base = static_cast<int>(blockIdx.y) * kTileM;
  // Four compute warps each consume every fourth stage group.
  const int k_loops = (k + 4 * kTileK * kStageUnroll - 1) / (4 * kTileK * kStageUnroll);

  if (warp_id >= 4) {
    if (!ElectOneSync()) {
      return;
    }
    const bool weight_warp = warp_id < 8;
    // B is a constant weight, so only the activation load waits for the preceding kernel.
    if (!weight_warp) {
      cudaGridDependencySynchronize();
      cudaTriggerProgrammaticLaunchCompletion();
    }
    int stage = warp_id % 4;
    int phase = 0;
    for (int ki = 0; ki < k_loops; ++ki) {
      const int k0 = (ki * 4 + warp_id % 4) * kTileK * kStageUnroll;
      BarrierWait(SmemAddr(&bar_consumed[stage]), phase ^ 1);
      if (weight_warp) {
        const uint32_t bar = SmemAddr(&bar_wt_ready[stage]);
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(bar),
                     "r"(kStageUnroll * kWeightTileBytes));
        for (int i = 0; i < kStageUnroll; ++i) {
          const uint32_t dst =
              SmemAddr(sh_weights + (stage * kStageUnroll + i) * kWeightTileBytes);
          LoadTile2d(dst, &weight_map, bar, col_base, k0 + i * kTileK);
        }
      } else {
        const uint32_t bar = SmemAddr(&bar_act_ready[stage]);
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(bar),
                     "r"(kStageUnroll * kActivationTileBytes));
        for (int i = 0; i < kStageUnroll; ++i) {
          const uint32_t dst =
              SmemAddr(sh_activations + (stage * kStageUnroll + i) * kActivationTileBytes);
          LoadTile2d(dst, &activation_map, bar, k0 + i * kTileK, row_base);
        }
      }
      stage += 4;
      if (stage >= kStages) {
        stage = warp_id % 4;
        phase ^= 1;
      }
    }
    // Loads still in flight target this CTA's shared memory, so stay until they are consumed.
    for (int i = 0; i < kStages / 4 - 1; ++i) {
      BarrierWait(SmemAddr(&bar_consumed[stage]), phase ^ 1);
      stage += 4;
      if (stage >= kStages) {
        stage = warp_id % 4;
        phase ^= 1;
      }
    }
    return;
  }

  // Compute warps.
  float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  int stage = warp_id;
  int phase = 0;
  // ldmatrix.x4.trans on B: lane group g reads output columns +8*(g&1) and K rows +8*(g>>1).
  const int group = lane_id / 8;
  const int wt_row = (group >> 1) * 8 + lane_id % 8;
  const int wt_col_bytes = (group & 1) * 16;
  // ldmatrix.x2 on A: lanes 0-7 read K bytes 0-15 of rows 0-7, lanes 8-15 read K bytes 16-31.
  const int act_row = lane_id % 8;
  const int act_chunk = (lane_id / 8) % 2;

  bool weight_ready = BarrierTryWait(SmemAddr(&bar_wt_ready[stage]), phase);
  bool act_ready = BarrierTryWait(SmemAddr(&bar_act_ready[stage]), phase);

#pragma unroll 2
  for (int ki = 0; ki < k_loops; ++ki) {
    int next_stage = stage + 4;
    int next_phase = phase;
    if (next_stage >= kStages) {
      next_stage = warp_id;
      next_phase ^= 1;
    }
    while (!weight_ready || !act_ready) {
      weight_ready = BarrierTryWait(SmemAddr(&bar_wt_ready[stage]), phase);
      act_ready = BarrierTryWait(SmemAddr(&bar_act_ready[stage]), phase);
    }
    if (ki + 1 < k_loops) {
      weight_ready = BarrierTryWait(SmemAddr(&bar_wt_ready[next_stage]), next_phase);
      act_ready = BarrierTryWait(SmemAddr(&bar_act_ready[next_stage]), next_phase);
    }

#pragma unroll
    for (int su = 0; su < kStageUnroll; ++su) {
      const uint32_t wt_tile = SmemAddr(sh_weights + (stage * kStageUnroll + su) * kWeightTileBytes);
      const uint32_t act_tile =
          SmemAddr(sh_activations + (stage * kStageUnroll + su) * kActivationTileBytes);
#pragma unroll
      for (int kii = 0; kii < kTileK / 16; ++kii) {
        uint32_t a[4];
        uint32_t b[2];
        uint32_t wt_addr = wt_tile + (kii * 16 + wt_row) * (kTileN * 2) + wt_col_bytes;
        wt_addr ^= (wt_addr >> 3) & 0x10;
        asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
                     : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                     : "r"(wt_addr));
        uint32_t act_addr = act_tile + act_row * (kTileK * 2) + (2 * kii + act_chunk) * 16;
        act_addr ^= ((act_addr >> 7) & 7) << 4;
        asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];"
                     : "=r"(b[0]), "=r"(b[1])
                     : "r"(act_addr));
        Mma16816<T>(accum, a, b);
      }
    }
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(SmemAddr(&bar_consumed[stage])));
    stage = next_stage;
    phase = next_phase;
  }

  reduction_buffer[threadIdx.x] = make_float4(accum[0], accum[1], accum[2], accum[3]);
  // Only the compute warps reach here; the loader warps may still be draining.
  asm volatile("bar.sync 1, %0;" ::"n"(kComputeThreads) : "memory");
  if (warp_id != 0) {
    return;
  }
  // Fixed summation order keeps the result deterministic.
  for (int w = 1; w < 4; ++w) {
    const float4 partial = reduction_buffer[w * 32 + lane_id];
    accum[0] += partial.x;
    accum[1] += partial.y;
    accum[2] += partial.z;
    accum[3] += partial.w;
  }
  const int col = col_base + lane_id / 4;
  const int row = row_base + 2 * (lane_id % 4);
  if (row < m) {
    if (col < n) output[static_cast<size_t>(row) * n + col] = FromFloat<T>(accum[0]);
    if (col + 8 < n) output[static_cast<size_t>(row) * n + col + 8] = FromFloat<T>(accum[2]);
  }
  if (row + 1 < m) {
    if (col < n) output[static_cast<size_t>(row + 1) * n + col] = FromFloat<T>(accum[1]);
    if (col + 8 < n) output[static_cast<size_t>(row + 1) * n + col + 8] = FromFloat<T>(accum[3]);
  }
#endif  // __CUDA_ARCH__ >= 900
}

}  // namespace tinygemm2
}  // namespace cuda
}  // namespace onnxruntime
