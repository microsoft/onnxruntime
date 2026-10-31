// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Fused gate kernels for the gated-DeltaNet linear-attention layer.
//
// Both fusions exist to remove kernel launches, not to remove FLOPs: the exported graph spends
// eleven launches per layer on tensors of a few thousand elements, purely because the reference
// model computes the gates in float32 while the rest of the graph is float16. Keeping the float32
// intermediates in registers collapses each chain into a single launch and, under CUDA graphs,
// also returns the per-node replay overhead of the nodes that disappear.

#include "core/providers/cuda/cu_inc/cub.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>
#include <limits>

#include "contrib_ops/cuda/bert/linear_attention_gates_impl.h"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::contrib::linear_attention_gates_helper;

namespace {

// Matches OP_Sigmoid in core/providers/cuda/activation/activations_impl.cu: the branch keeps the
// exponent argument non-positive so large-magnitude inputs cannot overflow.
__device__ __forceinline__ float SigmoidFloat(float x) {
  if (x > 0.0f) {
    return 1.0f / (1.0f + expf(-x));
  }
  const float e = expf(x);
  return e / (1.0f + e);
}

// Matches OP_Softplus in the same file.
__device__ __forceinline__ float SoftplusFloat(float x) {
  return x > 0.0f ? x + logf(expf(-x) + 1.0f) : logf(expf(x) + 1.0f);
}

template <typename T>
__global__ void LinearAttentionGateKernel(
    T* decay,
    T* beta,
    const T* a,
    const T* b,
    const float* dt_bias,
    const float* decay_scale,
    int64_t count,
    int num_heads) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  const int h = static_cast<int>(idx % num_heads);
  const float biased = to_float<T>(a[idx]) + dt_bias[h];
  decay[idx] = from_float<T>(decay_scale[h] * SoftplusFloat(biased));

  if (beta != nullptr) {
    beta[idx] = from_float<T>(SigmoidFloat(to_float<T>(b[idx])));
  }
}

template <typename T>
__device__ __forceinline__ void Unpack4Half(uint2 u, float (&v)[4]) {
  uint16_t b0 = static_cast<uint16_t>(u.x & 0xffffu);
  uint16_t b1 = static_cast<uint16_t>(u.x >> 16);
  uint16_t b2 = static_cast<uint16_t>(u.y & 0xffffu);
  uint16_t b3 = static_cast<uint16_t>(u.y >> 16);
  v[0] = to_float<T>(*reinterpret_cast<const T*>(&b0));
  v[1] = to_float<T>(*reinterpret_cast<const T*>(&b1));
  v[2] = to_float<T>(*reinterpret_cast<const T*>(&b2));
  v[3] = to_float<T>(*reinterpret_cast<const T*>(&b3));
}

template <typename T>
__device__ __forceinline__ uint2 Pack4Half(const float (&v)[4]) {
  T t0 = from_float<T>(v[0]);
  T t1 = from_float<T>(v[1]);
  T t2 = from_float<T>(v[2]);
  T t3 = from_float<T>(v[3]);
  uint2 u;
  u.x = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&t0)) |
        (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&t1)) << 16);
  u.y = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&t2)) |
        (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&t3)) << 16);
  return u;
}

template <typename T, int kWarpsPerBlock, bool kUseSilu>
__global__ void GatedRMSNormWarp128Kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    const T* __restrict__ scale,
    const T* __restrict__ gate,
    int64_t num_rows,
    float epsilon) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_id;
  if (row >= num_rows) return;

  const int64_t vec_offset = row * 32 + lane;
  const uint2 xv2 = __ldg(reinterpret_cast<const uint2*>(input) + vec_offset);
  const uint2 gv2 = __ldg(reinterpret_cast<const uint2*>(gate) + vec_offset);
  const uint2 sv2 = __ldg(reinterpret_cast<const uint2*>(scale) + lane);

  float xv[4], gv[4], sv[4];
  Unpack4Half<T>(xv2, xv);
  Unpack4Half<T>(gv2, gv);
  Unpack4Half<T>(sv2, sv);

  float sum_sq = (xv[0] * xv[0] + xv[1] * xv[1]) + (xv[2] * xv[2] + xv[3] * xv[3]);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, off);
  }
  const float inv_rms = rsqrtf(sum_sq * (1.0f / 128.0f) + epsilon);

  float yv[4];
#pragma unroll
  for (int m = 0; m < 4; ++m) {
    const float z = gv[m];
    const float normalized = xv[m] * inv_rms * sv[m];
    const float activated_gate = kUseSilu ? (z * SigmoidFloat(z)) : SigmoidFloat(z);
    yv[m] = normalized * activated_gate;
  }
  reinterpret_cast<uint2*>(output)[vec_offset] = Pack4Half<T>(yv);
}

// One block per normalization group. The input is read twice (once for the sum of squares, once
// for the output); the group is a few hundred bytes so the second read is an L1 hit.
template <typename T, int kThreadsPerBlock, bool kUseSilu>
__global__ void GatedRMSNormKernel(
    T* output,
    const T* input,
    const T* scale,
    const T* gate,
    int norm_size,
    float epsilon) {
  const int64_t offset = static_cast<int64_t>(blockIdx.x) * norm_size;
  const T* x = input + offset;
  const T* g = gate + offset;
  T* y = output + offset;

  float sum_sq = 0.0f;
  for (int i = threadIdx.x; i < norm_size; i += kThreadsPerBlock) {
    const float v = to_float<T>(x[i]);
    sum_sq += v * v;
  }

  using BlockReduce = cub::BlockReduce<float, kThreadsPerBlock>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  const float total = BlockReduce(temp_storage).Sum(sum_sq);

  __shared__ float shared_inv_rms;
  if (threadIdx.x == 0) {
    shared_inv_rms = rsqrtf(total / static_cast<float>(norm_size) + epsilon);
  }
  __syncthreads();
  const float inv_rms = shared_inv_rms;

  for (int i = threadIdx.x; i < norm_size; i += kThreadsPerBlock) {
    const float z = to_float<T>(g[i]);
    const float normalized = to_float<T>(x[i]) * inv_rms * to_float<T>(scale[i]);
    const float activated_gate = kUseSilu ? (z * SigmoidFloat(z)) : SigmoidFloat(z);
    y[i] = from_float<T>(normalized * activated_gate);
  }
}

}  // anonymous namespace

template <typename T>
Status LaunchLinearAttentionGateKernel(
    cudaStream_t stream,
    T* decay,
    T* beta,
    const T* a,
    const T* b,
    const float* dt_bias,
    const float* decay_scale,
    int64_t num_tokens,
    int num_heads) {
  const int64_t count = num_tokens * num_heads;
  if (count == 0) {
    return Status::OK();
  }

  constexpr int kThreads = 256;
  const int64_t blocks = (count - 1) / kThreads + 1;
  ORT_RETURN_IF_NOT(blocks <= std::numeric_limits<int>::max(),
                    "LinearAttentionGate launch requires too many blocks");
  LinearAttentionGateKernel<T><<<static_cast<int>(blocks), kThreads, 0, stream>>>(
      decay, beta, a, b, dt_bias, decay_scale, count, num_heads);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status LaunchGatedRMSNormKernel(
    cudaStream_t stream,
    T* output,
    const T* input,
    const T* scale,
    const T* gate,
    int64_t num_rows,
    int norm_size,
    float epsilon,
    GatedRMSNormActivation activation,
    int max_threads_per_block) {
  if (num_rows == 0) {
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(num_rows <= std::numeric_limits<int>::max(),
                    "GatedRMSNorm launch requires too many blocks");
  ORT_RETURN_IF_NOT(max_threads_per_block >= 64, "GatedRMSNorm requires maxThreadsPerBlock >= 64");
  if constexpr (sizeof(T) == 2) {
    if (norm_size == 128 && max_threads_per_block >= 128 &&
        ((reinterpret_cast<uintptr_t>(output) | reinterpret_cast<uintptr_t>(input) |
          reinterpret_cast<uintptr_t>(scale) | reinterpret_cast<uintptr_t>(gate)) &
         7u) == 0) {
      constexpr int kWarps = 4;
      const int warp_blocks = static_cast<int>((num_rows + kWarps - 1) / kWarps);
      if (activation == GatedRMSNormActivation::kSilu) {
        GatedRMSNormWarp128Kernel<T, kWarps, true><<<warp_blocks, kWarps * 32, 0, stream>>>(
            output, input, scale, gate, num_rows, epsilon);
      } else {
        GatedRMSNormWarp128Kernel<T, kWarps, false><<<warp_blocks, kWarps * 32, 0, stream>>>(
            output, input, scale, gate, num_rows, epsilon);
      }
      return CUDA_CALL(cudaGetLastError());
    }
  }
  const int blocks = static_cast<int>(num_rows);
#define LAUNCH_GATED_RMS_NORM(threads, use_silu_mode)                            \
  GatedRMSNormKernel<T, threads, use_silu_mode><<<blocks, threads, 0, stream>>>( \
      output, input, scale, gate, norm_size, epsilon)

  const int max_threads = max_threads_per_block;
  if (norm_size <= 64 || max_threads < 128) {
    if (activation == GatedRMSNormActivation::kSilu) {
      LAUNCH_GATED_RMS_NORM(64, true);
    } else {
      LAUNCH_GATED_RMS_NORM(64, false);
    }
  } else if (norm_size <= 128 || max_threads < 256) {
    if (activation == GatedRMSNormActivation::kSilu) {
      LAUNCH_GATED_RMS_NORM(128, true);
    } else {
      LAUNCH_GATED_RMS_NORM(128, false);
    }
  } else if (norm_size <= 256 || max_threads < 512) {
    if (activation == GatedRMSNormActivation::kSilu) {
      LAUNCH_GATED_RMS_NORM(256, true);
    } else {
      LAUNCH_GATED_RMS_NORM(256, false);
    }
  } else if (norm_size <= 512 || max_threads < 1024) {
    if (activation == GatedRMSNormActivation::kSilu) {
      LAUNCH_GATED_RMS_NORM(512, true);
    } else {
      LAUNCH_GATED_RMS_NORM(512, false);
    }
  } else {
    if (activation == GatedRMSNormActivation::kSilu) {
      LAUNCH_GATED_RMS_NORM(1024, true);
    } else {
      LAUNCH_GATED_RMS_NORM(1024, false);
    }
  }
#undef LAUNCH_GATED_RMS_NORM

  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_LINEAR_ATTENTION_GATES(T)                                                   \
  template Status LaunchLinearAttentionGateKernel<T>(cudaStream_t, T*, T*, const T*, const T*,  \
                                                     const float*, const float*, int64_t, int); \
  template Status LaunchGatedRMSNormKernel<T>(cudaStream_t, T*, const T*, const T*, const T*,   \
                                              int64_t, int, float, GatedRMSNormActivation, int);

INSTANTIATE_LINEAR_ATTENTION_GATES(float)
INSTANTIATE_LINEAR_ATTENTION_GATES(half)
INSTANTIATE_LINEAR_ATTENTION_GATES(__nv_bfloat16)

#undef INSTANTIATE_LINEAR_ATTENTION_GATES

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
