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
#include <cfloat>
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

// Warp-cooperative in-register group-limited sigmoid MoE router for the standard
// 128-expert, 8-group, group_topk=2 geometry. Each warp handles one token row:
// lane l (0..31) owns 4 consecutive experts [4*l .. 4*l+3], so each 16-expert group g (0..7)
// maps to the 4 consecutive lanes [4*g .. 4*g+3].
template <typename T, int kWarpsPerBlock>
__global__ void MoEGroupRouterWarp128Kernel(
    T* __restrict__ output,
    const T* __restrict__ logits,
    const float* __restrict__ expert_bias,
    int64_t num_rows,
    int topk_group,
    int top_k,
    float router_sentinel) {
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_id;
  if (row >= num_rows) {
    return;
  }

  const int base_e = lane * 4;
  const T* row_in = logits + row * 128 + base_e;
  const float4 bias4 = *reinterpret_cast<const float4*>(expert_bias + base_e);

  float raw[4];
  if constexpr (sizeof(T) == 2) {
    const uint2 u = __ldg(reinterpret_cast<const uint2*>(row_in));
    Unpack4Half<T>(u, raw);
  } else {
    const float4 f = __ldg(reinterpret_cast<const float4*>(row_in));
    raw[0] = f.x;
    raw[1] = f.y;
    raw[2] = f.z;
    raw[3] = f.w;
  }

  const float biases[4] = {bias4.x, bias4.y, bias4.z, bias4.w};
  float s[4], b[4];
  float m1 = -FLT_MAX, m2 = -FLT_MAX;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    s[j] = SigmoidFloat(raw[j]);
    b[j] = s[j] + biases[j];
    if (b[j] > m1) {
      m2 = m1;
      m1 = b[j];
    } else if (b[j] > m2) {
      m2 = b[j];
    }
  }

  // Reduce top-2 across the 4 lanes of each 16-expert group (lane / 4 == group_id).
#pragma unroll
  for (int offset = 1; offset <= 2; offset <<= 1) {
    const float o1 = __shfl_xor_sync(0xffffffffu, m1, offset);
    const float o2 = __shfl_xor_sync(0xffffffffu, m2, offset);
    const float top1 = fmaxf(m1, o1);
    const float top2 = fmaxf(fminf(m1, o1), fmaxf(m2, o2));
    m1 = top1;
    m2 = top2;
  }
  const float grp_score = m1 + m2;
  const int grp_id = lane >> 2;

  int better_groups = 0;
#pragma unroll
  for (int go = 0; go < 8; ++go) {
    const float os = __shfl_sync(0xffffffffu, grp_score, go * 4);
    if (os > grp_score || (os == grp_score && go < grp_id)) {
      ++better_groups;
    }
  }
  if (better_groups >= topk_group) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      b[j] = -FLT_MAX;
    }
  }

  bool selected[4] = {false, false, false, false};
  for (int round = 0; round < top_k; ++round) {
    float best_val = -FLT_MAX;
    int best_idx = 2147483647;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int idx = base_e + j;
      if (b[j] > best_val || (b[j] == best_val && idx < best_idx)) {
        best_val = b[j];
        best_idx = idx;
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best_val, offset);
      const int oi = __shfl_xor_sync(0xffffffffu, best_idx, offset);
      if (ov > best_val || (ov == best_val && oi < best_idx)) {
        best_val = ov;
        best_idx = oi;
      }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (base_e + j == best_idx) {
        selected[j] = true;
        b[j] = -FLT_MAX;
      }
    }
  }

  float out_f[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    out_f[j] = selected[j] ? logf(fmaxf(s[j], 1e-20f)) : router_sentinel;
  }

  T* row_out = output + row * 128 + base_e;
  if constexpr (sizeof(T) == 2) {
    *reinterpret_cast<uint2*>(row_out) = Pack4Half<T>(out_f);
  } else {
    *reinterpret_cast<float4*>(row_out) = make_float4(out_f[0], out_f[1], out_f[2], out_f[3]);
  }
}

template <typename T>
__global__ void MoEGroupRouterGeneralKernel(
    T* __restrict__ output,
    const T* __restrict__ logits,
    const float* __restrict__ expert_bias,
    int64_t num_rows,
    int num_experts,
    int n_group,
    int topk_group,
    int group_topk,
    int top_k,
    float router_sentinel) {
  __shared__ float s_scores[512];
  __shared__ float s_biased[512];
  __shared__ float s_grp_scores[512];
  __shared__ bool s_grp_active[512];
  __shared__ bool s_expert_used[512];

  const int64_t row = blockIdx.x;
  if (row >= num_rows) return;
  const int tid = threadIdx.x;
  const T* row_in = logits + row * num_experts;
  T* row_out = output + row * num_experts;

  for (int e = tid; e < num_experts; e += blockDim.x) {
    const float sv = SigmoidFloat(to_float<T>(row_in[e]));
    s_scores[e] = sv;
    s_biased[e] = sv + expert_bias[e];
    s_expert_used[e] = false;
    row_out[e] = from_float<T>(router_sentinel);
  }
  __syncthreads();

  const int experts_per_group = num_experts / n_group;
  const int eff_group_k = min(group_topk, experts_per_group);
  for (int g = tid; g < n_group; g += blockDim.x) {
    float sum = 0.0f;
    for (int k = 0; k < eff_group_k; ++k) {
      float best_v = -FLT_MAX;
      int best_i = -1;
      for (int i = 0; i < experts_per_group; ++i) {
        const int e = g * experts_per_group + i;
        if (s_expert_used[e]) continue;
        const float v = s_biased[e];
        if (v > best_v) {
          best_v = v;
          best_i = e;
        }
      }
      if (best_i >= 0) {
        sum += best_v;
        s_expert_used[best_i] = true;
      }
    }
    s_grp_scores[g] = sum;
  }
  __syncthreads();

  for (int g = tid; g < n_group; g += blockDim.x) {
    int better = 0;
    const float my_s = s_grp_scores[g];
    for (int go = 0; go < n_group; ++go) {
      if (s_grp_scores[go] > my_s || (s_grp_scores[go] == my_s && go < g)) {
        ++better;
      }
    }
    s_grp_active[g] = (better < topk_group);
  }
  __syncthreads();

  for (int e = tid; e < num_experts; e += blockDim.x) {
    if (!s_grp_active[e / experts_per_group]) {
      s_biased[e] = -FLT_MAX;
    }
  }
  __syncthreads();

  if (tid == 0) {
    for (int k = 0; k < top_k; ++k) {
      float best_v = -FLT_MAX;
      int best_e = -1;
      for (int e = 0; e < num_experts; ++e) {
        if (s_biased[e] > best_v) {
          best_v = s_biased[e];
          best_e = e;
        }
      }
      if (best_e >= 0) {
        row_out[best_e] = from_float<T>(logf(fmaxf(s_scores[best_e], 1e-20f)));
        s_biased[best_e] = -FLT_MAX;
      }
    }
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

template <typename T>
Status LaunchMoEGroupRouterKernel(
    cudaStream_t stream,
    T* output,
    const T* logits,
    const float* expert_bias,
    int64_t num_rows,
    int num_experts,
    int n_group,
    int topk_group,
    int group_topk,
    int top_k,
    float router_sentinel) {
  if (num_rows == 0) {
    return Status::OK();
  }
  ORT_RETURN_IF_NOT(num_rows <= std::numeric_limits<int>::max(),
                    "MoEGroupRouter launch requires too many blocks");

  const uintptr_t align_mask = sizeof(T) == 2 ? 7u : 15u;
  if (num_experts == 128 && n_group == 8 && group_topk == 2 &&
      ((reinterpret_cast<uintptr_t>(output) | reinterpret_cast<uintptr_t>(logits)) & align_mask) == 0 &&
      (reinterpret_cast<uintptr_t>(expert_bias) & 15u) == 0) {
    constexpr int kWarps = 4;
    const int blocks = static_cast<int>((num_rows + kWarps - 1) / kWarps);
    MoEGroupRouterWarp128Kernel<T, kWarps><<<blocks, kWarps * 32, 0, stream>>>(
        output, logits, expert_bias, num_rows, topk_group, top_k, router_sentinel);
    return CUDA_CALL(cudaGetLastError());
  }

  const int blocks = static_cast<int>(num_rows);
  MoEGroupRouterGeneralKernel<T><<<blocks, 128, 0, stream>>>(
      output, logits, expert_bias, num_rows, num_experts, n_group, topk_group, group_topk, top_k,
      router_sentinel);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_LINEAR_ATTENTION_GATES(T)                                                    \
  template Status LaunchLinearAttentionGateKernel<T>(cudaStream_t, T*, T*, const T*, const T*,   \
                                                     const float*, const float*, int64_t, int);  \
  template Status LaunchGatedRMSNormKernel<T>(cudaStream_t, T*, const T*, const T*, const T*,    \
                                              int64_t, int, float, GatedRMSNormActivation, int); \
  template Status LaunchMoEGroupRouterKernel<T>(cudaStream_t, T*, const T*, const float*,        \
                                                int64_t, int, int, int, int, int, float);

INSTANTIATE_LINEAR_ATTENTION_GATES(float)
INSTANTIATE_LINEAR_ATTENTION_GATES(half)
INSTANTIATE_LINEAR_ATTENTION_GATES(__nv_bfloat16)

#undef INSTANTIATE_LINEAR_ATTENTION_GATES

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
