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
#include <limits>

#include "contrib_ops/cuda/bert/linear_attention_gates_impl.h"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// Matches OP_Sigmoid in core/providers/cuda/activation/activations_impl.cu: the branch keeps the
// exponent argument non-positive so large-magnitude inputs cannot overflow.
__device__ __forceinline__ float SigmoidFloat(float x) {
  return x > 0.0f ? 1.0f / (1.0f + expf(-x)) : 1.0f - 1.0f / (1.0f + expf(x));
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

// One block per normalization group. The input is read twice (once for the sum of squares, once
// for the output); the group is a few hundred bytes so the second read is an L1 hit.
template <typename T, int kThreadsPerBlock>
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
    y[i] = from_float<T>(normalized * (z * SigmoidFloat(z)));
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
    float epsilon) {
  if (num_rows == 0) {
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(num_rows <= std::numeric_limits<int>::max(),
                    "GatedRMSNorm launch requires too many blocks");
  const int blocks = static_cast<int>(num_rows);
#define LAUNCH_GATED_RMS_NORM(threads)                            \
  GatedRMSNormKernel<T, threads><<<blocks, threads, 0, stream>>>( \
      output, input, scale, gate, norm_size, epsilon)

  if (norm_size <= 64) {
    LAUNCH_GATED_RMS_NORM(64);
  } else if (norm_size <= 128) {
    LAUNCH_GATED_RMS_NORM(128);
  } else if (norm_size <= 256) {
    LAUNCH_GATED_RMS_NORM(256);
  } else if (norm_size <= 512) {
    LAUNCH_GATED_RMS_NORM(512);
  } else {
    LAUNCH_GATED_RMS_NORM(1024);
  }
#undef LAUNCH_GATED_RMS_NORM

  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_LINEAR_ATTENTION_GATES(T)                                                   \
  template Status LaunchLinearAttentionGateKernel<T>(cudaStream_t, T*, T*, const T*, const T*,  \
                                                     const float*, const float*, int64_t, int); \
  template Status LaunchGatedRMSNormKernel<T>(cudaStream_t, T*, const T*, const T*, const T*,   \
                                              int64_t, int, float);

INSTANTIATE_LINEAR_ATTENTION_GATES(float)
INSTANTIATE_LINEAR_ATTENTION_GATES(half)
INSTANTIATE_LINEAR_ATTENTION_GATES(__nv_bfloat16)

#undef INSTANTIATE_LINEAR_ATTENTION_GATES

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
