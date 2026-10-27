// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

#include <cmath>
#include <limits>

#include "contrib_ops/cuda/bert/gated_add_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fast_divmod.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

constexpr int kProjectedGateThreads = 256;

__device__ __forceinline__ float StableSigmoid(float value) {
  if (value >= 0.0f) {
    return 1.0f / (1.0f + expf(-value));
  }
  const float exp_value = expf(value);
  return exp_value / (1.0f + exp_value);
}

__device__ __forceinline__ float RoundedBf16Sigmoid(float value) {
  return __bfloat162float(__float2bfloat16_rn(StableSigmoid(value)));
}

__device__ float ProjectedRowGate(const __nv_bfloat16* activation, const __nv_bfloat16* weight,
                                  int64_t row, int64_t input_width) {
  using Reduce = cub::BlockReduce<float, kProjectedGateThreads>;
  __shared__ typename Reduce::TempStorage reduction;
  __shared__ float gate;
  float dot = 0.0f;
  for (int64_t k = threadIdx.x; k < input_width; k += kProjectedGateThreads) {
    dot = fmaf(__bfloat162float(activation[row * input_width + k]),
               __bfloat162float(weight[k]), dot);
  }
  const float total = Reduce(reduction).Sum(dot);
  if (threadIdx.x == 0) {
    const float value = __bfloat162float(__float2bfloat16_rn(total));
    gate = RoundedBf16Sigmoid(value);
  }
  __syncthreads();
  return gate;
}

__global__ void ProjectedSigmoidGateMulKernel(
    const __nv_bfloat16* attention, const __nv_bfloat16* activation,
    const __nv_bfloat16* weight, const __nv_bfloat16* residual, __nv_bfloat16* output,
    int64_t input_width, int64_t output_width) {
  const int64_t row = blockIdx.x;
  const float gate = ProjectedRowGate(activation, weight, row, input_width);
  const int64_t row_base = row * output_width;
  if ((output_width & 7) == 0 &&
      (reinterpret_cast<uintptr_t>(attention + row_base) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(output + row_base) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(residual + row_base) & 15) == 0) {
    const int64_t vec_width = output_width >> 3;
    const uint4* attn_vec = reinterpret_cast<const uint4*>(attention + row_base);
    const uint4* res_vec = reinterpret_cast<const uint4*>(residual + row_base);
    uint4* out_vec = reinterpret_cast<uint4*>(output + row_base);
    for (int64_t v = threadIdx.x; v < vec_width; v += kProjectedGateThreads) {
      const uint4 a_u4 = attn_vec[v];
      const uint4 r_u4 = res_vec[v];
      const __nv_bfloat16* a_bf = reinterpret_cast<const __nv_bfloat16*>(&a_u4);
      const __nv_bfloat16* r_bf = reinterpret_cast<const __nv_bfloat16*>(&r_u4);
      uint4 o_u4;
      __nv_bfloat16* o_bf = reinterpret_cast<__nv_bfloat16*>(&o_u4);
#pragma unroll
      for (int m = 0; m < 8; ++m) {
        const __nv_bfloat16 scaled = __float2bfloat16_rn(__bfloat162float(a_bf[m]) * gate);
        o_bf[m] = __float2bfloat16_rn(__bfloat162float(scaled) + __bfloat162float(r_bf[m]));
      }
      out_vec[v] = o_u4;
    }
    return;
  }
  for (int64_t column = threadIdx.x; column < output_width; column += kProjectedGateThreads) {
    const int64_t index = row_base + column;
    const __nv_bfloat16 scaled = __float2bfloat16_rn(__bfloat162float(attention[index]) * gate);
    output[index] = __float2bfloat16_rn(__bfloat162float(scaled) + __bfloat162float(residual[index]));
  }
}

template <typename T>
__device__ __forceinline__ T RoundedMulAdd(T x, T y, T gate);

template <>
__device__ __forceinline__ float RoundedMulAdd(float x, float y, float gate) {
  return __fadd_rn(x, __fmul_rn(y, gate));
}

template <>
__device__ __forceinline__ half RoundedMulAdd(half x, half y, half gate) {
  unsigned short product;
  unsigned short output;
  asm volatile("mul.rn.f16 %0, %1, %2;"
               : "=h"(product)
               : "h"(__half_as_ushort(y)), "h"(__half_as_ushort(gate)));
  asm volatile("add.rn.f16 %0, %1, %2;"
               : "=h"(output)
               : "h"(__half_as_ushort(x)), "h"(product));
  return __ushort_as_half(output);
}

template <>
__device__ __forceinline__ __nv_bfloat16 RoundedMulAdd(
    __nv_bfloat16 x, __nv_bfloat16 y, __nv_bfloat16 gate) {
  // FP32 represents BF16 products exactly; round between operations to match separate Mul and Add.
  const __nv_bfloat16 product = __float2bfloat16_rn(
      __fmul_rn(__bfloat162float(y), __bfloat162float(gate)));
  return __float2bfloat16_rn(
      __fadd_rn(__bfloat162float(x), __bfloat162float(product)));
}

template <typename T, typename IndexT, typename DivModT>
__global__ void GatedAddKernel(T* output,
                               const T* x,
                               const T* y,
                               const T* gate,
                               IndexT count,
                               DivModT hidden_size) {
  const IndexT index = static_cast<IndexT>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = RoundedMulAdd(x[index], y[index], gate[hidden_size.div(index)]);
  }
}

template <typename T>
__global__ void GatedAddVec8Kernel(uint4* output,
                                   const uint4* x,
                                   const uint4* y,
                                   const T* gate,
                                   int vec_count,
                                   onnxruntime::cuda::fast_divmod vec_hidden_size) {
  const int v = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v < vec_count) {
    const T g = gate[vec_hidden_size.div(v)];
    const uint4 x_u4 = x[v];
    const uint4 y_u4 = y[v];
    uint4 out_u4;
    const T* x_t = reinterpret_cast<const T*>(&x_u4);
    const T* y_t = reinterpret_cast<const T*>(&y_u4);
    T* out_t = reinterpret_cast<T*>(&out_u4);
#pragma unroll
    for (int m = 0; m < 8; ++m) {
      out_t[m] = RoundedMulAdd(x_t[m], y_t[m], g);
    }
    output[v] = out_u4;
  }
}

}  // namespace

template <typename T>
Status LaunchGatedAddKernel(cudaStream_t stream,
                            T* output,
                            const T* x,
                            const T* y,
                            const T* gate,
                            int64_t count,
                            int64_t hidden_size) {
  if (count == 0) {
    return Status::OK();
  }

  constexpr int kThreads = 256;
  if constexpr (sizeof(T) == 2) {
    if ((hidden_size & 7) == 0 && (count >> 3) <= std::numeric_limits<int>::max() &&
        (reinterpret_cast<uintptr_t>(output) & 15) == 0 &&
        (reinterpret_cast<uintptr_t>(x) & 15) == 0 &&
        (reinterpret_cast<uintptr_t>(y) & 15) == 0) {
      const int vec_count = static_cast<int>(count >> 3);
      const int vec_blocks = (vec_count - 1) / kThreads + 1;
      GatedAddVec8Kernel<T><<<vec_blocks, kThreads, 0, stream>>>(
          reinterpret_cast<uint4*>(output),
          reinterpret_cast<const uint4*>(x),
          reinterpret_cast<const uint4*>(y),
          gate,
          vec_count,
          onnxruntime::cuda::fast_divmod(static_cast<int>(hidden_size >> 3)));
      return CUDA_CALL(cudaGetLastError());
    }
  }
  const int64_t blocks = (count - 1) / kThreads + 1;
  ORT_RETURN_IF_NOT(blocks <= std::numeric_limits<int>::max(),
                    "GatedAdd launch requires too many blocks");
  if (count <= std::numeric_limits<int>::max()) {
    GatedAddKernel<T, int, onnxruntime::cuda::fast_divmod><<<static_cast<int>(blocks), kThreads, 0, stream>>>(
        output, x, y, gate, static_cast<int>(count),
        onnxruntime::cuda::fast_divmod(static_cast<int>(hidden_size)));
  } else {
    GatedAddKernel<T, int64_t, onnxruntime::cuda::DivMod<int64_t>><<<static_cast<int>(blocks), kThreads, 0, stream>>>(
        output, x, y, gate, count, onnxruntime::cuda::DivMod<int64_t>(hidden_size));
  }
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_GATED_ADD(T)                                      \
  template Status LaunchGatedAddKernel<T>(cudaStream_t, T*, const T*, \
                                          const T*, const T*, int64_t, int64_t);

INSTANTIATE_GATED_ADD(float)
INSTANTIATE_GATED_ADD(half)
INSTANTIATE_GATED_ADD(__nv_bfloat16)

#undef INSTANTIATE_GATED_ADD

Status LaunchProjectedSigmoidGateMul(cudaStream_t stream, const uint16_t* attention,
                                     const uint16_t* activation, const uint16_t* weight,
                                     const uint16_t* residual, uint16_t* output,
                                     int64_t rows, int64_t input_width, int64_t output_width) {
  ORT_RETURN_IF_NOT(rows >= 0 && rows <= std::numeric_limits<int>::max() &&
                        input_width > 0 && output_width > 0,
                    "GatedAdd projected gate dimensions exceed CUDA limits");
  if (rows == 0) return Status::OK();
  ProjectedSigmoidGateMulKernel<<<static_cast<int>(rows), kProjectedGateThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(attention),
      reinterpret_cast<const __nv_bfloat16*>(activation),
      reinterpret_cast<const __nv_bfloat16*>(weight),
      reinterpret_cast<const __nv_bfloat16*>(residual), reinterpret_cast<__nv_bfloat16*>(output),
      input_width, output_width);
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime