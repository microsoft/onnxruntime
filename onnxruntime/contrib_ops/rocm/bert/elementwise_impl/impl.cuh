// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/tunable/util.h"
#include "core/providers/rocm/cu_inc/common.cuh"
#include "contrib_ops/rocm/bert/elementwise.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

namespace functor {

struct FastGeLU {
  template <typename T>
  __host__ __device__ __forceinline__ void operator()(T& y, const T& x) const {
    constexpr const float b = 0.7978845608028654f;  // sqrt(2.0/M_PI)

    // const T cdf = a + a * _Tanh(in * (c * in * in + b));
    const T xb = x * T(b);
    const T u = xb * T(0.044715f) * x * x + xb;
    const T emu = __expf(-u - u);
    const T cdf = T(1.0f) / (T(1.0f) + emu);
    y = x * cdf;
  }
};

struct GeLU {
  template <typename T>
  __host__ __device__ __forceinline__ void operator()(T& y, const T& x) const {
    y = T(0.5f) * x * (T(1.f) + T(erf(0.70710678118f * float(x))));
  }
};

struct ReLU {
  template <typename T>
  __host__ __device__ __forceinline__ void operator()(T& y, const T& x) const {
    y = x >= T{} ? x : T{};
  }
};

}  // namespace functor

using onnxruntime::rocm::CeilDiv;
using onnxruntime::rocm::GPU_WARP_SIZE;

template <typename Fn, typename T, unsigned TPB>
__global__ void ElementwiseKernel(
    const T* __restrict__ input, int input_length,
    const T* __restrict__ bias, int bias_length,
    T* __restrict__ output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;
  Fn f{};

  if (idx < input_length) {
    const T x = input[idx] + (bias == nullptr ? T{} : bias[idx % bias_length]);
    f(output[idx], x);
  }
}

template <typename Fn, typename T, unsigned TPB, int ILP>
__global__ void ElementwiseKernelVec(
    const T* __restrict__ input, int input_length,
    const T* __restrict__ bias, int bias_length,
    T* output) {
  using VecT = onnxruntime::rocm::aligned_vector<T, ILP>;
  Fn f{};

  const int idx = (blockIdx.x * TPB + threadIdx.x) * ILP;
  if (idx < input_length) {
    T input_v[ILP];
    VecT* input_val = reinterpret_cast<VecT*>(&input_v);
    *input_val = *reinterpret_cast<const VecT*>(&input[idx]);
    T output_v[ILP];
    VecT* output_val = reinterpret_cast<VecT*>(&output_v);
    T bias_v[ILP];
    if (bias != nullptr) {
      VecT* bias_val = reinterpret_cast<VecT*>(&bias_v);
      *bias_val = *reinterpret_cast<const VecT*>(&bias[idx % bias_length]);
    }

#pragma unroll
    for (int i = 0; i < ILP; i++) {
      const T x = (bias == nullptr) ? input_v[i] : (T)(input_v[i] + bias_v[i]);
      f(output_v[i], x);
    }
    *(reinterpret_cast<VecT*>(&output[idx])) = *output_val;
  }
}

template <typename Fn, typename T>
Status LaunchElementwiseKernel(
    RocmTuningContext* tuning_ctx, hipStream_t stream,
    const T* input, int input_length,
    const T* bias, int bias_length,
    T* output) {
  internal::ElementwiseParams<T> params(tuning_ctx, stream, input, bias, output, input_length, bias_length);
  if (tuning_ctx->IsTunableOpEnabled()) {
    static internal::ElementwiseTunableOp<Fn, T> op;
    return op(&params);
  }

  return internal::ElementwiseStaticSelection<Fn, T>(&params);
}

namespace internal {

template <typename Fn, typename T, int ThreadsPerBlock, int VecSize>
Status ElementwiseOp<Fn, T, ThreadsPerBlock, VecSize>::operator()(const ElementwiseParams<T>* params) {
  ElementwiseKernelVec<Fn, T, ThreadsPerBlock, VecSize>
      <<<dim3(CeilDiv(params->input_length, ThreadsPerBlock * VecSize)), dim3(ThreadsPerBlock), 0, params->stream>>>(
          params->input, params->input_length,
          params->bias, params->bias_length,
          params->output);
  return HIP_CALL(hipGetLastError());
}

template <typename Fn, typename T, int ThreadsPerBlock, int VecSize>
Status ElementwiseOp<Fn, T, ThreadsPerBlock, VecSize>::IsSupported(const ElementwiseParams<T>* params) {
  // TODO(anyone): Add tail handling for FastGelu
  TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(
      !((params->bias_length > 0 && params->bias_length % VecSize == 0 && params->input_length % VecSize == 0) ||
        (params->bias_length == 0 && params->input_length % VecSize == 0)));
  // Avoid redundant configurations
  TUNABLE_OP_RETURN_UNSUPPORTED_ARGUMENT_IF(!(params->input_length > (ThreadsPerBlock - GPU_WARP_SIZE) * VecSize));

  return Status::OK();
}

template <typename Fn, typename T>
Status ElementwiseStaticSelection(const ElementwiseParams<T>* params) {
  constexpr int block_size = 256;
  if constexpr (std::is_same_v<T, half>) {
    if (params->bias != nullptr) {
      if (0 == (params->bias_length % 8) && (params->input_length >= 3145728)) {  // 3145728=8*128*3072
        const int grid_size = (params->input_length / 8 + block_size - 1) / block_size;
        ElementwiseKernelVec<Fn, half, block_size, 8><<<dim3(grid_size), dim3(block_size), 0, params->stream>>>(
            params->input, params->input_length, params->bias, params->bias_length, params->output);
      } else if (0 == (params->bias_length % 4)) {
        const int grid_size = (params->input_length / 4 + block_size - 1) / block_size;
        ElementwiseKernelVec<Fn, half, block_size, 4><<<dim3(grid_size), dim3(block_size), 0, params->stream>>>(
            params->input, params->input_length, params->bias, params->bias_length, params->output);
      } else if (0 == (params->bias_length % 2)) {
        const int grid_size = (params->input_length / 2 + block_size - 1) / block_size;
        ElementwiseKernelVec<Fn, half, block_size, 2><<<dim3(grid_size), dim3(block_size), 0, params->stream>>>(
            params->input, params->input_length, params->bias, params->bias_length, params->output);
      } else {
        const int grid_size = (params->input_length + block_size - 1) / block_size;
        ElementwiseKernel<Fn, half, block_size><<<dim3(grid_size), dim3(block_size), 0, params->stream>>>(
            params->input, params->input_length, params->bias, params->bias_length, params->output);
      }
    } else {
      if (0 == (params->input_length % 8) && (params->input_length >= 3145728)) {  // 3145728=8*128*3072
        const int grid_size = (params->input_length / 8 + block_size - 1) / block_size;
        ElementwiseKernelVec<Fn, half, block_size, 8><<<dim3(grid_size), dim3(block_size), 0, params->stream>>>(
            params->input, params->input_length, params->bias, params->bias_length, params->output);
      } else if (0 == (params->input_length % 4)) {
        const int grid_size = (params->input_length / 4 + block_size - 1) / block_size;
        ElementwiseKernelVec<Fn, half, block_size, 4><<<dim3(grid_size), dim3(block_size), 0, params->stream>>>(
            params->input, params->input_length, params->bias, params->bias_length, params->output);
      } else if (0 == (params->input_length % 2)) {
        const int grid_size = (params->input_length / 2 + block_size - 1) / block_size;
        ElementwiseKernelVec<Fn, half, block_size, 2><<<dim3(grid_size), dim3(block_size), 0, params->stream>>>(
            params->input, params->input_length, params->bias, params->bias_length, params->output);
      } else {
        const int grid_size = (params->input_length + block_size - 1) / block_size;
        ElementwiseKernel<Fn, half, block_size><<<dim3(grid_size), dim3(block_size), 0, params->stream>>>(
            params->input, params->input_length, params->bias, params->bias_length, params->output);
      }
    }
  } else {
    const int grid_size = (params->input_length + block_size - 1) / block_size;
    ElementwiseKernel<Fn, T, block_size><<<dim3(grid_size), dim3(block_size), 0, params->stream>>>(
        params->input, params->input_length, params->bias, params->bias_length, params->output);
  }
  return HIP_CALL(hipGetLastError());
}

template <typename Fn, typename T>
ElementwiseTunableOp<Fn, T>::ElementwiseTunableOp() {
  this->RegisterOp(ElementwiseStaticSelection<Fn, T>);

  this->RegisterOp(ElementwiseOp<Fn, T, 64, 1>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 64, 2>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 64, 4>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 64, 8>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 64, 16>{});

  this->RegisterOp(ElementwiseOp<Fn, T, 128, 1>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 128, 2>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 128, 4>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 128, 8>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 128, 16>{});

  this->RegisterOp(ElementwiseOp<Fn, T, 192, 1>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 192, 2>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 192, 4>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 192, 8>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 192, 16>{});

  this->RegisterOp(ElementwiseOp<Fn, T, 256, 1>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 256, 2>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 256, 4>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 256, 8>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 256, 16>{});

  this->RegisterOp(ElementwiseOp<Fn, T, 320, 1>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 320, 2>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 320, 4>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 320, 8>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 320, 16>{});

  this->RegisterOp(ElementwiseOp<Fn, T, 384, 1>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 384, 2>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 384, 4>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 384, 8>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 384, 16>{});

  this->RegisterOp(ElementwiseOp<Fn, T, 448, 1>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 448, 2>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 448, 4>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 448, 8>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 448, 16>{});

  this->RegisterOp(ElementwiseOp<Fn, T, 512, 1>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 512, 2>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 512, 4>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 512, 8>{});
  this->RegisterOp(ElementwiseOp<Fn, T, 512, 16>{});
}

#undef ADD_OP

}  // namespace internal

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime

#define ELEMENTWISE_KERNEL_IMPL(Fn, T)                        \
  namespace onnxruntime {                                     \
  namespace contrib {                                         \
  namespace rocm {                                            \
  template Status LaunchElementwiseKernel<Fn, T>(             \
      RocmTuningContext * tuning_ctx, hipStream_t stream,     \
      const T* input, int input_length,                       \
      const T* bias, int bias_length,                         \
      T* output);                                             \
  namespace internal {                                        \
  template class ElementwiseTunableOp<Fn, T>; \
  }                                                           \
  }                                                           \
  }                                                           \
  }
