/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "contrib_ops/cuda/llm/fpA_intB_gemv/fpA_intB_gemv.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/details.h"
#include "core/common/common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime::llm {
namespace kernels {
namespace fpA_intB_gemv {

// This code are only relevant for CUDA architectures where the fpA_intB_gemv is intended to run (sm_75 and above).
// Therefore, we conditionally compile this block only when __CUDA_ARCH__ >= 750.
// This prevents compilation errors that half2 requires target sm_53 or higher.
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)) || !defined(__CUDA_ARCH__)

template <typename DetailsA>
struct MathWrapper {
};

template <>
struct MathWrapper<FP16DetailsA> {
  using Type = typename FP16DetailsA::Type;
  using Type2 = typename FP16DetailsA::Type2;

  __device__ __forceinline__ static Type2 to_vec2(Type const& v) {
    return __half2half2(v);
  }

  __device__ __forceinline__ static Type2 fma2(Type2 const& a, Type2 const& b, Type2 const& c) {
    return __hfma2(a, b, c);
  }

  __device__ __forceinline__ static Type2 mul2(Type2 const& a, Type2 const& b) {
    return __hmul2(a, b);
  }
};

template <>
struct MathWrapper<BF16DetailsA> {
  using Type = typename BF16DetailsA::Type;
  using Type2 = typename BF16DetailsA::Type2;

  __device__ __forceinline__ static Type2 to_vec2(Type const& v) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    return __bfloat162bfloat162(v);
#else
    uint32_t val = 0;
    Type2 ret = reinterpret_cast<Type2&>(val);
    return ret;
#endif
  }

  __device__ __forceinline__ static Type2 fma2(Type2 const& a, Type2 const& b, Type2 const& c) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    return __hfma2(a, b, c);
#else
    return to_vec2(static_cast<Type>(0.f));
#endif
  }

  __device__ __forceinline__ static Type2 mul2(Type2 const& a, Type2 const& b) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    return __hmul2(a, b);
#else
    return to_vec2(static_cast<Type>(0.f));
#endif
  }
};

template <typename Details, int M, int K, bool Enable>
__device__ __forceinline__ void apply_scale(void* act, void* act_scale) {
  using Type2 = typename MathWrapper<typename Details::TypeDetailsA>::Type2;
  static_assert(K % 2 == 0);
  [[maybe_unused]] static constexpr int VecK = K / 2;
  if constexpr (Enable) {
    Type2* pa = reinterpret_cast<Type2*>(act);
    Type2* pb = reinterpret_cast<Type2*>(act_scale);
#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
      for (int k = 0; k < VecK; ++k) {
        pa[m * VecK + k] = MathWrapper<typename Details::TypeDetailsA>::mul2(pa[m * VecK + k], pb[k]);
      }
    }
  }
}

template <typename Details, int N, int K, bool EnableZero, bool ApplyAlphaInAdvance>
__device__ __forceinline__ void dequantize(void* w, void* quantized_w, void* scales, void* zeros, float alpha) {
  using Type = typename MathWrapper<typename Details::TypeDetailsA>::Type;
  using Type2 = typename MathWrapper<typename Details::TypeDetailsA>::Type2;
  using Converter = typename ConverterWrapper<Details>::Converter;
  static_assert(K % 2 == 0);
  static constexpr int VecK = K / 2;
#pragma unroll
  for (int n = 0; n < N; ++n) {
    Converter::convert<K>(reinterpret_cast<uint8_t*>(quantized_w) + n * K / Details::kElemsPerByteW,
                          reinterpret_cast<Type*>(w) + n * K);
    Type2 vec_scale, vec_zero;
    if constexpr (ApplyAlphaInAdvance) {
      // For W4A8, we assume scales/zero is always half data type, no matter activation dtype is bf16 or fp16
      Type scales_ = static_cast<float>(reinterpret_cast<half*>(scales)[n]) * alpha;
      vec_scale = MathWrapper<typename Details::TypeDetailsA>::to_vec2(scales_);
      vec_zero = MathWrapper<typename Details::TypeDetailsA>::to_vec2(static_cast<Type>(0.f));
      if constexpr (EnableZero) {
        vec_zero = MathWrapper<typename Details::TypeDetailsA>::to_vec2(
            static_cast<float>(reinterpret_cast<half*>(zeros)[n]) * alpha);
      }
    } else {
      vec_scale = MathWrapper<typename Details::TypeDetailsA>::to_vec2(reinterpret_cast<Type*>(scales)[n]);
      vec_zero = MathWrapper<typename Details::TypeDetailsA>::to_vec2(static_cast<Type>(0.f));
      if constexpr (EnableZero) {
        vec_zero = MathWrapper<typename Details::TypeDetailsA>::to_vec2(reinterpret_cast<Type*>(zeros)[n]);
      }
    }
#pragma unroll
    for (int k = 0; k < VecK; ++k) {
      reinterpret_cast<Type2*>(w)[n * VecK + k] = MathWrapper<typename Details::TypeDetailsA>::fma2(
          reinterpret_cast<Type2*>(w)[n * VecK + k], vec_scale, vec_zero);
    }
  }
}

template <typename Details, int K>
__device__ __forceinline__ void pack_to_vec2(void* dst, void* src, int n) {
  using Type = typename MathWrapper<typename Details::TypeDetailsA>::Type;
  typename Details::LayoutDetails::Mapper mapper;
  int n0 = n & ~0x1, n1 = n & 0x1;
  for (int k = 0; k < K; ++k) {
    int physical_idx = mapper(k);
    reinterpret_cast<Type*>(dst)[n0 * K + k * 2 + n1] = reinterpret_cast<Type*>(src)[physical_idx];
  }
}

template <typename Details, int M, int N, int K>
__device__ __forceinline__ void mma(void* acc, void* w_pack2, void* act) {
  using Type = typename MathWrapper<typename Details::TypeDetailsA>::Type;
  using Type2 = typename MathWrapper<typename Details::TypeDetailsA>::Type2;
  static_assert(N % 2 == 0);
  static constexpr int VecN = N / 2;
#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int n = 0; n < VecN; ++n) {
#pragma unroll
      for (int k = 0; k < K; ++k) {
        reinterpret_cast<Type2*>(acc)[m * VecN + n] = MathWrapper<typename Details::TypeDetailsA>::fma2(
            reinterpret_cast<Type2*>(w_pack2)[n * K + k],
            MathWrapper<typename Details::TypeDetailsA>::to_vec2(reinterpret_cast<Type*>(act)[m * K + k]),
            reinterpret_cast<Type2*>(acc)[m * VecN + n]);
      }
    }
  }
}

template <int Interleave, int ThreadsPerInterleavedTile, typename T>
__device__ __forceinline__ T warp_reduce_sum(T& val) {
  val += __shfl_xor_sync(~0, val, 16);
  val += __shfl_xor_sync(~0, val, 8);
  if (Interleave != 2 && Interleave != 4)
    val += __shfl_xor_sync(~0, val, 4);
  if (Interleave != 4)
    val += __shfl_xor_sync(~0, val, 2);
  val += __shfl_xor_sync(~0, val, 1);
  return val;
}

template <typename Details, int CtaM, int CtaN, int Threads, bool EnableBias, bool ApplyAlphaInAdvance>
__device__ __forceinline__ void epilogue(void* out, int stride, void* tile_acc, void* bias, float alpha) {
  using Type = typename MathWrapper<typename Details::TypeDetailsA>::Type;
  static constexpr int Interleave = Details::kInterleave;
  static constexpr int ThreadsPerInterleavedTile = Details::kThreadsPerInterleavedTile;
  static constexpr int WarpSize = Details::kWarpSize;
  static constexpr int WarpNum = Threads / WarpSize;
  static_assert(Threads % WarpSize == 0);
  __shared__ float shmem[CtaM * CtaN * Interleave * WarpNum];
  int tid = threadIdx.x;
  int warp_id = tid / WarpSize, lane_id = tid % WarpSize;
#pragma unroll
  for (int m = 0; m < CtaM; ++m) {
#pragma unroll
    for (int n = 0; n < CtaN; ++n) {
      float v = static_cast<float>(reinterpret_cast<Type*>(tile_acc)[m * CtaN + n]);
      v = warp_reduce_sum<Interleave, ThreadsPerInterleavedTile>(v);
      if (lane_id < Interleave * ThreadsPerInterleavedTile && lane_id % ThreadsPerInterleavedTile == 0) {
        shmem[warp_id * CtaM * CtaN * Interleave + m * CtaN * Interleave + n * Interleave + lane_id / ThreadsPerInterleavedTile] = v;
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int ii = tid; ii < CtaM * CtaN * Interleave; ii += Threads) {
    int m = ii / (CtaN * Interleave), n = ii % (CtaN * Interleave);
    float val = 0.f, v_bias = 0.f;
    if constexpr (EnableBias) {
      v_bias = static_cast<float>(reinterpret_cast<Type*>(bias)[n]);
    }
#pragma unroll
    for (int jj = 0; jj < WarpNum; ++jj) {
      val += shmem[jj * CtaM * CtaN * Interleave + ii];
    }
    if constexpr (ApplyAlphaInAdvance) {
      reinterpret_cast<Type*>(out)[m * stride + n] = static_cast<Type>(val + v_bias);
    } else {
      reinterpret_cast<Type*>(out)[m * stride + n] = static_cast<Type>(alpha * val + v_bias);
    }
  }
}

template <int N, typename T>
__device__ __forceinline__ void fill(void* tile, T v) {
#pragma unroll
  for (int ii = 0; ii < N; ++ii) {
    reinterpret_cast<T*>(tile)[ii] = v;
  }
}
#endif  // (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)) || !defined(__CUDA_ARCH__)

template <typename Details, int CtaM, int CtaN, int Threads, int GroupSize, bool EnableActScale, bool EnableZero,
          bool EnableBias, bool ApplyAlphaInAdvance, typename TypeA = typename Details::TypeDetailsA::Type>
__global__ void kernel(TypeA* act, TypeA* act_scale, uint8_t* weight, TypeA* scales, TypeA* zeros, TypeA* bias,
                       TypeA* out, float alpha, int m, int n, int k) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
  // ArgType          ArgName          DataType           Shape                 Layout
  // input            act              fp16/bf16          [m, k]                RowMajor
  // input            act_scale        fp16/bf16          [1, k]                RowMajor
  // input            weight           int4b/int8b        [k, n]                ColumnMajor or ColumnMajorInterleaved
  // input            scales           fp16/bf16          [k / GroupSize, n]    RowMajor
  // input            zeros            fp16/bf16          [k / GroupSize, n]    RowMajor
  // input            bias             fp16/bf16          [1, n]                RowMajor
  // output           out              fp16/bf16          [m, n]                RowMajor

  using AccessTypeA = typename Details::AccessTypeA;
  using AccessTypeW = typename Details::AccessTypeW;

  static constexpr bool Mandatory = true;
  static constexpr int StepK = Details::kStepK;
  static constexpr int CtaK = StepK * Threads;
  static_assert(CtaN % 2 == 0);
  if constexpr (GroupSize != 0) {
    static_assert((CtaK / Details::kInterleave) % GroupSize == 0);
  }

  int const origin_k = k, interleaved_k = k * Details::kInterleave;

  int const tile_id_m = blockIdx.x, tile_id_n = blockIdx.y, tid = threadIdx.x;
  int const offset_m = tile_id_m * CtaM, interleaved_offset_n = tile_id_n * CtaN;
  int const real_offset_n = interleaved_offset_n * Details::kInterleave + ((tid * StepK / Details::LayoutDetails::kTileSize) % Details::kInterleave);
  int const real_offset_k = (tid * StepK / (Details::kInterleave * Details::LayoutDetails::kTileSize)) * Details::LayoutDetails::kTileSize + ((tid * StepK) % Details::LayoutDetails::kTileSize);

  GMemIterator<Mandatory, AccessTypeA, CtaM, Details::kAccessNumA, TypeA> act_iterator(
      act, offset_m * origin_k + real_offset_k, CtaK / Details::kInterleave, origin_k);
  GMemIterator<EnableActScale, AccessTypeA, 1, Details::kAccessNumA, TypeA> act_scale_iterator(
      act_scale, real_offset_k, CtaK / Details::kInterleave, 0);
  GMemIterator<Mandatory, AccessTypeW, CtaN, Details::kAccessNumW, uint8_t> weight_iterator(
      weight,
      (interleaved_offset_n * interleaved_k + tid * StepK) / Details::kElemsPerByteW, CtaK / Details::kElemsPerByteW,
      interleaved_k / Details::kElemsPerByteW);

  GMemIterator<Mandatory, TypeA, CtaN, 1, TypeA> scales_iterator(
      scales,
      (GroupSize != 0 ? real_offset_k / GroupSize * n : 0) + real_offset_n,
      (GroupSize != 0 ? CtaK / Details::kInterleave / GroupSize * n : 0), Details::kInterleave);

  GMemIterator<EnableZero, TypeA, CtaN, 1, TypeA> zeros_iterator(
      zeros,
      (GroupSize != 0 ? real_offset_k / GroupSize * n : 0) + real_offset_n,
      (GroupSize != 0 ? CtaK / Details::kInterleave / GroupSize * n : 0), Details::kInterleave);

  out += offset_m * n + tile_id_n * CtaN * Details::kInterleave;
  if constexpr (EnableBias) {
    bias += tile_id_n * CtaN * Details::kInterleave;
  }

  TypeA tile_acc[CtaM * CtaN];
  fill<CtaM * CtaN>(tile_acc, static_cast<TypeA>(0.f));

  for (int idx_k = tid * StepK, iter = 0; idx_k < interleaved_k; idx_k += CtaK, ++iter) {
    TypeA vec_act_scale[StepK];
    TypeA vec_scale[CtaN], vec_zero[CtaN];
    TypeA tile_a[StepK], tile_w[StepK], tile_w_pack2[CtaN * StepK];
    uint8_t tile_w_quantized[StepK / Details::kElemsPerByteW];
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      scales_iterator.load(vec_scale + i, iter, i);
      zeros_iterator.load(vec_zero + i, iter, i);
    }
    act_scale_iterator.load(vec_act_scale, iter);
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      weight_iterator.load(tile_w_quantized, iter, i);
      dequantize<Details, 1, StepK, EnableZero, ApplyAlphaInAdvance>(
          tile_w, tile_w_quantized, vec_scale + i, vec_zero + i, alpha);
      pack_to_vec2<Details, StepK>(tile_w_pack2, tile_w, i);
    }
#pragma unroll
    for (int i = 0; i < CtaM; ++i) {
      act_iterator.load(tile_a, iter, i);
      apply_scale<Details, 1, StepK, EnableActScale>(tile_a, vec_act_scale);
      mma<Details, 1, CtaN, StepK>(tile_acc + i * CtaN, tile_w_pack2, tile_a);
    }
  }
  epilogue<Details, CtaM, CtaN, Threads, EnableBias, ApplyAlphaInAdvance>(out, n, tile_acc, bias, alpha);
#endif
}

template <typename Details, int CtaM, int CtaN, int Threads, int GroupSize, bool EnableActScale, bool EnableZero,
          bool EnableBias, bool ApplyAlphaInAdvance>
void exec_kernel(Params& params, cudaStream_t s) {
  using T = typename Details::TypeDetailsA::Type;
  if (params.m % CtaM || params.n % (CtaN * Details::kInterleave)) {
    ORT_THROW("launch failed");
  }
  dim3 grid(params.m / CtaM, params.n / (CtaN * Details::kInterleave));
  dim3 block(Threads);
  kernel<Details, CtaM, CtaN, Threads, GroupSize, EnableActScale, EnableZero, EnableBias, ApplyAlphaInAdvance><<<grid, block, 0, s>>>(
      reinterpret_cast<T*>(params.act),
      reinterpret_cast<T*>(params.act_scale),
      reinterpret_cast<uint8_t*>(params.weight),
      reinterpret_cast<T*>(params.scales),
      reinterpret_cast<T*>(params.zeros),
      reinterpret_cast<T*>(params.bias),
      reinterpret_cast<T*>(params.out),
      params.alpha,
      params.m, params.n, params.k);
}

template <typename Details, int GroupSize, bool EnableActScale, bool EnableZero, bool EnableBias, bool ApplyAlphaInAdvance>
void dispatcher(Params& params, cudaStream_t s) {
#define DISPATCHER_FOR_M(target_m, CtaM, CtaN, Threads)                                            \
  do {                                                                                             \
    if (params.m == target_m) {                                                                    \
      exec_kernel<Details, CtaM, CtaN, Threads, GroupSize, EnableActScale, EnableZero, EnableBias, \
                  ApplyAlphaInAdvance>(params, s);                                                 \
      return;                                                                                      \
    }                                                                                              \
  } while (0);

  if constexpr (EnableZero) {
    DISPATCHER_FOR_M(1, 1, 4, 128);
    DISPATCHER_FOR_M(2, 2, 4, 128);
    DISPATCHER_FOR_M(3, 3, 4, 128);
    DISPATCHER_FOR_M(4, 4, 4, 128);
    DISPATCHER_FOR_M(5, 5, 4, 128);
    DISPATCHER_FOR_M(6, 6, 4, 128);
    DISPATCHER_FOR_M(7, 7, 4, 128);
    DISPATCHER_FOR_M(8, 8, 4, 128);
    DISPATCHER_FOR_M(9, 9, 4, 128);
    DISPATCHER_FOR_M(10, 10, 4, 128);
    DISPATCHER_FOR_M(11, 11, 4, 128);
    DISPATCHER_FOR_M(12, 12, 4, 128);
    DISPATCHER_FOR_M(13, 13, 4, 128);
    DISPATCHER_FOR_M(14, 14, 4, 128);
    DISPATCHER_FOR_M(15, 15, 4, 128);
  } else {
    DISPATCHER_FOR_M(1, 1, 8, 128);
    DISPATCHER_FOR_M(2, 2, 8, 128);
    DISPATCHER_FOR_M(3, 3, 8, 128);
    DISPATCHER_FOR_M(4, 4, 8, 128);
    DISPATCHER_FOR_M(5, 5, 8, 128);
    DISPATCHER_FOR_M(6, 6, 8, 128);
    DISPATCHER_FOR_M(7, 7, 8, 128);
    DISPATCHER_FOR_M(8, 8, 8, 128);
    DISPATCHER_FOR_M(9, 9, 8, 128);
    DISPATCHER_FOR_M(10, 10, 8, 128);
    DISPATCHER_FOR_M(11, 11, 8, 128);
    DISPATCHER_FOR_M(12, 12, 8, 128);
    DISPATCHER_FOR_M(13, 13, 8, 128);
    DISPATCHER_FOR_M(14, 14, 8, 128);
    DISPATCHER_FOR_M(15, 15, 8, 128);
  }
  ORT_THROW("unsupported m");
#undef DISPATCHER_FOR_M
}

template <typename Details, int GroupSize>
void check_pointer(Params& params, cudaStream_t s) {
  assert(!params.act_scale);               // act_scale is not supported for now.
  assert(!params.apply_alpha_in_advance);  // apply_alpha_in_advance is not supported for now.

  if (params.zeros && params.bias) {
    dispatcher<Details, GroupSize, false, true, true, false>(params, s);
  } else if (!params.zeros && params.bias) {
    dispatcher<Details, GroupSize, false, false, true, false>(params, s);
  } else if (params.zeros && !params.bias) {
    dispatcher<Details, GroupSize, false, true, false, false>(params, s);
  } else {
    dispatcher<Details, GroupSize, false, false, false, false>(params, s);
  }
}

template <bool isGroupwise, typename Details>
void select_gs(Params& params, cudaStream_t s) {
  if constexpr (isGroupwise) {
    if (params.groupsize == 64) {
      check_pointer<Details, 64>(params, s);
      return;
    } else if (params.groupsize == 128) {
      check_pointer<Details, 128>(params, s);
      return;
    }
  }

  ORT_THROW("unsupported block_size: ", params.groupsize);
}

#define INSTANTIATE_WEIGHT_ONLY_CUDA_DISPATCHERS(KType, A, B, Layout, ConverterInterleave, KTile) \
  template void select_gs<kernel_type_traits<KType>::isGroupwise,                                 \
                          KernelDetails<A, B, Layout, ConverterInterleave, KTile>>(Params & params, cudaStream_t s);

}  // namespace fpA_intB_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
