/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once
#include "contrib_ops/cuda/llm/weightOnlyBatchedGemv/common.h"
#include "contrib_ops/cuda/llm/weightOnlyBatchedGemv/details.h"

namespace onnxruntime::llm
{
namespace kernels
{
namespace weight_only
{
template <typename Details>
struct ConverterWrapper
{
    using TypeDetailsA = typename Details::TypeDetailsA;
    using TypeDetailsW = typename Details::TypeDetailsW;
    static constexpr bool kUseInterleavedConverter = Details::kUseInterleavedConverter;
    using Converter = I2FConverter<typename TypeDetailsA::Type, TypeDetailsW::kElemBits, kUseInterleavedConverter>;
};

template <typename DetailsA>
struct MathWrapper
{
};

template <>
struct MathWrapper<FP16DetailsA>
{
    using Type = typename FP16DetailsA::Type;
    using Type2 = typename FP16DetailsA::Type2;

    __device__ __forceinline__ static Type2 to_vec2(Type const& v)
    {
        return __half2half2(v);
    }

    __device__ __forceinline__ static Type2 fma2(Type2 const& a, Type2 const& b, Type2 const& c)
    {
        return __hfma2(a, b, c);
    }

    __device__ __forceinline__ static Type2 mul2(Type2 const& a, Type2 const& b)
    {
        return __hmul2(a, b);
    }
};

template <>
struct MathWrapper<BF16DetailsA>

{
    using Type = typename BF16DetailsA::Type;
    using Type2 = typename BF16DetailsA::Type2;

    __device__ __forceinline__ static Type2 to_vec2(Type const& v)
    {
#if ((defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)) && defined(ENABLE_BF16))
        return __bfloat162bfloat162(v);
#else
        uint32_t val = 0;
        Type2 ret = reinterpret_cast<Type2&>(val);
        return ret;
#endif
    }

    __device__ __forceinline__ static Type2 fma2(Type2 const& a, Type2 const& b, Type2 const& c)
    {
#if ((defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)) && defined(ENABLE_BF16))
        return __hfma2(a, b, c);
#else
        return to_vec2(static_cast<Type>(0.f));
#endif
    }

    __device__ __forceinline__ static Type2 mul2(Type2 const& a, Type2 const& b)
    {
#if ((defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)) && defined(ENABLE_BF16))
        return __hmul2(a, b);
#else
        return to_vec2(static_cast<Type>(0.f));
#endif
    }
};

template <typename Details, int M, int K, bool Enable>
__device__ __forceinline__ void apply_scale(void* act, void* act_scale)
{
    using Type2 = typename MathWrapper<typename Details::TypeDetailsA>::Type2;
    static_assert(K % 2 == 0);
    [[maybe_unused]] static constexpr int VecK = K / 2;
    if constexpr (Enable)
    {
        Type2* pa = reinterpret_cast<Type2*>(act);
        Type2* pb = reinterpret_cast<Type2*>(act_scale);
#pragma unroll
        for (int m = 0; m < M; ++m)
        {
#pragma unroll
            for (int k = 0; k < VecK; ++k)
            {
                pa[m * VecK + k] = MathWrapper<typename Details::TypeDetailsA>::mul2(pa[m * VecK + k], pb[k]);
            }
        }
    }
}

template <typename Details, int N, int K, bool EnableZero, bool ApplyAlphaInAdvance>
__device__ __forceinline__ void dequantize(void* w, void* quantized_w, void* scales, void* zeros, float alpha)
{
    using Type = typename MathWrapper<typename Details::TypeDetailsA>::Type;
    using Type2 = typename MathWrapper<typename Details::TypeDetailsA>::Type2;
    using Converter = typename ConverterWrapper<Details>::Converter;
    static_assert(K % 2 == 0);
    static constexpr int VecK = K / 2;
#pragma unroll
    for (int n = 0; n < N; ++n)
    {
        Converter::convert<K>(reinterpret_cast<uint8_t*>(quantized_w) + n * K / Details::kElemsPerByteW,
            reinterpret_cast<Type*>(w) + n * K);
        Type2 vec_scale, vec_zero;
        if constexpr (ApplyAlphaInAdvance)
        {
            // For W4A8, we assume scales/zero is always half data type, no matter activation dtype is bf16 or fp16
            Type scales_ = static_cast<float>(reinterpret_cast<half*>(scales)[n]) * alpha;
            vec_scale = MathWrapper<typename Details::TypeDetailsA>::to_vec2(scales_);
            vec_zero = MathWrapper<typename Details::TypeDetailsA>::to_vec2(static_cast<Type>(0.f));
            if constexpr (EnableZero)
            {
                vec_zero = MathWrapper<typename Details::TypeDetailsA>::to_vec2(
                    static_cast<float>(reinterpret_cast<half*>(zeros)[n]) * alpha);
            }
        }
        else
        {
            vec_scale = MathWrapper<typename Details::TypeDetailsA>::to_vec2(reinterpret_cast<Type*>(scales)[n]);
            vec_zero = MathWrapper<typename Details::TypeDetailsA>::to_vec2(static_cast<Type>(0.f));
            if constexpr (EnableZero)
            {
                vec_zero = MathWrapper<typename Details::TypeDetailsA>::to_vec2(reinterpret_cast<Type*>(zeros)[n]);
            }
        }
#pragma unroll
        for (int k = 0; k < VecK; ++k)
        {
            reinterpret_cast<Type2*>(w)[n * VecK + k] = MathWrapper<typename Details::TypeDetailsA>::fma2(
                reinterpret_cast<Type2*>(w)[n * VecK + k], vec_scale, vec_zero);
        }
    }
}

template <typename Details, int K>
__device__ __forceinline__ void pack_to_vec2(void* dst, void* src, int n)
{
    using Type = typename MathWrapper<typename Details::TypeDetailsA>::Type;
    typename Details::LayoutDetails::Mapper mapper;
    int n0 = n & ~0x1, n1 = n & 0x1;
    for (int k = 0; k < K; ++k)
    {
        int physical_idx = mapper(k);
        reinterpret_cast<Type*>(dst)[n0 * K + k * 2 + n1] = reinterpret_cast<Type*>(src)[physical_idx];
    }
}

template <typename Details, int M, int N, int K>
__device__ __forceinline__ void mma(void* acc, void* w_pack2, void* act)
{
    using Type = typename MathWrapper<typename Details::TypeDetailsA>::Type;
    using Type2 = typename MathWrapper<typename Details::TypeDetailsA>::Type2;
    static_assert(N % 2 == 0);
    static constexpr int VecN = N / 2;
#pragma unroll
    for (int m = 0; m < M; ++m)
    {
#pragma unroll
        for (int n = 0; n < VecN; ++n)
        {
#pragma unroll
            for (int k = 0; k < K; ++k)
            {
                reinterpret_cast<Type2*>(acc)[m * VecN + n]
                    = MathWrapper<typename Details::TypeDetailsA>::fma2(reinterpret_cast<Type2*>(w_pack2)[n * K + k],
                        MathWrapper<typename Details::TypeDetailsA>::to_vec2(reinterpret_cast<Type*>(act)[m * K + k]),
                        reinterpret_cast<Type2*>(acc)[m * VecN + n]);
            }
        }
    }
}

template <int Interleave, int ThreadsPerInterleavedTile, typename T>
__device__ __forceinline__ T warp_reduce_sum(T& val)
{
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
__device__ __forceinline__ void epilogue(void* out, int stride, void* tile_acc, void* bias, float alpha)
{
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
    for (int m = 0; m < CtaM; ++m)
    {
#pragma unroll
        for (int n = 0; n < CtaN; ++n)
        {
            float v = static_cast<float>(reinterpret_cast<Type*>(tile_acc)[m * CtaN + n]);
            v = warp_reduce_sum<Interleave, ThreadsPerInterleavedTile>(v);
            if (lane_id < Interleave * ThreadsPerInterleavedTile && lane_id % ThreadsPerInterleavedTile == 0)
            {
                shmem[warp_id * CtaM * CtaN * Interleave + m * CtaN * Interleave + n * Interleave
                    + lane_id / ThreadsPerInterleavedTile]
                    = v;
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (int ii = tid; ii < CtaM * CtaN * Interleave; ii += Threads)
    {
        int m = ii / (CtaN * Interleave), n = ii % (CtaN * Interleave);
        float val = 0.f, v_bias = 0.f;
        if constexpr (EnableBias)
        {
            v_bias = static_cast<float>(reinterpret_cast<Type*>(bias)[n]);
        }
#pragma unroll
        for (int jj = 0; jj < WarpNum; ++jj)
        {
            val += shmem[jj * CtaM * CtaN * Interleave + ii];
        }
        if constexpr (ApplyAlphaInAdvance)
        {
            reinterpret_cast<Type*>(out)[m * stride + n] = static_cast<Type>(val + v_bias);
        }
        else
        {
            reinterpret_cast<Type*>(out)[m * stride + n] = static_cast<Type>(alpha * val + v_bias);
        }
    }
}

template <int N, typename T>
__device__ __forceinline__ void fill(void* tile, T v)
{
#pragma unroll
    for (int ii = 0; ii < N; ++ii)
    {
        reinterpret_cast<T*>(tile)[ii] = v;
    }
}

template <bool Enable, typename TVec, int Strided, int Continuous, typename T>
class GMemIterator
{
public:
    __device__ __forceinline__ GMemIterator(T* addr, int offset, int step, int stride)
        : addr_(Enable ? (addr + offset) : nullptr)
        , step_(step)
        , stride_(stride)
    {
    }

    __device__ __forceinline__ void load(void* dst, int iter, int ii = 0)
    {
        if constexpr (Enable)
        {
#pragma unroll
            for (int jj = 0; jj < Continuous; ++jj)
            {
                reinterpret_cast<TVec*>(dst)[jj] = reinterpret_cast<TVec*>(addr_ + iter * step_ + ii * stride_)[jj];
            }
        }
    }

private:
    T* addr_;
    int step_;
    int stride_;
};
} // namespace weight_only
} // namespace kernels
} // namespace onnxruntime::llm
