/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace onnxruntime {
namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ uint32_t relu2(const uint32_t x);

template <>
inline __device__ uint32_t relu2<cutlass::half_t>(const uint32_t x) {
  uint32_t res;
  const uint32_t zero = 0u;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("max.f16x2 %0, %1, %2;\n"
               : "=r"(res)
               : "r"(x), "r"(zero));
#else
  asm volatile(
      "{\n"
      "\t .reg .f16x2 sela;\n"
      "\t set.gtu.u32.f16x2 sela, %1, %2;\n"
      "\t and.b32 %0, sela, %1;\n"
      "}\n"
      : "=r"(res)
      : "r"(x), "r"(zero));
#endif
  return res;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <>
inline __device__ uint32_t relu2<cutlass::bfloat16_t>(const uint32_t x) {
  uint32_t res;
  const uint32_t zero = 0u;
  asm volatile("max.bf16x2 %0, %1, %2;\n"
               : "=r"(res)
               : "r"(x), "r"(zero));
  return res;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

template <typename T>
inline __device__ uint32_t convert_relu2(const float2 x);

template <>
inline __device__ uint32_t convert_relu2<cutlass::half_t>(const float2 x) {
  uint32_t res;
  const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
  const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
  asm volatile("cvt.rn.relu.f16x2.f32 %0, %1, %2;\n"
               : "=r"(res)
               : "r"(b), "r"(a));
  return res;
}

template <>
inline __device__ uint32_t convert_relu2<cutlass::bfloat16_t>(const float2 x) {
  uint32_t res;
  const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
  const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
  asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n"
               : "=r"(res)
               : "r"(b), "r"(a));
  return res;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct MaxOp {
  __device__ inline T operator()(T const& x, T const& y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
  // This is slightly faster
  __device__ inline float operator()(float const& x, float const& y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct SumOp {
  __device__ inline T operator()(T const& x, T const& y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int THREADS>
struct Allreduce {
  static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
  template <typename T, typename Operator>
  static __device__ inline T run(T x, Operator& op) {
    constexpr int OFFSET = THREADS / 2;
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
    return Allreduce<OFFSET>::run(x, op);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Allreduce<2> {
  template <typename T, typename Operator>
  static __device__ inline T run(T x, Operator& op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool A_in_regs = false, bool B_in_regs = false, typename Tensor0, typename Tensor1,
          typename Tensor2, typename Tensor3, typename Tensor4,
          typename TiledMma, typename TiledCopyA, typename TiledCopyB,
          typename ThrCopyA, typename ThrCopyB>
inline __device__ void gemm(Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));   // MMA_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));   // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));  // MMA_K
  Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));  // M
  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));  // N
  if (!A_in_regs) {
    cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
  }
  if (!B_in_regs) {
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
  }
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      if (!A_in_regs) {
        cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
      }
      if (!B_in_regs) {
        cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
      }
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
          typename TiledMma, typename TiledCopy, typename ThrCopy>
inline __device__ void gemm_A_in_regs(Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB, Tensor3 const& tCsB,
                                      TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                      ThrCopy smem_thr_copy_B) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));   // MMA_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));   // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));  // MMA_K
  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));  // N
  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template <typename Layout>
inline __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
  static_assert(decltype(size<0>(acc_layout))::value == 4);
  static_assert(decltype(rank(acc_layout))::value == 3);
  auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
                                                     // TD [2023-08-13]: Idk why but get<0, 1>(l) doesn't work for Cutlass 3.2, I'm getting
  // "int_tuple.hpp(74): error: conversion to inaccessible base class"
  // return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
  return make_layout(make_layout(get<1>(get<0>(l)), get<1>(l)), make_layout(get<0>(get<0>(l)), get<2>(l)));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert rowcol_layout from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to ((2, 2, 1), MMA_M, MMA_N) if using m16n8k8.
template <typename MMA_traits, typename Layout>
inline __device__ auto convert_layout_rowcol_Aregs(Layout rowcol_layout) {
  using X = Underscore;
  static_assert(decltype(size<0, 0>(rowcol_layout))::value == 2);
  static_assert(decltype(size<1, 0>(rowcol_layout))::value == 2);
  constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
  static_assert(mma_shape_K == 8 || mma_shape_K == 16);
  constexpr int MMA_N_divisor = mma_shape_K == 8 ? 1 : 2;
  auto l = logical_divide(rowcol_layout, Shape<X, Shape<X, Int<MMA_N_divisor>>>{});  // ((2, MMA_M), (2, (2, MMA_N / 2)))
                                                                                     // TD [2023-08-13]: Same error as above on Cutlass 3.2
  // return make_layout(make_layout(get<1, 0>(l), get<0, 0>(l), get<1, 1, 0>(l)),
  //                    get<0, 1>(l),
  //                    get<1, 1, 1>(l));
  return make_layout(make_layout(get<0>(get<1>(l)), get<0>(get<0>(l)), get<0>(get<1>(get<1>(l)))),
                     get<1>(get<0>(l)),
                     get<1>(get<1>(get<1>(l))));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
inline __device__ auto convert_type(Tensor<Engine, Layout> const& tensor) {
  using From_type = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  // HACK: this requires tensor to be "contiguous"
  auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
inline __device__ void relu_(Tensor<Engine, Layout>& tensor) {
  constexpr int numel = decltype(size(tensor))::value;
  static_assert(numel % 2 == 0);
  using value_t = typename Engine::value_type;
  // HACK: this requires tensor to be "contiguous"
  Tensor tensor_uint32 = recast<uint32_t>(tensor);
#pragma unroll
  for (int i = 0; i < size(tensor_uint32); ++i) {
    tensor_uint32(i) = relu2<value_t>(tensor_uint32(i));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// On SM80 and above, we can fuse fp32 -> fp16/bf16 conversion and relu into 1 instruction
template <typename To_type, typename Engine, typename Layout>
inline __device__ auto convert_type_relu(Tensor<Engine, Layout> const& tensor) {
  using From_type = typename Engine::value_type;
  static_assert(std::is_same_v<To_type, cutlass::half_t> || std::is_same_v<To_type, cutlass::bfloat16_t>);
  static_assert(std::is_same_v<float, From_type>);
  constexpr int numel = decltype(size(tensor))::value;
  static_assert(numel % 2 == 0);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // HACK: this requires tensor to be "contiguous"
  Tensor tensor_float2 = recast<float2>(tensor);
  Tensor out_uint32 = make_tensor<uint32_t>(tensor_float2.layout());
#pragma unroll
  for (int i = 0; i < size(out_uint32); ++i) {
    out_uint32(i) = convert_relu2<To_type>(tensor_float2(i));
  }
  Tensor out = make_tensor(make_rmem_ptr<To_type>(out_uint32.data()), tensor.layout());
#else
  Tensor out = flash::convert_type<To_type>(tensor);
  flash::relu_(out);
#endif
  return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
CUTE_HOST_DEVICE void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN = true, bool Is_even_K = true, bool Clear_OOB_MN = false, bool Clear_OOB_K = true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
inline __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const& S,
                            Tensor<Engine1, Layout1>& D, Tensor<Engine2, Layout2> const& identity_MN,
                            Tensor<Engine3, Layout3> const& predicate_K, const int max_MN = 0) {
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));  // MMA
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));  // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));  // MMA_K
  // There's no case where !Clear_OOB_K && Clear_OOB_MN
  static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
#pragma unroll
  for (int m = 0; m < size<1>(S); ++m) {
    if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
      for (int k = 0; k < size<2>(S); ++k) {
        if (Is_even_K || predicate_K(k)) {
          cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
        } else if (Clear_OOB_K) {
          cute::clear(D(_, m, k));
        }
      }
    } else if (Clear_OOB_MN) {
      cute::clear(D(_, m, _));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K = true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
inline __device__ void copy_w_min_idx(Tensor<Engine0, Layout0> const& S,
                                      Tensor<Engine1, Layout1>& D, Tensor<Engine2, Layout2> const& identity_MN,
                                      Tensor<Engine3, Layout3> const& predicate_K,
                                      const int max_MN = 0, const int min_MN = 0) {
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));  // MMA
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));  // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));  // MMA_K
// if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, max_MN = %d, min_MN = %d\n", blockIdx.y, max_MN, min_MN); }
#pragma unroll
  for (int m = 0; m < size<1>(S); ++m) {
    // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
    if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
// if (threadIdx.x == 0 && blockIdx.z == 0) { printf("Inner loop, blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
#pragma unroll
      for (int k = 0; k < size<2>(S); ++k) {
        if (Is_even_K || predicate_K(k)) {
          cute::copy(S(_, m, k), D(_, m, k));
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K = true, bool Clear_OOB_K = true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
inline __device__ void copy_rotary_interleaved(Tensor<Engine0, Layout0> const& S,
                                               Tensor<Engine1, Layout1>& D,
                                               Tensor<Engine2, Layout2> const& Cos,
                                               Tensor<Engine2, Layout2> const& Sin,
                                               Tensor<Engine3, Layout3> const& identity_MN,
                                               const int max_MN, const int min_MN,
                                               const int dim, const int rotary_dim) {
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));      // MMA
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));      // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));      // MMA_K
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Cos));    // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Cos));    // MMA_K
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Sin));    // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Sin));    // MMA_K
  CUTE_STATIC_ASSERT_V(size<0>(Cos) == size<0>(Sin));  // MMA_K
  static_assert(decltype(size<0>(S))::value == decltype(size<0>(Cos))::value * 2);
  static_assert(decltype(size<0>(Cos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
  Tensor rCos = make_fragment_like(Cos);
  Tensor rSin = make_fragment_like(Sin);
  Tensor rS = make_fragment_like(S);
#pragma unroll
  for (int m = 0; m < size<1>(S); ++m) {
    if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
      for (int k = 0; k < size<2>(S); ++k) {
        if (Is_even_K || get<1>(identity_MN(0, 0, k)) < dim) {
          cute::copy(S(_, m, k), rS(_, m, k));
          if (get<1>(identity_MN(0, 0, k)) < rotary_dim) {
            cute::copy(Cos(_, m, k), rCos(_, m, k));
            cute::copy(Sin(_, m, k), rSin(_, m, k));
            Tensor S_fp32 = convert_type<float>(rS(_, m, k));
            Tensor cos_fp32 = convert_type<float>(rCos(_, m, k));
            Tensor sin_fp32 = convert_type<float>(rSin(_, m, k));
#pragma unroll
            for (int i = 0; i < size<0>(rS) / 2; ++i) {
              float real = S_fp32(2 * i) * cos_fp32(i) - S_fp32(2 * i + 1) * sin_fp32(i);
              float imag = S_fp32(2 * i) * sin_fp32(i) + S_fp32(2 * i + 1) * cos_fp32(i);
              S_fp32(2 * i) = real;
              S_fp32(2 * i + 1) = imag;
            }
            // Idk but I need to copy for the convert_type to work
            Tensor S_fp32_copy = make_fragment_like(S_fp32);
            cute::copy(S_fp32, S_fp32_copy);
            using T = typename Engine0::value_type;
            Tensor S_og_type = convert_type<T>(S_fp32_copy);
            cute::copy(S_og_type, rS(_, m, k));
          }
          cute::copy(rS(_, m, k), D(_, m, k));
        } else if (Clear_OOB_K) {
          cute::clear(D(_, m, k));
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K = true, bool Clear_OOB_K = true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
inline __device__ void copy_rotary_contiguous(Tensor<Engine0, Layout0> const& S,
                                              Tensor<Engine1, Layout1>& D,
                                              Tensor<Engine2, Layout2> const& Cos,
                                              Tensor<Engine2, Layout2> const& Sin,
                                              Tensor<Engine3, Layout3> const& identity_MN,
                                              const int max_MN, const int min_MN,
                                              const int dim, const int rotary_dim) {
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));    // MMA
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));    // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));    // MMA_K
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Cos));  // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Cos));  // MMA_K
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Sin));  // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Sin));  // MMA_K
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(Cos));  // MMA
  CUTE_STATIC_ASSERT_V(size<0>(Cos) == size<0>(Sin));
  static_assert(decltype(size<0>(Cos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
  Tensor rCos = make_fragment_like(Cos);
  Tensor rSin = make_fragment_like(Sin);
  Tensor rS = make_fragment_like(S);
  Tensor rS_other = make_fragment_like(rS(_, 0, 0));
#pragma unroll
  for (int m = 0; m < size<1>(S); ++m) {
    if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
      for (int k = 0; k < size<2>(S); ++k) {
        if (Is_even_K || get<1>(identity_MN(0, 0, k)) < dim) {
          cute::copy(S(_, m, k), rS(_, m, k));
          if (get<1>(identity_MN(0, 0, k)) < rotary_dim) {
            const bool is_left = get<1>(identity_MN(0, 0, k)) < rotary_dim / 2;
            Tensor gS_other = make_tensor(S(_, m, k).data() + (is_left ? rotary_dim / 2 : -rotary_dim / 2), S(_, m, k).layout());
            cute::copy(gS_other, rS_other);
            // if (cute::thread0()) { print_tensor(rS(_, m, k)); print_tensor(rS_other); }
            Tensor gCos = make_tensor(Cos(_, m, k).data() + (is_left ? 0 : -rotary_dim / 2), Cos(_, m, k).layout());
            Tensor gSin = make_tensor(Sin(_, m, k).data() + (is_left ? 0 : -rotary_dim / 2), Sin(_, m, k).layout());
            cute::copy(gCos, rCos(_, m, k));
            cute::copy(gSin, rSin(_, m, k));
            // if (cute::thread0()) { print_tensor(rCos(_, m, k)); print_tensor(rSin(_, m, k)); }
            Tensor S_fp32 = convert_type<float>(rS(_, m, k));
            Tensor S_other_fp32 = convert_type<float>(rS_other);
            Tensor cos_fp32 = convert_type<float>(rCos(_, m, k));
            Tensor sin_fp32 = convert_type<float>(rSin(_, m, k));
#pragma unroll
            for (int i = 0; i < size<0>(rS); ++i) {
              S_fp32(i) = S_fp32(i) * cos_fp32(i) + S_other_fp32(i) * (is_left ? -sin_fp32(i) : sin_fp32(i));
            }
            // Idk but I need to copy for the convert_type to work
            Tensor S_fp32_copy = make_fragment_like(S_fp32);
            cute::copy(S_fp32, S_fp32_copy);
            using T = typename Engine0::value_type;
            Tensor S_og_type = convert_type<T>(S_fp32_copy);
            cute::copy(S_og_type, rS(_, m, k));
            // if (cute::thread0()) { print_tensor(rS(_, m, k)); }
          }
          cute::copy(rS(_, m, k), D(_, m, k));
        } else if (Clear_OOB_K) {
          cute::clear(D(_, m, k));
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
}  // namespace onnxruntime
