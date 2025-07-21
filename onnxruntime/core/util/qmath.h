// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include "core/common/narrow.h"
#include "core/framework/element_type_lists.h"
#include "core/framework/float8.h"
#include "core/framework/int4.h"
#include <cmath>
#include <limits>
#include <algorithm>

namespace onnxruntime {

inline float RoundHalfToEven(float input) {
  if (!std::isfinite(input)) {
    return input;
  }
  // std::remainder returns x - n, where n is the integral value nearest to x. When |x - n| = 0.5, n is chosen to be even
  return input - std::remainderf(input, 1.f);
}

template <typename T>
struct is_quant_type : std::false_type {};

template <>
struct is_quant_type<int8_t> : std::true_type {};

template <>
struct is_quant_type<uint8_t> : std::true_type {};

// Define max number of parallel threads for computing min max values
// This is hacky. Ideally we should let the thread pool handle work partition.
// Unfortunately I can't find an elegant way for aggregation at this point.
#define MAX_DEGREE_OF_PAR_FOR_MINMAX 32

struct FloatMinMax {
  float min;
  float max;
};

// ReduceRange and Symmetric is for test only
template <typename QType,
          bool ReduceRange = false,
          bool Symmetric = false,
          typename std::enable_if<is_quant_type<QType>::value, int>::type = 0>
void GetQuantizationParameter(const float* data, int64_t num_of_elements, float& scale, QType& zp, concurrency::ThreadPool* thread_pool) {
  FloatMinMax aggregate[MAX_DEGREE_OF_PAR_FOR_MINMAX];

  // Min max operation granularity: AVX512 can potentially handle 64 ~ 128 floats
  // per iteration.
  constexpr int granularity = 128;
  std::ptrdiff_t block_size;
  std::ptrdiff_t num_blocks;
  if (concurrency::ThreadPool::ShouldParallelize(thread_pool) && num_of_elements > granularity) {
    block_size = onnxruntime::narrow<std::ptrdiff_t>((num_of_elements + MAX_DEGREE_OF_PAR_FOR_MINMAX - 1) / MAX_DEGREE_OF_PAR_FOR_MINMAX);
    block_size = (block_size + granularity - 1) / granularity * granularity;
    num_blocks = onnxruntime::narrow<std::ptrdiff_t>((num_of_elements + block_size - 1) / block_size);
  } else {
    num_blocks = 1;
    block_size = onnxruntime::narrow<std::ptrdiff_t>(num_of_elements);
  }

  for (int i = 0; i < narrow<int>(num_blocks); i++) {
    aggregate[i].min = std::numeric_limits<float>::max();
    aggregate[i].max = std::numeric_limits<float>::lowest();
  }

  const TensorOpCost unit_cost{static_cast<double>(block_size) * sizeof(float), 2.0, static_cast<double>(block_size)};
  concurrency::ThreadPool::TryParallelFor(thread_pool, num_blocks, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    auto begin_idx = begin * block_size;
    auto end_idx = std::min(std::ptrdiff_t(num_of_elements), end * block_size);
    auto agg_idx = begin % num_blocks;
    MlasFindMinMaxElement(&(data[begin_idx]), &aggregate[agg_idx].min, &aggregate[agg_idx].max, end_idx - begin_idx);
  });

  float& min = aggregate[0].min;
  float& max = aggregate[0].max;
  for (int i = 1; i < narrow<int>(num_blocks); i++) {
    min = std::min(min, aggregate[i].min);
    max = std::max(max, aggregate[i].max);
  }
  // ensure the input range includes zero
  min = std::min(min, 0.0f);
  max = std::max(max, 0.0f);

  // find scale and zero point
  QType qmin = std::numeric_limits<QType>::min();
  QType qmax = std::numeric_limits<QType>::max();
  if (std::is_same<QType, int8_t>::value) {
    if (ReduceRange) {
      qmin = static_cast<QType>(-64);
      qmax = static_cast<QType>(64);
    }

    if (Symmetric) {
      zp = 0;
      float max_value = std::max(max, -min);
      scale = max_value > 0 ? max_value / qmax : 1.f;
      return;
    }
  }
  scale = max == min ? 1.0f : (max - min) / float(qmax - qmin);

  float initial_zero_point = qmin - min / scale;
  zp = static_cast<QType>(RoundHalfToEven(std::max(float(qmin), std::min(float(qmax), initial_zero_point))));
}

/**
 * @brief Run MlasQuantizeLinear in parallel, with provided thread pool
 */

template <typename OutputType>
#if !defined(DISABLE_FLOAT8_TYPES)
typename std::enable_if<!boost::mp11::mp_contains<element_type_lists::AllFloat8, OutputType>::value, void>::type
#else
void
#endif
ParQuantizeLinearStd(const float* Input,
                     OutputType* Output,
                     size_t N,
                     float Scale,
                     OutputType ZeroPoint,
                     concurrency::ThreadPool* thread_pool) {
  constexpr std::ptrdiff_t block_size = 128;
  const std::ptrdiff_t num_blocks = (N + block_size - 1) / block_size;
  const TensorOpCost unit_cost{static_cast<double>(block_size * sizeof(float)), static_cast<double>(block_size * sizeof(uint8_t)), static_cast<double>(block_size) * 2.0};
  concurrency::ThreadPool::TryParallelFor(thread_pool, num_blocks, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    auto begin_idx = begin * block_size;
    auto end_idx = std::min(static_cast<std::ptrdiff_t>(N), end * block_size);
    MlasQuantizeLinear(&(Input[begin_idx]), &(Output[begin_idx]), end_idx - begin_idx, Scale, ZeroPoint);
  });
}

/**
 * Defines a function for int4 quantization. Calls MLAS kernel in parallel with the provided thread pool.
 *
 * \param FUNC_NAME The name of the generated function.
 * \param INT4_TYPE The int4 type (i.e., either Int4x2 or UInt4x2)
 * \param MLAS_FUNC The MLAS quantization kernel to call.
 * \param Input The input float values to quantize. Must contain `out_end - out_start` elements.
 * \param Output The output buffer that will contain the quantized values.
 * \param out_start The int4 element index at which to start writing to the output buffer.
 *                  Divide by 2 to get index into Output buffer.
 * \param out_end The int4 element index at which to stop writing to the output buffer.
 *                Divide by 2 to get index into Output buffer.
 * \param Scale The quantization scale value.
 * \param ZeroPoint The quantization zero-point value.
 * \param thread_pool The thread pool to use.
 */
#define DEFINE_PAR_QUANT_LINEAR_STD_4BIT(FUNC_NAME, INT4_TYPE, MLAS_FUNC)                                        \
  inline void FUNC_NAME(const float* Input,                                                                      \
                        INT4_TYPE* Output,                                                                       \
                        size_t out_start,                                                                        \
                        size_t out_end,                                                                          \
                        float Scale,                                                                             \
                        INT4_TYPE ZeroPoint,                                                                     \
                        concurrency::ThreadPool* thread_pool) {                                                  \
    size_t inp_start = 0;                                                                                        \
    size_t inp_end = out_end - out_start;                                                                        \
                                                                                                                 \
    /* If starting at an int4 element in the middle of a byte, quantize it by itself. */                         \
    if (out_start & 0x1) {                                                                                       \
      int32_t ival = static_cast<int32_t>(std::nearbyintf(Input[inp_start] / Scale)) +                           \
                     static_cast<int32_t>(ZeroPoint.GetElem(0));                                                 \
      size_t output_index = out_start >> 1;                                                                      \
                                                                                                                 \
      INT4_TYPE::UnpackedType quant_val = static_cast<INT4_TYPE::UnpackedType>(                                  \
          std::min(static_cast<int32_t>(INT4_TYPE::max_val),                                                     \
                   std::max(static_cast<int32_t>(INT4_TYPE::min_val), ival)));                                   \
      Output[output_index].SetElem(1, quant_val);                                                                \
                                                                                                                 \
      out_start += 1;                                                                                            \
      inp_start += 1;                                                                                            \
    }                                                                                                            \
                                                                                                                 \
    /* If ending at element that ends in the middle of a byte, quantize it by itself. */                         \
    if (out_end & 0x1) {                                                                                         \
      int32_t ival = static_cast<int32_t>(std::nearbyintf(Input[inp_end - 1] / Scale)) +                         \
                     static_cast<int32_t>(ZeroPoint.GetElem(0));                                                 \
      size_t output_index = (out_end - 1) >> 1;                                                                  \
                                                                                                                 \
      INT4_TYPE::UnpackedType quant_val = static_cast<INT4_TYPE::UnpackedType>(                                  \
          std::min(static_cast<int32_t>(INT4_TYPE::max_val),                                                     \
                   std::max(static_cast<int32_t>(INT4_TYPE::min_val), ival)));                                   \
      Output[output_index].SetElem(0, quant_val);                                                                \
                                                                                                                 \
      out_end -= 1;                                                                                              \
      inp_end -= 1;                                                                                              \
    }                                                                                                            \
                                                                                                                 \
    if (out_start == out_end) {                                                                                  \
      return;                                                                                                    \
    }                                                                                                            \
                                                                                                                 \
    /* At this point, should only need to quantize an *even* number of int4 elements that start and end at */    \
    /* a byte boundary. This is necessary to ensure that no two threads write to different int4 elements that */ \
    /* are stored in the same byte. */                                                                           \
    size_t N = out_end - out_start;                                                                              \
    assert(N % 2 == 0); /* Should be guaranteed by previous code that quantizes boundary elements. */            \
                                                                                                                 \
    constexpr std::ptrdiff_t block_size = 128;                                                                   \
    static_assert(block_size % 2 == 0,                                                                           \
                  "Block size must also be even to ensure no two threads write to the same byte.");              \
                                                                                                                 \
    const std::ptrdiff_t num_blocks = (N + block_size - 1) / block_size;                                         \
    const TensorOpCost unit_cost{static_cast<double>(block_size * sizeof(float)),                                \
                                 static_cast<double>(block_size * sizeof(INT4_TYPE::UnpackedType)) / 2.0,        \
                                 static_cast<double>(block_size) * 2.0};                                         \
    concurrency::ThreadPool::TryParallelFor(                                                                     \
        thread_pool, num_blocks, unit_cost,                                                                      \
        [&](std::ptrdiff_t begin, std::ptrdiff_t end) {                                                          \
          auto begin_idx = begin * block_size;                                                                   \
          auto end_idx = std::min(static_cast<std::ptrdiff_t>(N), end * block_size);                             \
          auto inp_idx = begin_idx + static_cast<std::ptrdiff_t>(inp_start);                                     \
          auto out_idx = begin_idx + static_cast<std::ptrdiff_t>(out_start);                                     \
                                                                                                                 \
          MLAS_FUNC(&(Input[inp_idx]),                                                                           \
                    reinterpret_cast<uint8_t*>(&(Output[out_idx >> 1])),                                         \
                    end_idx - begin_idx,                                                                         \
                    Scale,                                                                                       \
                    static_cast<int8_t>(ZeroPoint.GetElem(0)));                                                  \
        });                                                                                                      \
  }

DEFINE_PAR_QUANT_LINEAR_STD_4BIT(ParQuantizeLinearStdS4, Int4x2, MlasQuantizeLinearS4)
DEFINE_PAR_QUANT_LINEAR_STD_4BIT(ParQuantizeLinearStdU4, UInt4x2, MlasQuantizeLinearU4)

// This implementation could be more efficient however the cast from float16 to other types
// usually happens on GPU.
template <typename OutputType>
#if !defined(DISABLE_FLOAT8_TYPES)
typename std::enable_if<!boost::mp11::mp_contains<element_type_lists::AllFloat8, OutputType>::value, void>::type
#else
void
#endif
ParQuantizeLinearStd(const MLFloat16* Input,
                     OutputType* Output,
                     size_t N,
                     MLFloat16 Scale,
                     OutputType ZeroPoint,
                     concurrency::ThreadPool* thread_pool) {
  constexpr std::ptrdiff_t block_size = 128;
  const std::ptrdiff_t num_blocks = (N + block_size - 1) / block_size;
  const TensorOpCost unit_cost{static_cast<double>(block_size * sizeof(MLFloat16)), static_cast<double>(block_size * sizeof(uint8_t)), static_cast<double>(block_size) * 2.0};
  concurrency::ThreadPool::TryParallelFor(thread_pool, num_blocks, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    auto begin_idx = begin * block_size;
    auto end_idx = std::min(static_cast<std::ptrdiff_t>(N), end * block_size);
    float fscale = Scale.ToFloat();
    for (; begin_idx != end_idx; ++begin_idx) {
      int32_t ival = static_cast<int32_t>(Input[begin_idx].ToFloat() / fscale) + ZeroPoint;
      Output[begin_idx] = static_cast<OutputType>(std::min(static_cast<int32_t>(std::numeric_limits<OutputType>::max()),
                                                           std::max(static_cast<int32_t>(std::numeric_limits<OutputType>::lowest()), ival)));
    }
  });
}

#if !defined(DISABLE_FLOAT8_TYPES)

template <typename OutputFloat8Type>
typename std::enable_if<boost::mp11::mp_contains<element_type_lists::AllFloat8, OutputFloat8Type>::value, void>::type
ParQuantizeLinearSat(const float* Input,
                     OutputFloat8Type* Output,
                     size_t N,
                     float Scale,
                     const OutputFloat8Type& /* ORT_UNUSED_PARAMETER(ZeroPoint) */,
                     bool saturate,
                     concurrency::ThreadPool* thread_pool) {
  constexpr std::ptrdiff_t block_size = 128;
  const std::ptrdiff_t num_blocks = (N + block_size - 1) / block_size;
  const TensorOpCost unit_cost{static_cast<double>(block_size * sizeof(float)), static_cast<double>(block_size * sizeof(uint8_t)), static_cast<double>(block_size) * 2.0};
  concurrency::ThreadPool::TryParallelFor(thread_pool, num_blocks, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    auto begin_idx = begin * block_size;
    auto end_idx = std::min(static_cast<std::ptrdiff_t>(N), end * block_size);
    for (; begin_idx < end_idx; ++begin_idx) {
      Output[begin_idx] = OutputFloat8Type(Input[begin_idx] / Scale, saturate);
    }
  });
}

// The implementation converts float16 to float and then do a quantization.
// This is not efficient and is mostly added to enable unittest on CPU.
// This case usually happens on GPU.
template <typename OutputFloat8Type>
typename std::enable_if<boost::mp11::mp_contains<element_type_lists::AllFloat8, OutputFloat8Type>::value, void>::type
ParQuantizeLinearSat(const MLFloat16* Input,
                     OutputFloat8Type* Output,
                     size_t N,
                     MLFloat16 Scale,
                     const OutputFloat8Type& /* ORT_UNUSED_PARAMETER(ZeroPoint) */,
                     bool saturate,
                     concurrency::ThreadPool* thread_pool) {
  constexpr std::ptrdiff_t block_size = 128;
  const std::ptrdiff_t num_blocks = (N + block_size - 1) / block_size;
  const TensorOpCost unit_cost{static_cast<double>(block_size * sizeof(MLFloat16)), static_cast<double>(block_size * sizeof(uint8_t)), static_cast<double>(block_size) * 2.0};
  concurrency::ThreadPool::TryParallelFor(thread_pool, num_blocks, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    auto begin_idx = begin * block_size;
    auto end_idx = std::min(static_cast<std::ptrdiff_t>(N), end * block_size);
    for (; begin_idx < end_idx; ++begin_idx) {
      Output[begin_idx] = OutputFloat8Type(Input[begin_idx].ToFloat() / Scale.ToFloat(), saturate);
    }
  });
}

#endif

/**
 * @brief  compute blocked quantization
 *
 * @tparam TIn
 * @tparam TOut
 * @tparam output_type_group        0: int other than int4.
 *                                  1: float8
 *                                  2: int4
 * @method op0                      baseline implementation. Single thread. Scalar instructions.
 * @method op1                      multi-threading implementation. Vector instructions.
 */
template <typename TIn, typename TOut, int output_type_group>
struct BlockedQuantizeLinear {
  /**
   * @brief Compute blocked quantization using multi-threading and vector instructions.
   *        Quantize axis is not the last axis. Block the last axis using thread_block_size.
   *        N is usually large. Within a block, scale's index increments along with output's index.
   *
   * @param thread_pool           thread pool
   * @param input                 input tensor
   * @param scale                 scale tensor
   * @param zero_point            zero point tensor
   * @param output                output tensor
   * @param M                     total size of dimensions before quantize axis
   * @param K                     size of dimension on quantize axis
   * @param N                     total size of dimensions after quantize axis
   * @param quant_block_size      quantization block size
   * @param thread_block_size     task block size
   * @param saturate              used by float8
   */
  static void opNotLastAxis(concurrency::ThreadPool* thread_pool, const float* input, const float* scale,
                            const TOut* zero_point, TOut* output, std::ptrdiff_t M, std::ptrdiff_t K,
                            std::ptrdiff_t N, const std::ptrdiff_t quant_block_size,
                            const std::ptrdiff_t thread_block_size, bool saturate);

  /**
   * @brief Compute blocked quantization using multi-threading and vector instructions.
   *        Quantize axis is the last axis. Block along quantize axis using quant_block_size
   *        as thread_block_size. quant_block_size is usually 2's power between 16 and 256.
   *        Within a block, scale index does not change.
   *
   * @param thread_pool           thread pool
   * @param input                 input tensor
   * @param scale                 scale tensor
   * @param zero_point            zero point tensor
   * @param output                output tensor
   * @param M                     total size of dimensions before quantize axis
   * @param K                     size of dimension on quantize axis
   * @param quant_block_size      quantization block size
   * @param saturate              used by float8
   */
  static void opLastAxis(concurrency::ThreadPool* thread_pool, const float* input, const float* scale,
                         const TOut* zero_point, TOut* output, std::ptrdiff_t M, std::ptrdiff_t K,
                         const std::ptrdiff_t quant_block_size, bool saturate);
};

template <typename TOut>
struct BlockedQuantizeLinear<float, TOut, 0> {
  static void opNotLastAxis(concurrency::ThreadPool* thread_pool, const float* input, const float* scale,
                            const TOut* zero_point, TOut* output, std::ptrdiff_t M, std::ptrdiff_t K,
                            std::ptrdiff_t N, const std::ptrdiff_t quant_block_size,
                            const std::ptrdiff_t thread_block_size, bool saturate) {
    ORT_UNUSED_PARAMETER(saturate);
    constexpr auto low = static_cast<int32_t>(std::numeric_limits<TOut>::lowest());
    constexpr auto high = static_cast<int32_t>(std::numeric_limits<TOut>::max());
    const auto num_thread_block_N = (N + thread_block_size - 1) / thread_block_size;
    const auto num_thread_block = M * K * num_thread_block_N;
    const TensorOpCost unit_cost{static_cast<double>(thread_block_size * sizeof(float) * 2),
                                 static_cast<double>(thread_block_size * sizeof(TOut)),
                                 static_cast<double>(thread_block_size) * 2.0};
    auto KN = K * N;
    auto num_quant_block_KN = (K + quant_block_size - 1) / quant_block_size * N;
    const auto num_thread_block_KN = K * num_thread_block_N;

    concurrency::ThreadPool::TryParallelFor(
        thread_pool,
        num_thread_block,
        unit_cost,
        [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
          auto m = begin / num_thread_block_KN, k = begin % num_thread_block_KN / num_thread_block_N;
          auto n_blk = begin % num_thread_block_N, n = n_blk * thread_block_size;
          auto output_idx = m * KN + k * N + n;
          auto quant_param_idx = m * num_quant_block_KN + k / quant_block_size * N;
          auto quant_param_idx_t = quant_param_idx + n;

          for (; begin < end; ++begin) {
            auto n_end = std::min(N, n + thread_block_size);
            // TODO(fajin): 1> use SIMD, 2> set block to quant_block_size * thread_block_size
            for (; n < n_end; ++n, ++output_idx, ++quant_param_idx_t) {
              // TODO(fajin): perf difference
              auto zp = zero_point ? static_cast<int32_t>(zero_point[quant_param_idx_t]) : 0;
              auto sc = scale[quant_param_idx_t];
              auto v = std::clamp(static_cast<int32_t>(std::nearbyint(input[output_idx] / sc)) + zp, low, high);
              output[output_idx] = static_cast<TOut>(v);
            }

            // A simpler approach is to calculate m, k, n in every block.
            // This approach tries to reduce division and modulo in the block loop. Not benchmarked.
            if (n == N) {
              n = 0;
              ++k;
              if (k == K) {
                k = 0;
                quant_param_idx += N;
              } else if (k % quant_block_size == 0) {
                quant_param_idx += N;
              }

              quant_param_idx_t = quant_param_idx;
            }
          }
        });
  }

  static void opLastAxis(concurrency::ThreadPool* thread_pool, const float* input, const float* scale,
                         const TOut* zero_point, TOut* output, std::ptrdiff_t M, std::ptrdiff_t K,
                         const std::ptrdiff_t quant_block_size, bool saturate) {
    ORT_UNUSED_PARAMETER(saturate);
    // quant block size is used as thread block size
    const auto num_thread_block_K = (K + quant_block_size - 1) / quant_block_size;
    const auto num_thread_block = num_thread_block_K * M;
    const TensorOpCost unit_cost{static_cast<double>(quant_block_size * sizeof(float)),
                                 static_cast<double>(quant_block_size * sizeof(TOut)),
                                 static_cast<double>(quant_block_size) * 2.0};
    concurrency::ThreadPool::TryParallelFor(
        thread_pool,
        num_thread_block,
        unit_cost,
        [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
          auto m = begin / num_thread_block_K, k_blk = begin % num_thread_block_K, k = k_blk * quant_block_size;
          auto output_idx = m * K + k;

          for (; begin < end; ++begin) {
            // each thread block is also a quantization block
            auto zp = zero_point ? zero_point[begin] : static_cast<TOut>(0);
            auto sc = scale[begin];
            size_t output_size = std::min(K - k, quant_block_size);
            MlasQuantizeLinear(input + output_idx, output + output_idx, output_size, sc, zp);
            output_idx += output_size;
            k = output_idx % K;
          }
        });
  }
};

template <typename TOut>
struct BlockedQuantizeLinear<MLFloat16, TOut, 0> {
  static void opNotLastAxis(concurrency::ThreadPool* thread_pool, const MLFloat16* input, const MLFloat16* scale,
                            const TOut* zero_point, TOut* output, std::ptrdiff_t M, std::ptrdiff_t K,
                            std::ptrdiff_t N, const std::ptrdiff_t quant_block_size,
                            const std::ptrdiff_t thread_block_size, bool saturate) {
    ORT_UNUSED_PARAMETER(saturate);
    constexpr auto low = static_cast<int32_t>(std::numeric_limits<TOut>::lowest());
    constexpr auto high = static_cast<int32_t>(std::numeric_limits<TOut>::max());
    const auto num_thread_block_N = (N + thread_block_size - 1) / thread_block_size;
    const auto num_thread_block = M * K * num_thread_block_N;
    const TensorOpCost unit_cost{static_cast<double>(thread_block_size * sizeof(MLFloat16) * 2),
                                 static_cast<double>(thread_block_size * sizeof(TOut)),
                                 static_cast<double>(thread_block_size) * 2.0};
    auto KN = K * N;
    auto num_quant_block_KN = (K + quant_block_size - 1) / quant_block_size * N;
    const auto num_thread_block_KN = K * num_thread_block_N;

    concurrency::ThreadPool::TryParallelFor(
        thread_pool,
        num_thread_block,
        unit_cost,
        [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
          auto m = begin / num_thread_block_KN, k = begin % num_thread_block_KN / num_thread_block_N;
          auto n_blk = begin % num_thread_block_N, n = n_blk * thread_block_size;
          auto output_idx = m * KN + k * N + n;
          auto quant_param_idx = m * num_quant_block_KN + k / quant_block_size * N;
          auto quant_param_idx_t = quant_param_idx + n;

          for (; begin < end; ++begin) {
            auto n_end = std::min(N, n + thread_block_size);
            // TODO(fajin): 1> use SIMD, 2> set block to quant_block_size * thread_block_size
            for (; n < n_end; ++n, ++output_idx, ++quant_param_idx_t) {
              // TODO(fajin): perf difference
              auto zp = zero_point ? static_cast<int32_t>(zero_point[quant_param_idx_t]) : 0;
              auto sc = scale[quant_param_idx_t].ToFloat();
              auto v = std::clamp(static_cast<int32_t>(std::nearbyint(input[output_idx].ToFloat() / sc)) + zp,
                                  low, high);
              output[output_idx] = static_cast<TOut>(v);
            }

            if (n == N) {
              n = 0;
              ++k;
              if (k == K) {
                k = 0;
                quant_param_idx += N;
              } else if (k % quant_block_size == 0) {
                quant_param_idx += N;
              }

              quant_param_idx_t = quant_param_idx;
            }
          }
        });
  }

  static void opLastAxis(concurrency::ThreadPool* thread_pool, const MLFloat16* input, const MLFloat16* scale,
                         const TOut* zero_point, TOut* output, std::ptrdiff_t M, std::ptrdiff_t K,
                         const std::ptrdiff_t quant_block_size, bool saturate) {
    ORT_UNUSED_PARAMETER(saturate);
    constexpr auto low = static_cast<int32_t>(std::numeric_limits<TOut>::lowest());
    constexpr auto high = static_cast<int32_t>(std::numeric_limits<TOut>::max());
    // quant block size is used as thread block size
    const auto num_thread_block_K = (K + quant_block_size - 1) / quant_block_size;
    const auto num_thread_block = num_thread_block_K * M;
    const TensorOpCost unit_cost{static_cast<double>(quant_block_size * sizeof(MLFloat16)),
                                 static_cast<double>(quant_block_size * sizeof(TOut)),
                                 static_cast<double>(quant_block_size) * 2.0};
    concurrency::ThreadPool::TryParallelFor(
        thread_pool,
        num_thread_block,
        unit_cost,
        [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
          auto m = begin / num_thread_block_K, k_blk = begin % num_thread_block_K, k = k_blk * quant_block_size;
          auto output_idx = m * K + k;

          for (; begin < end; ++begin) {
            // each thread block is also a quantization block
            auto zp = zero_point ? static_cast<int32_t>(zero_point[begin]) : 0;
            auto sc = scale[begin].ToFloat();
            auto output_idx_end = std::min(K - k, quant_block_size) + output_idx;
            for (; output_idx < output_idx_end; ++output_idx) {
              auto v = std::clamp(static_cast<int32_t>(std::nearbyint(input[output_idx].ToFloat() / sc)) + zp,
                                  low, high);
              output[output_idx] = static_cast<TOut>(v);
            }
            k = output_idx % K;
          }
        });
  }
};

template <typename TOut>
struct BlockedQuantizeLinear<float, TOut, 2> {
  static void opNotLastAxis(concurrency::ThreadPool* thread_pool, const float* input, const float* scale,
                            const TOut* zero_point, TOut* output, std::ptrdiff_t M, std::ptrdiff_t K,
                            std::ptrdiff_t N, const std::ptrdiff_t quant_block_size,
                            const std::ptrdiff_t thread_block_size, bool saturate) {
    ORT_UNUSED_PARAMETER(saturate);
    // to avoid a byte being written from mutiple threads, use 2 * N as thread block
    ORT_UNUSED_PARAMETER(thread_block_size);
    constexpr auto low = static_cast<int32_t>(TOut::min_val);
    constexpr auto high = static_cast<int32_t>(TOut::max_val);
    auto size_thread_block = 2 * N;
    auto num_thread_block = (M * K + 1) / 2;
    auto num_quant_block_K = (K + quant_block_size - 1) / quant_block_size;
    auto num_quant_block_KN = num_quant_block_K * N;
    auto MK = M * K;
    const TensorOpCost unit_cost{static_cast<double>(size_thread_block * sizeof(float) * 2),
                                 static_cast<double>(size_thread_block * sizeof(typename TOut::UnpackedType)),
                                 static_cast<double>(size_thread_block) * 2.0};

    concurrency::ThreadPool::TryParallelFor(
        thread_pool,
        num_thread_block,
        unit_cost,
        [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
          begin <<= 1, end = std::min(end << 1, MK);
          auto output_idx = begin * N;
          auto m = begin / K, k = begin % K;
          auto zp_idx = m * num_quant_block_KN + k / quant_block_size * N;

          for (; begin < end; ++begin) {
            auto zp_idx_t = zp_idx;
            auto output_idx_end = output_idx + N;

            // leading unaligned output
            if (output_idx & 1) {
              auto zp = zero_point
                            ? static_cast<int32_t>(zero_point[zp_idx_t >> 1].GetElem(zp_idx_t & 1))
                            : 0;
              auto sc = scale[zp_idx_t];
              auto v = std::clamp(static_cast<int32_t>(std::nearbyint(input[output_idx] / sc)) + zp, low, high);
              output[output_idx >> 1].SetElem(1, static_cast<typename TOut::UnpackedType>(v));
              ++output_idx;
              ++zp_idx_t;
            }

            // TODO(fajin): use SIMD
            // aligned output
            auto output_t = reinterpret_cast<typename TOut::UnpackedType*>(output);
            for (; output_idx < output_idx_end - 1; output_idx += 2, zp_idx_t += 2) {
              auto zp0 = zero_point
                             ? static_cast<int32_t>(zero_point[zp_idx_t >> 1].GetElem(zp_idx_t & 1))
                             : 0;
              auto zp1 = zero_point
                             ? static_cast<int32_t>(zero_point[(zp_idx_t + 1) >> 1].GetElem((zp_idx_t + 1) & 1))
                             : 0;
              auto sc0 = scale[zp_idx_t];
              auto sc1 = scale[zp_idx_t + 1];
              auto v0 = std::clamp(static_cast<int32_t>(std::nearbyint(input[output_idx] / sc0)) + zp0, low, high);
              auto v1 = std::clamp(static_cast<int32_t>(std::nearbyint(input[output_idx + 1] / sc1)) + zp1, low, high);
              output_t[output_idx >> 1] = static_cast<typename TOut::UnpackedType>((v0 & 0xF) | ((v1 & 0xF) << 4));
            }

            // tailing unaligned output
            if (output_idx < output_idx_end) {
              auto zp = zero_point
                            ? static_cast<int32_t>(zero_point[zp_idx_t >> 1].GetElem(zp_idx_t & 1))
                            : 0;
              auto sc = scale[zp_idx_t];
              auto v = std::clamp(static_cast<int32_t>(std::nearbyint(input[output_idx] / sc)) + zp, low, high);
              output[output_idx >> 1].SetElem(0, static_cast<typename TOut::UnpackedType>(v));

              ++output_idx;
            }

            ++k;
            if (k == K) {
              k = 0;
              zp_idx += N;
            } else if (k % quant_block_size == 0) {
              zp_idx += N;
            }
          }
        });
  }

  static void opLastAxis(concurrency::ThreadPool* thread_pool, const float* input, const float* scale,
                         const TOut* zero_point, TOut* output, std::ptrdiff_t M, std::ptrdiff_t K,
                         const std::ptrdiff_t quant_block_size, bool saturate) {
    ORT_UNUSED_PARAMETER(saturate);
    constexpr auto low = static_cast<int32_t>(TOut::min_val);
    constexpr auto high = static_cast<int32_t>(TOut::max_val);
    // to avoid a byte being written from mutiple threads, use 2 * K as thread block
    auto size_thread_block = 2 * K;
    auto quant_block_num_K = (K + quant_block_size - 1) / quant_block_size;
    auto num_thread_block = (M + 1) / 2;
    TensorOpCost unit_cost{static_cast<double>(size_thread_block * sizeof(float)),
                           static_cast<double>(size_thread_block * sizeof(typename TOut ::UnpackedType)),
                           static_cast<double>(size_thread_block) * 2.0};
    concurrency::ThreadPool::TryParallelFor(
        thread_pool,
        num_thread_block,
        unit_cost,
        [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
          begin <<= 1, end = std::min(end << 1, M);
          auto output_idx = begin * K;
          auto zp_idx = begin * quant_block_num_K;

          for (; begin < end; ++begin, output_idx += K) {
            auto output_row_idx_start = output_idx;
            auto output_row_idx_end = output_row_idx_start + K;

            for (; output_row_idx_start < output_row_idx_end; output_row_idx_start += quant_block_size, ++zp_idx) {
              auto zp = zero_point ? static_cast<int32_t>(zero_point[zp_idx >> 1].GetElem(zp_idx & 1)) : 0;
              auto sc = scale[zp_idx];
              size_t out_start = output_row_idx_start;
              size_t out_end = std::min(output_row_idx_start + quant_block_size, output_row_idx_end);

              if (out_start & 1) {
                auto v = std::clamp(static_cast<int32_t>(std::nearbyint(input[out_start] / sc)) + zp, low, high);
                output[out_start >> 1].SetElem(1, static_cast<typename TOut::UnpackedType>(v));
                ++out_start;
              }

              if (out_end & 1) {
                --out_end;
                auto v = std::clamp(static_cast<int32_t>(std::nearbyint(input[out_end] / sc)) + zp, low, high);
                output[out_end >> 1].SetElem(0, static_cast<typename TOut::UnpackedType>(v));
              }

              if constexpr (std::is_same<TOut, Int4x2>::value) {
                MlasQuantizeLinearS4(input + out_start, reinterpret_cast<uint8_t*>(&(output[out_start >> 1])),
                                     out_end - out_start, sc, static_cast<int8_t>(zp));
              } else {
                MlasQuantizeLinearU4(input + out_start, reinterpret_cast<uint8_t*>(&(output[out_start >> 1])),
                                     out_end - out_start, sc, static_cast<int8_t>(zp));
              }
            }
          }
        });
  }
};

// Bug(fajin): the same byte in output / zero_point must not be written by different threads, otherwise
// the result is undefined. This is not handled in the current implementation.
template <typename TOut>
struct BlockedQuantizeLinear<MLFloat16, TOut, 2> {
  static void opNotLastAxis(concurrency::ThreadPool* thread_pool, const MLFloat16* input, const MLFloat16* scale,
                            const TOut* zero_point, TOut* output, std::ptrdiff_t M, std::ptrdiff_t K,
                            std::ptrdiff_t N, const std::ptrdiff_t quant_block_size,
                            const std::ptrdiff_t thread_block_size, bool saturate) {
    ORT_UNUSED_PARAMETER(saturate);
    // to avoid a byte being written from mutiple threads, use 2 * N as thread block
    ORT_UNUSED_PARAMETER(thread_block_size);
    constexpr auto low = static_cast<int32_t>(TOut::min_val);
    constexpr auto high = static_cast<int32_t>(TOut::max_val);
    auto size_thread_block = 2 * N;
    auto num_thread_block = (M * K + 1) / 2;
    auto num_quant_block_K = (K + quant_block_size - 1) / quant_block_size;
    auto num_quant_block_KN = num_quant_block_K * N;
    auto MK = M * K;
    const TensorOpCost unit_cost{static_cast<double>(size_thread_block * sizeof(float) * 2),
                                 static_cast<double>(size_thread_block * sizeof(typename TOut::UnpackedType)),
                                 static_cast<double>(size_thread_block) * 2.0};

    concurrency::ThreadPool::TryParallelFor(
        thread_pool,
        num_thread_block,
        unit_cost,
        [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
          begin <<= 1, end = std::min(end << 1, MK);
          auto output_idx = begin * N;
          auto m = begin / K, k = begin % K;
          auto zp_idx = m * num_quant_block_KN + k / quant_block_size * N;

          for (; begin < end; ++begin) {
            auto zp_idx_t = zp_idx;
            auto output_idx_end = output_idx + N;

            // leading unaligned output
            if (output_idx & 1) {
              auto zp = zero_point
                            ? static_cast<int32_t>(zero_point[zp_idx_t >> 1].GetElem(zp_idx_t & 1))
                            : 0;
              auto sc = scale[zp_idx_t].ToFloat();
              auto v = std::clamp(
                  static_cast<int32_t>(std::nearbyint(input[output_idx].ToFloat() / sc)) + zp, low, high);
              output[output_idx >> 1].SetElem(1, static_cast<typename TOut::UnpackedType>(v));
              ++output_idx;
              ++zp_idx_t;
            }

            // TODO(fajin): use SIMD
            // aligned output
            auto output_t = reinterpret_cast<typename TOut::UnpackedType*>(output);
            for (; output_idx < output_idx_end - 1; output_idx += 2, zp_idx_t += 2) {
              auto zp0 = zero_point
                             ? static_cast<int32_t>(zero_point[zp_idx_t >> 1].GetElem(zp_idx_t & 1))
                             : 0;
              auto zp1 = zero_point
                             ? static_cast<int32_t>(zero_point[(zp_idx_t + 1) >> 1].GetElem((zp_idx_t + 1) & 1))
                             : 0;
              auto sc0 = scale[zp_idx_t].ToFloat();
              auto sc1 = scale[zp_idx_t + 1].ToFloat();
              auto v0 = std::clamp(
                  static_cast<int32_t>(std::nearbyint(input[output_idx].ToFloat() / sc0)) + zp0, low, high);
              auto v1 = std::clamp(
                  static_cast<int32_t>(std::nearbyint(input[output_idx + 1].ToFloat() / sc1)) + zp1, low, high);
              output_t[output_idx >> 1] = static_cast<typename TOut::UnpackedType>((v0 & 0xF) | ((v1 & 0xF) << 4));
            }

            // tailing unaligned output
            if (output_idx < output_idx_end) {
              auto zp = zero_point
                            ? static_cast<int32_t>(zero_point[zp_idx_t >> 1].GetElem(zp_idx_t & 1))
                            : 0;
              auto sc = scale[zp_idx_t].ToFloat();
              auto v = std::clamp(
                  static_cast<int32_t>(std::nearbyint(input[output_idx].ToFloat() / sc)) + zp, low, high);
              output[output_idx >> 1].SetElem(0, static_cast<typename TOut::UnpackedType>(v));

              ++output_idx;
            }

            ++k;
            if (k == K) {
              k = 0;
              zp_idx += N;
            } else if (k % quant_block_size == 0) {
              zp_idx += N;
            }
          }
        });
  }

  static void opLastAxis(concurrency::ThreadPool* thread_pool, const MLFloat16* input, const MLFloat16* scale,
                         const TOut* zero_point, TOut* output, std::ptrdiff_t M, std::ptrdiff_t K,
                         const std::ptrdiff_t quant_block_size, bool saturate) {
    ORT_UNUSED_PARAMETER(saturate);
    constexpr auto low = static_cast<int32_t>(TOut::min_val);
    constexpr auto high = static_cast<int32_t>(TOut::max_val);
    // to avoid a byte being written from mutiple threads, use 2 * K as thread block
    auto size_thread_block = 2 * K;
    auto quant_block_num_K = (K + quant_block_size - 1) / quant_block_size;
    auto num_thread_block = (M + 1) / 2;
    TensorOpCost unit_cost{static_cast<double>(size_thread_block * sizeof(float)),
                           static_cast<double>(size_thread_block * sizeof(typename TOut ::UnpackedType)),
                           static_cast<double>(size_thread_block) * 2.0};
    concurrency::ThreadPool::TryParallelFor(
        thread_pool,
        num_thread_block,
        unit_cost,
        [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
          begin <<= 1, end = std::min(end << 1, M);
          auto output_idx = begin * K;
          auto zp_idx = begin * quant_block_num_K;

          for (; begin < end; ++begin, output_idx += K) {
            auto output_row_idx_start = output_idx;
            auto output_row_idx_end = output_row_idx_start + K;

            for (; output_row_idx_start < output_row_idx_end; output_row_idx_start += quant_block_size, ++zp_idx) {
              auto zp = zero_point ? static_cast<int32_t>(zero_point[zp_idx >> 1].GetElem(zp_idx & 1)) : 0;
              auto sc = scale[zp_idx].ToFloat();
              size_t out_start = output_row_idx_start;
              size_t out_end = std::min(output_row_idx_start + quant_block_size, output_row_idx_end);

              if (out_start & 1) {
                auto v = std::clamp(
                    static_cast<int32_t>(std::nearbyint(input[out_start].ToFloat() / sc)) + zp, low, high);
                output[out_start >> 1].SetElem(1, static_cast<typename TOut::UnpackedType>(v));
                ++out_start;
              }

              if (out_end & 1) {
                --out_end;
                auto v = std::clamp(
                    static_cast<int32_t>(std::nearbyint(input[out_end].ToFloat() / sc)) + zp, low, high);
                output[out_end >> 1].SetElem(0, static_cast<typename TOut::UnpackedType>(v));
              }

              auto output_t = reinterpret_cast<typename TOut::UnpackedType*>(output);
              for (; out_start < out_end; out_start += 2) {
                auto v0 = std::clamp(
                    static_cast<int32_t>(std::nearbyint(input[out_start].ToFloat() / sc)) + zp, low, high);
                auto v1 = std::clamp(
                    static_cast<int32_t>(std::nearbyint(input[out_start + 1].ToFloat() / sc)) + zp, low, high);
                output_t[out_start >> 1] = static_cast<typename TOut::UnpackedType>((v0 & 0xF) | ((v1 & 0xF) << 4));
              }
            }
          }
        });
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)

template <typename TOut>
struct BlockedQuantizeLinear<float, TOut, 1> {
  static void opNotLastAxis(concurrency::ThreadPool* thread_pool, const float* input, const float* scale,
                            const TOut* zero_point, TOut* output, std::ptrdiff_t M, std::ptrdiff_t K,
                            std::ptrdiff_t N, const std::ptrdiff_t quant_block_size,
                            const std::ptrdiff_t thread_block_size, bool saturate) {
    ORT_UNUSED_PARAMETER(zero_point);
    const auto num_thread_block_N = (N + thread_block_size - 1) / thread_block_size;
    const auto num_thread_block = M * K * num_thread_block_N;
    const TensorOpCost unit_cost{static_cast<double>(thread_block_size * sizeof(float) * 2),
                                 static_cast<double>(thread_block_size * sizeof(uint8_t)),
                                 static_cast<double>(thread_block_size) * 2.0};
    auto KN = K * N;
    auto num_quant_block_KN = (K + quant_block_size - 1) / quant_block_size * N;
    const auto num_thread_block_KN = K * num_thread_block_N;

    concurrency::ThreadPool::TryParallelFor(
        thread_pool,
        num_thread_block,
        unit_cost,
        [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
          auto m = begin / num_thread_block_KN, k = begin % num_thread_block_KN / num_thread_block_N;
          auto n_blk = begin % num_thread_block_N, n = n_blk * thread_block_size;
          auto output_idx = m * KN + k * N + n;
          auto quant_param_idx = m * num_quant_block_KN + k / quant_block_size * N;
          auto quant_param_idx_t = quant_param_idx + n;

          for (; begin < end; ++begin) {
            auto n_end = std::min(N, n + thread_block_size);
            for (; n < n_end; ++n, ++output_idx, ++quant_param_idx_t) {
              output[output_idx] = TOut(input[output_idx] / scale[quant_param_idx_t], saturate);
            }

            if (n == N) {
              n = 0;
              ++k;
              if (k == K) {
                k = 0;
                quant_param_idx += N;
              } else if (k % quant_block_size == 0) {
                quant_param_idx += N;
              }

              quant_param_idx_t = quant_param_idx;
            }
          }
        });
  }

  static void opLastAxis(concurrency::ThreadPool* thread_pool, const float* input, const float* scale,
                         const TOut* zero_point, TOut* output, std::ptrdiff_t M, std::ptrdiff_t K,
                         const std::ptrdiff_t quant_block_size, bool saturate) {
    ORT_UNUSED_PARAMETER(zero_point);
    // quant block size is used as thread block size
    const auto num_thread_block_K = (K + quant_block_size - 1) / quant_block_size;
    const auto num_thread_block = num_thread_block_K * M;
    const TensorOpCost unit_cost{static_cast<double>(quant_block_size * sizeof(float)),
                                 static_cast<double>(quant_block_size * sizeof(uint8_t)),
                                 static_cast<double>(quant_block_size) * 2.0};
    concurrency::ThreadPool::TryParallelFor(
        thread_pool,
        num_thread_block,
        unit_cost,
        [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
          auto m = begin / num_thread_block_K, k_blk = begin % num_thread_block_K, k = k_blk * quant_block_size;
          auto output_idx = m * K + k;

          for (; begin < end; ++begin) {
            auto sc = scale[begin];
            auto output_idx_end = std::min(K - k, quant_block_size) + output_idx;
            for (; output_idx < output_idx_end; ++output_idx) {
              output[output_idx] = TOut(input[output_idx] / sc, saturate);
            }
            k = output_idx % K;
          }
        });
  }
};

template <typename TOut>
struct BlockedQuantizeLinear<MLFloat16, TOut, 1> {
  static void opNotLastAxis(concurrency::ThreadPool* thread_pool, const MLFloat16* input, const MLFloat16* scale,
                            const TOut* zero_point, TOut* output, std::ptrdiff_t M, std::ptrdiff_t K,
                            std::ptrdiff_t N, const std::ptrdiff_t quant_block_size,
                            const std::ptrdiff_t thread_block_size, bool saturate) {
    ORT_UNUSED_PARAMETER(zero_point);
    const auto num_thread_block_N = (N + thread_block_size - 1) / thread_block_size;
    const auto num_thread_block = M * K * num_thread_block_N;
    const TensorOpCost unit_cost{static_cast<double>(thread_block_size * sizeof(MLFloat16) * 2),
                                 static_cast<double>(thread_block_size * sizeof(uint8_t)),
                                 static_cast<double>(thread_block_size) * 2.0};
    auto KN = K * N;
    auto num_quant_block_KN = (K + quant_block_size - 1) / quant_block_size * N;
    const auto num_thread_block_KN = K * num_thread_block_N;

    concurrency::ThreadPool::TryParallelFor(
        thread_pool,
        num_thread_block,
        unit_cost,
        [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
          auto m = begin / num_thread_block_KN, k = begin % num_thread_block_KN / num_thread_block_N;
          auto n_blk = begin % num_thread_block_N, n = n_blk * thread_block_size;
          auto output_idx = m * KN + k * N + n;
          auto quant_param_idx = m * num_quant_block_KN + k / quant_block_size * N;
          auto quant_param_idx_t = quant_param_idx + n;

          for (; begin < end; ++begin) {
            auto n_end = std::min(N, n + thread_block_size);
            for (; n < n_end; ++n, ++output_idx, ++quant_param_idx_t) {
              output[output_idx] = TOut(input[output_idx].ToFloat() / scale[quant_param_idx_t].ToFloat(), saturate);
            }

            if (n == N) {
              n = 0;
              ++k;
              if (k == K) {
                k = 0;
                quant_param_idx += N;
              } else if (k % quant_block_size == 0) {
                quant_param_idx += N;
              }

              quant_param_idx_t = quant_param_idx;
            }
          }
        });
  }

  static void opLastAxis(concurrency::ThreadPool* thread_pool, const MLFloat16* input, const MLFloat16* scale,
                         const TOut* zero_point, TOut* output, std::ptrdiff_t M, std::ptrdiff_t K,
                         const std::ptrdiff_t quant_block_size, bool saturate) {
    ORT_UNUSED_PARAMETER(zero_point);
    const auto num_thread_block_K = (K + quant_block_size - 1) / quant_block_size;
    const auto num_thread_block = num_thread_block_K * M;
    const TensorOpCost unit_cost{static_cast<double>(quant_block_size * sizeof(MLFloat16)),
                                 static_cast<double>(quant_block_size * sizeof(uint8_t)),
                                 static_cast<double>(quant_block_size) * 2.0};
    concurrency::ThreadPool::TryParallelFor(
        thread_pool,
        num_thread_block,
        unit_cost,
        [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
          auto m = begin / num_thread_block_K, k_blk = begin % num_thread_block_K, k = k_blk * quant_block_size;
          auto output_idx = m * K + k;

          for (; begin < end; ++begin) {
            auto sc = scale[begin].ToFloat();
            auto output_idx_end = std::min(K - k, quant_block_size) + output_idx;
            for (; output_idx < output_idx_end; ++output_idx) {
              output[output_idx] = TOut(input[output_idx].ToFloat() / sc, saturate);
            }
            k = output_idx % K;
          }
        });
  }
};

#endif

/**
 * @brief Run MlasDequantizeLinear in parallel, with provided thread pool
 */

template <typename InputQuantType>
void ParDequantizeLinearStd(const InputQuantType* input,
                            float* output,
                            size_t num_elems,
                            float scale,
                            InputQuantType zero_point,
                            concurrency::ThreadPool* thread_pool) {
  constexpr std::ptrdiff_t block_size = 128;
  const std::ptrdiff_t num_blocks = (num_elems + block_size - 1) / block_size;
  const TensorOpCost unit_cost{static_cast<double>(block_size * sizeof(InputQuantType)),
                               static_cast<double>(block_size * sizeof(float)),
                               static_cast<double>(block_size) * 2.0};
  concurrency::ThreadPool::TryParallelFor(thread_pool, num_blocks, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    auto begin_idx = begin * block_size;
    auto end_idx = std::min(static_cast<std::ptrdiff_t>(num_elems), end * block_size);
    MlasDequantizeLinear(&(input[begin_idx]), &(output[begin_idx]), end_idx - begin_idx, scale, zero_point);
  });
}

// Note: this doesn't use MLAS kernel. There are currently no MLAS kernels for fp16 QuantizeLinear or DequantizeLinear.
template <typename InputQuantType>
void ParDequantizeLinearStd(const InputQuantType* input,
                            MLFloat16* output,
                            size_t num_elems,
                            MLFloat16 scale,
                            InputQuantType zero_point,
                            concurrency::ThreadPool* thread_pool) {
  constexpr std::ptrdiff_t block_size = 128;
  const std::ptrdiff_t num_blocks = (num_elems + block_size - 1) / block_size;
  const TensorOpCost unit_cost{static_cast<double>(block_size * sizeof(InputQuantType)),
                               static_cast<double>(block_size * sizeof(MLFloat16)),
                               static_cast<double>(block_size) * 2.0};

  const int32_t zp_s32 = static_cast<int32_t>(zero_point);
  const float sc_f32 = scale.ToFloat();

  concurrency::ThreadPool::TryParallelFor(thread_pool, num_blocks, unit_cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    auto begin_idx = begin * block_size;
    auto end_idx = std::min(static_cast<std::ptrdiff_t>(num_elems), end * block_size);
    for (; begin_idx != end_idx; ++begin_idx) {
      output[begin_idx] = MLFloat16(static_cast<float>(static_cast<int32_t>(input[begin_idx]) - zp_s32) * sc_f32);
    }
  });
}

}  // namespace onnxruntime
