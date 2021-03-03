// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/variadic_elementwise_ops_impl.h"

#include "core/providers/cuda/cu_inc/variadic_elementwise_impl.cuh"
#include "core/providers/cuda/math/binary_elementwise_ops_impl.h"
#include "core/providers/cuda/math/binary_elementwise_ops_impl_functors.cuh"
#include "core/providers/cuda/math/variadic_elementwise_ops_tags.h"

namespace onnxruntime {
namespace cuda {

template <typename T, typename VariadicElementwiseOpTag>
struct VariadicElementwiseOpTraits;

#define DEFINE_TRAITS(VariadicElementwiseOpTag, ImplName)           \
  template <typename T>                                             \
  struct VariadicElementwiseOpTraits<T, VariadicElementwiseOpTag> { \
    using ScalarComputeFunctor = OP_##ImplName<T, T, T>;            \
                                                                    \
    static void ComputeFn(                                          \
        cudaStream_t stream,                                        \
        int32_t output_rank_or_simple_broadcast,                    \
        const TArray<int64_t>* lhs_padded_strides,                  \
        const T* lhs_data,                                          \
        const TArray<int64_t>* rhs_padded_strides,                  \
        const T* rhs_data,                                          \
        const TArray<fast_divmod>* fdm_output_strides,              \
        const fast_divmod& fdm_H,                                   \
        const fast_divmod& fdm_C,                                   \
        T* output_data,                                             \
        size_t count) {                                             \
      Impl_##ImplName(                                              \
          stream,                                                   \
          output_rank_or_simple_broadcast,                          \
          lhs_padded_strides,                                       \
          lhs_data,                                                 \
          rhs_padded_strides,                                       \
          rhs_data,                                                 \
          fdm_output_strides,                                       \
          fdm_H,                                                    \
          fdm_C,                                                    \
          output_data,                                              \
          count);                                                   \
    }                                                               \
  };

DEFINE_TRAITS(variadic_elementwise_ops::Sum, Add)
DEFINE_TRAITS(variadic_elementwise_ops::Min, Min)
DEFINE_TRAITS(variadic_elementwise_ops::Max, Max)

#undef DEFINE_TRAITS

template <typename T, typename VariadicElementwiseOpTag>
void Impl_General(
    cudaStream_t stream,
    int32_t output_rank_or_simple_broadcast,
    const TArray<int64_t>* lhs_padded_strides,
    const T* lhs_data,
    const TArray<int64_t>* rhs_padded_strides,
    const T* rhs_data,
    const TArray<fast_divmod>* fdm_output_strides,
    const fast_divmod& fdm_H,
    const fast_divmod& fdm_C,
    T* output_data,
    size_t count) {
  VariadicElementwiseOpTraits<T, VariadicElementwiseOpTag>::ComputeFn(
      stream,
      output_rank_or_simple_broadcast,
      lhs_padded_strides,
      lhs_data,
      rhs_padded_strides,
      rhs_data,
      fdm_output_strides,
      fdm_H,
      fdm_C,
      output_data,
      count);
}

template <typename T, typename VariadicElementwiseOpTag>
void Impl_NoBroadcastInputBatch(
    cudaStream_t stream,
    InputBatchArray<T> input_data_batch,
    T* output_data,
    size_t count) {
  VariadicElementWiseNoBroadcastInputBatchImpl<
      T, typename VariadicElementwiseOpTraits<T, VariadicElementwiseOpTag>::ScalarComputeFunctor,
      k_max_input_batch_size>(
      stream,
      typename VariadicElementwiseOpTraits<T, VariadicElementwiseOpTag>::ScalarComputeFunctor{},
      count,
      input_data_batch,
      output_data);
}

#define SPECIALIZE_IMPL(T, VariadicElementwiseOpTag)                     \
  template void Impl_General<T, VariadicElementwiseOpTag>(               \
      cudaStream_t stream,                                               \
      int32_t output_rank_or_simple_broadcast,                           \
      const TArray<int64_t>* lhs_padded_strides,                         \
      const T* lhs_data,                                                 \
      const TArray<int64_t>* rhs_padded_strides,                         \
      const T* rhs_data,                                                 \
      const TArray<fast_divmod>* fdm_output_strides,                     \
      const fast_divmod& fdm_H,                                          \
      const fast_divmod& fdm_C,                                          \
      T* output_data,                                                    \
      size_t count);                                                     \
                                                                         \
  template void Impl_NoBroadcastInputBatch<T, VariadicElementwiseOpTag>( \
      cudaStream_t stream,                                               \
      InputBatchArray<T> input_data_batch,                               \
      T * output_data,                                                   \
      size_t count);

// the postfix means the types supported by the op:
// B: uint8_t
// W: uint16_t
// U: uint32_t
// Z: uint64_t
// C: int8_t
// S: int16_t
// I: int32_t
// L: int64_t
// H: float16
// F: float
// D: double
// O: bool

#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
#define SPECIALIZE_IMPL_BF16(VariadicElementwiseOpTag) SPECIALIZE_IMPL(nv_bfloat16, VariadicElementwiseOpTag)
#else
#define SPECIALIZE_IMPL_BF16(VariadicElementwiseOpTag)
#endif

#define SPECIALIZE_IMPL_HFD(VariadicElementwiseOpTag) \
  SPECIALIZE_IMPL(half, VariadicElementwiseOpTag)     \
  SPECIALIZE_IMPL_BF16(VariadicElementwiseOpTag)      \
  SPECIALIZE_IMPL(float, VariadicElementwiseOpTag)    \
  SPECIALIZE_IMPL(double, VariadicElementwiseOpTag)

#define SPECIALIZE_IMPL_UZILHFD(VariadicElementwiseOpTag) \
  SPECIALIZE_IMPL(uint32_t, VariadicElementwiseOpTag)     \
  SPECIALIZE_IMPL(uint64_t, VariadicElementwiseOpTag)     \
  SPECIALIZE_IMPL(int32_t, VariadicElementwiseOpTag)      \
  SPECIALIZE_IMPL(int64_t, VariadicElementwiseOpTag)      \
  SPECIALIZE_IMPL_HFD(VariadicElementwiseOpTag)

SPECIALIZE_IMPL_HFD(variadic_elementwise_ops::Sum)
SPECIALIZE_IMPL_UZILHFD(variadic_elementwise_ops::Min)
SPECIALIZE_IMPL_UZILHFD(variadic_elementwise_ops::Max)

#undef SPECIALIZE_IMPL_UZILHFD
#undef SPECIALIZE_IMPL_HFD
#undef SPECIALIZE_IMPL

}  // namespace cuda
}  // namespace onnxruntime
