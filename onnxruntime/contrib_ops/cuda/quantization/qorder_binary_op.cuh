#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

using onnxruntime::cuda::TArray;
using onnxruntime::cuda::fast_divmod;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define QORDERED_BINARY_TWO_SCALE_DECLARATION(name)  \
  void QOrdered_Impl_##name(                         \
      cudaStream_t stream,                           \
      int32_t output_rank_or_simple_broadcast,       \
      const TArray<int64_t>* lhs_padded_strides,     \
      const int8_t* lhs_data,                        \
      float lhs_scale,                               \
      const TArray<int64_t>* rhs_padded_strides,     \
      const int8_t* rhs_data,                        \
      float rhs_scale,                               \
      const TArray<fast_divmod>* fdm_output_strides, \
      const fast_divmod& fdm_H,                      \
      const fast_divmod& fdm_C,                      \
      int8_t* output_data,                           \
      size_t count)

#define QORDERED_BINARY_THREE_SCALE_DECLARATION(name) \
  void QOrdered_Impl_##name(                          \
      cudaStream_t stream,                            \
      int32_t output_rank_or_simple_broadcast,        \
      const TArray<int64_t>* lhs_padded_strides,      \
      const int8_t* lhs_data,                         \
      float lhs_scale,                                \
      const TArray<int64_t>* rhs_padded_strides,      \
      const int8_t* rhs_data,                         \
      float rhs_scale,                                \
      const TArray<fast_divmod>* fdm_output_strides,  \
      const fast_divmod& fdm_H,                       \
      const fast_divmod& fdm_C,                       \
      float y_scale,                                  \
      int8_t* output_data,                            \
      size_t count)


QORDERED_BINARY_TWO_SCALE_DECLARATION(Add);

// QORDERED_BINARY_THREE_SCALE_DECLARATION(BiasGelu);

  void QOrdered_Col32OrderImpl_BiasGelu(
    cudaStream_t stream,
    const int8_t* input_tensor,
    float input_scale,
    const int8_t* bias_tensor,
    float bias_scale,
    int8_t* output_tensor,
    float output_scale,
    const fast_divmod& batch_size,
    const fast_divmod& rows_times_thirty_two,
    const fast_divmod& thirty_two,
    size_t count);



}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
