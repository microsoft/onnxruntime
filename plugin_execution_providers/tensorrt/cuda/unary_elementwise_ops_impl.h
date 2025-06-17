// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace onnxruntime {
namespace cuda {

// Cast

#define DECL_IMPL_CAST(InT, OutT) \
  void Explicit_Impl_Cast(cudaStream_t stream, const InT* input_data, OutT* output_data, size_t count);

#define DECL_IMPL_CAST_FROM(T) \
  DECL_IMPL_CAST(T, half)      \
  DECL_IMPL_CAST(T, float)     \
  DECL_IMPL_CAST(T, double)    \
  DECL_IMPL_CAST(T, int8_t)    \
  DECL_IMPL_CAST(T, int16_t)   \
  DECL_IMPL_CAST(T, int32_t)   \
  DECL_IMPL_CAST(T, int64_t)   \
  DECL_IMPL_CAST(T, uint8_t)   \
  DECL_IMPL_CAST(T, uint16_t)  \
  DECL_IMPL_CAST(T, uint32_t)  \
  DECL_IMPL_CAST(T, uint64_t)  \
  DECL_IMPL_CAST(T, bool)      \
  //DECL_IMPL_CAST(T, BFloat16)

DECL_IMPL_CAST_FROM(half)
DECL_IMPL_CAST_FROM(float)
DECL_IMPL_CAST_FROM(double)
DECL_IMPL_CAST_FROM(int8_t)
DECL_IMPL_CAST_FROM(int16_t)
DECL_IMPL_CAST_FROM(int32_t)
DECL_IMPL_CAST_FROM(int64_t)
DECL_IMPL_CAST_FROM(uint8_t)
DECL_IMPL_CAST_FROM(uint16_t)
DECL_IMPL_CAST_FROM(uint32_t)
DECL_IMPL_CAST_FROM(uint64_t)
DECL_IMPL_CAST_FROM(bool)
//DECL_IMPL_CAST_FROM(BFloat16)

template <typename InT, typename OutT>
void Impl_Cast(cudaStream_t stream, const InT* input_data, OutT* output_data, size_t count) {
  Explicit_Impl_Cast(stream, input_data, output_data, count);
}

}  // namespace cuda

}  // namespace onnxruntime
