// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <class T>
bool TryMatMul4Bits(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    int shared_mem_per_block,
    cudaStream_t stream);


#if !defined(USE_MIGRAPHX) && !defined(USE_ROCM)

template <typename ElementT>
Status blkq4_fp16_gemm_sm80_dispatch(
  int block_size,
  bool column_wise_blocking,
  int m, int n, int k, cudaStream_t stream,
  ElementT const* a_ptr, size_t a_size,
  uint8_t const* weights_ptr, size_t weights_size,
  ElementT const* scales_ptr, size_t scales_size,
  uint8_t const* offsets_ptr, size_t offsets_size,
  ElementT* output_ptr, size_t output_size);

#endif

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
