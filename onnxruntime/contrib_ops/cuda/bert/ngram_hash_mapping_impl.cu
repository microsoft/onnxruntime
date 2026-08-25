// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/ngram_hash_mapping_impl.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "contrib_ops/cuda/bert/kernel_helper.cuh"
#include "core/providers/cuda/cu_inc/cuda_type_helper.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

template <typename T>
__global__ void NgramHashMappingKernel(
    const T* input_ids,
    const T* multipliers,
    const T* vocab_sizes,
    T* output,
    int64_t total,
    int64_t sequence_length,
    int64_t max_ngram_size,
    int64_t n_head_per_ngram,
    T pad_id) {
  const int64_t num_heads = (max_ngram_size - 1) * n_head_per_ngram;
  for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < total;
       linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t t = linear % sequence_length;
    const int64_t b = linear / sequence_length;
    const int64_t input_base = b * sequence_length;
    const int64_t output_base = linear * num_heads;

    for (int64_t n = 2; n <= max_ngram_size; ++n) {
      T mix = 0;
      for (int64_t k = 0; k < n; ++k) {
        const int64_t source_t = t - k;
        const T token = source_t < 0 ? pad_id : input_ids[input_base + source_t];
        const T product = kernel_helper::WrappedMultiply<T>(token, multipliers[k]);
        mix = k == 0 ? product : static_cast<T>(mix ^ product);
      }

      const int64_t ngram_offset = (n - 2) * n_head_per_ngram;
      for (int64_t h = 0; h < n_head_per_ngram; ++h) {
        const int64_t out_h = ngram_offset + h;
        const T mod = vocab_sizes[out_h];
        output[output_base + out_h] = mod <= 0 ? T{} : kernel_helper::PositiveMod(mix, mod);
      }
    }
  }
}

}  // namespace

template <typename T>
Status LaunchNgramHashMappingKernel(
    cudaStream_t stream,
    const T* input_ids,
    const T* multipliers,
    const T* vocab_sizes,
    T* output,
    int64_t batch_size,
    int64_t sequence_length,
    int64_t max_ngram_size,
    int64_t n_head_per_ngram,
    T pad_id) {
  const int64_t total = batch_size * sequence_length;
  if (total == 0) {
    return Status::OK();
  }
  NgramHashMappingKernel<T><<<kernel_helper::GridSize(total), kernel_helper::kThreads, 0, stream>>>(
      input_ids, multipliers, vocab_sizes, output, total, sequence_length, max_ngram_size,
      n_head_per_ngram, pad_id);
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchNgramHashMappingKernel<int32_t>(cudaStream_t, const int32_t*, const int32_t*, const int32_t*, int32_t*, int64_t, int64_t, int64_t, int64_t, int32_t);
template Status LaunchNgramHashMappingKernel<int64_t>(cudaStream_t, const int64_t*, const int64_t*, const int64_t*, int64_t*, int64_t, int64_t, int64_t, int64_t, int64_t);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
