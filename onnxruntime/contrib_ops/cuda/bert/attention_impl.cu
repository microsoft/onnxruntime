/*
 The implementation of this file is based on qkvToContext plugin in TensorRT demo:
 https://github.com/NVIDIA/TensorRT/tree/release/5.1/demo/BERT/

Copyright 2019 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "attention_impl.h"
#include "attention_softmax.h"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

static size_t AlignTo(size_t a, size_t b) {
  return CeilDiv(a, b) * b;
}

size_t GetAttentionScratchSize(size_t element_size, int batch_size, int num_heads, int sequence_length, int all_sequence_length) {
  const size_t len = batch_size * num_heads * sequence_length * all_sequence_length;
  const size_t bytes = len * element_size;

  const size_t alignment = 256;
  const size_t bytesAligned = AlignTo(bytes, alignment);
  return bytesAligned;
}

size_t GetAttentionWorkspaceSize(
    size_t element_size,
    int batch_size,
    int num_heads,
    int head_size,
    int sequence_length,
    int past_sequence_length) {
  size_t qkv_size = 3 * batch_size * sequence_length * num_heads * head_size * element_size;
  return qkv_size + 2 * GetAttentionScratchSize(element_size, batch_size, num_heads, sequence_length, past_sequence_length + sequence_length);
}

template <typename T>
bool QkvToContext(
    const cudaDeviceProp& prop, cublasHandle_t& cublas, cudaStream_t stream,
    const int batch_size, const int sequence_length, const int num_heads, const int head_size, const size_t element_size,
    const T* input, T* output, T* workspace,
    const int* mask_index, const std::vector<int64_t>* mask_index_dims,
    bool is_unidirectional, int past_sequence_length, const T* past, T* present) {
  const int all_sequence_length = past_sequence_length + sequence_length;
  const size_t bytes = GetAttentionScratchSize(element_size, batch_size, num_heads, sequence_length, all_sequence_length);
  T* scratch1 = workspace;
  T* scratch2 = scratch1 + (bytes / element_size);
  T* scratch3 = scratch2 + (bytes / element_size);

  const int max_threads_per_block(prop.maxThreadsPerBlock);

  // input should be BxSx3xNxH => scratch3: 3xBxNxSxH
  if (!LaunchTransQkv(stream, sequence_length, batch_size, head_size, num_heads, max_threads_per_block, input, scratch3)) {
    return false;
  }

  // now scratch3 has Q, K, V: each has size BxNxSxH
  const int batches = batch_size * num_heads;
  const int size_per_batch = sequence_length * head_size;
  const int total_size = batches * size_per_batch;

  const T* q = scratch3;
  const T* k = q + total_size;
  const T* v = k + total_size;

  cublasSetStream(cublas, stream);

  // Concat past (2xBxNxS'xH) to present (2xBxNxS*xH):
  // past_k (BxNxS'xH) + k (BxNxSxH) => present_k (BxNxS*xH)
  // past_v (BxNxS'xH) + v (BxNxSxH) => present_v (BxNxS*xH)
  const int present_size_per_batch = all_sequence_length * head_size;
  if (nullptr != present) {
    if (!LaunchConcatPastToPresent(stream, all_sequence_length, sequence_length, batch_size, head_size, num_heads, max_threads_per_block, past, k, present)) {
      return false;
    }

    // update pointers to present_k and present_v.
    k = present;
    v = present + batches * present_size_per_batch;
  }

  // Raw attention mask could be 2D (BxS) or 3D (BxSxS*)
  bool use_raw_attention_mask = (nullptr != mask_index && nullptr != mask_index_dims && mask_index_dims->size() >= 2);

  // compute Q*K' (as K'*Q), scaled by 1/sqrt(H) and store in scratch1: BxNxSxS*
  // Q: BxNxSxH, K (present_k): BxNxS*xH, Q*K': BxNxSxS*
  const float rsqrt_head_size = 1.f / sqrt(static_cast<float>(head_size));
  const int temp_matrix_size = sequence_length * all_sequence_length;
  T one = (T)(1.0f);
  T zero = (T)(0.f);

  // For raw attention mask, the scalar if 1/sqrt(H) is moved to softmax computation.
  // TODO: move scalar to softmax computation since converting 1/Sqrt(H) to half might have loss in precision.
  T alpha = use_raw_attention_mask ? one : (T)(rsqrt_head_size);
  
  if (!CUBLAS_CALL(cublasGemmStridedBatchedHelper(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, all_sequence_length, sequence_length, head_size, &alpha, k, head_size, present_size_per_batch,
          q, head_size, size_per_batch, &zero, scratch1, all_sequence_length, temp_matrix_size, batches, prop))) {
    return false;
  }

  // apply softmax and store result P to scratch2: BxNxSxS*
  if (use_raw_attention_mask) {  // 2d or 3d attention mask
    if (!ComputeSoftmaxWithRawMask<T>(stream, all_sequence_length, sequence_length, batch_size, num_heads, mask_index, scratch1, scratch2, is_unidirectional, rsqrt_head_size, static_cast<int>(mask_index_dims->size()))) {
      return false;
    }
  } else if (nullptr != mask_index) {  // 1d mask index
    ORT_ENFORCE(nullptr != mask_index_dims && mask_index_dims->size() == 1);
    // mask_index has 1D shape: either (batch_size) or (2*batch_size). Only the later one has start postions.
    const int* mask_start = (mask_index_dims->at(0) > batch_size) ? mask_index + batch_size : nullptr;
    if (!ComputeSoftmaxWithMask1D<T>(stream, all_sequence_length, sequence_length, batch_size, num_heads, mask_index, mask_start, scratch1, scratch2, is_unidirectional)) {
      return false;
    }
  } else {  // no mask
    if (!ComputeSoftmax<T>(stream, all_sequence_length, sequence_length, batch_size, num_heads, scratch1, scratch2, is_unidirectional)) {
      return false;
    }
  }

  // compute P*V (as V*P), and store in scratch3: BxNxSxH
  if (!CUBLAS_CALL(cublasGemmStridedBatchedHelper(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, head_size, sequence_length, all_sequence_length, &one, v, head_size, present_size_per_batch,
          scratch2, all_sequence_length, temp_matrix_size, &zero, scratch3, head_size, size_per_batch, batches, prop))) {
    return false;
  }

  // scratch3 is BxNxSxH, transpose to output BxSxNxH
  return LaunchTransCtx(stream, sequence_length, batch_size, head_size, num_heads, max_threads_per_block, scratch3, output);
}

bool LaunchAttentionKernel(
    const cudaDeviceProp& prop,
    cudaStream_t stream,
    const void* input,
    const int* mask_index,
    const std::vector<int64_t>* mask_index_dims,
    void* output,
    const int batch_size,
    const int sequence_length,
    const int num_heads,
    const int head_size,
    void* workspace,
    cublasHandle_t& cublas,
    const size_t element_size,
    bool is_unidirectional,
    int past_sequence_length,
    const void* past,
    void* present) {
  if (element_size == 2) {
    return QkvToContext(prop, cublas, stream,
                        batch_size, sequence_length, num_heads, head_size, element_size,
                        reinterpret_cast<const half*>(input), reinterpret_cast<half*>(output), reinterpret_cast<half*>(workspace),
                        mask_index, mask_index_dims, is_unidirectional,
                        past_sequence_length, reinterpret_cast<const half*>(past), reinterpret_cast<half*>(present));
  } else {
    return QkvToContext(prop, cublas, stream,
                        batch_size, sequence_length, num_heads, head_size, element_size,
                        reinterpret_cast<const float*>(input), reinterpret_cast<float*>(output), reinterpret_cast<float*>(workspace),
                        mask_index, mask_index_dims, is_unidirectional,
                        past_sequence_length, reinterpret_cast<const float*>(past), reinterpret_cast<float*>(present));
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
