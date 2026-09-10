// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "contrib_ops/cuda/bert/attention_data.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, typename TCACHE>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* stream,
    contrib::PagedAttentionParameters& parameters,
    PagedAttentionData<T, TCACHE>& data);

template <typename T>
Status LaunchUnpackQKVCumulative(const T* packed_qkv, T* unpacked_q, T* unpacked_k, T* unpacked_v, const int num_heads,
                                 const int kv_num_heads, const int head_size, const int token_count, cudaStream_t stream,
                                 const int max_threads_per_block);

// Exposed so paged_attention.cc can populate cumulative_seqlens_kv on both the FA and MEA
// dispatch paths (producer hoisted out of FlashAttention/UnfusedAttention in impl.cu).
Status LaunchGetCumulativeSeqlensKV(int32_t* cumulative_seqlens_kv, const int32_t* cumulative_seqlens_q,
                                    const int32_t* past_seqlens, const int batch_size, cudaStream_t stream);

Status LaunchSanitizeBlockTable(const int32_t* block_table, int32_t* sanitized_block_table,
                                int element_count, int num_blocks, cudaStream_t stream);

// Paged decode backend sizing helpers, used by paged_attention.cc to test eligibility (the kernel
// needs more dynamic shared memory than the device provides for very wide heads) and to size the
// split-KV workspaces.
size_t GetPagedDecodeSharedMemoryBytes(const int head_size);
int ComputePagedDecodeSplits(const int token_count, const int num_heads, const int max_kv_len,
                             const int multi_processor_count);

// Shared memory required by the unfused latent (absorbed MLA) kernel. Used by paged_attention.cc to
// reject a latent configuration the device cannot hold instead of failing at launch time.
size_t GetPagedLatentSharedMemoryBytes(const int head_size, const int v_head_size);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
