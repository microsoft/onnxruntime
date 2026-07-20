// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "xqa_loader.h"
#include <cassert>

// HEAD_ELEMS must be defined by the including file
#ifndef HEAD_ELEMS
#error "HEAD_ELEMS must be defined before including xqa_loader_fp16_impl.cuh"
#endif

// HEAD_DIM_NAMESPACE must be defined by the including file
#ifndef HEAD_DIM_NAMESPACE
#error "HEAD_DIM_NAMESPACE must be defined before including xqa_loader_fp16_impl.cuh"
#endif

// Define global constants based on macros
#define USE_PAGED_KV_CACHE 0
#define TOKENS_PER_PAGE 0
#define INPUT_FP16 1
#define ALLOW_MULTI_BLOCK_MODE 1
// Compile the non-quantized fp16 XQA kernels with sliding-window support so the same
// kernels can serve both global attention (local_window_size == -1, mapped to a window
// >= max_seq_len -> zero masking overhead) and sliding-window models (GPT-OSS / Mistral /
// Gemma2). Attention sinks (head_sink) already work and compose with the window in-kernel.
#define SLIDING_WINDOW 1

#pragma nv_diag_suppress 177
#pragma nv_diag_suppress 20012

// Include common headers once
#include "cuda_hint.cuh"
#include "mha.h"
// Include all helpers globally to ensure visibility
#include "ldgsts.cuh"
#include "mhaUtils.cuh"
#include "mha_components.cuh"
#include "mma.cuh"
#include "utils.cuh"
#include "hostUtils.h"

// Undefine HEAD_GRP_SIZE and M_TILESIZE to allow re-definition in impl gen
#undef HEAD_GRP_SIZE
#undef M_TILESIZE

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace HEAD_DIM_NAMESPACE {

// ============================================================================
// FP16 KV Cache Instantiations
// ============================================================================

#define NAMESPACE_NAME grp1_fp16
#define GRP_SIZE 1
#define M_TILESIZE 8
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME grp2_fp16
#define GRP_SIZE 2
#define M_TILESIZE 8
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME grp4_fp16
#define GRP_SIZE 4
#define M_TILESIZE 8
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME grp5_fp16
#define GRP_SIZE 5
#define M_TILESIZE 8
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME grp8_fp16
#define GRP_SIZE 8
#define M_TILESIZE 8
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME grp16_fp16
#define GRP_SIZE 16
#define M_TILESIZE 16
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME grp32_fp16
#define GRP_SIZE 32
#define M_TILESIZE 32
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

// Extern declarations for INT8 kernels (implemented in xqa_loader_fp16_int8_impl.cuh via instantiation)
Status LaunchXQAInt8Kernel(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    void* output,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_seq_len,
    const float scale,
    const int local_window_size,
    const bool is_bsnh,
    const int* past_seq_lens,
    const float* kv_cache_scale,
    void* workspace,
    size_t workspace_size);

#ifdef USE_FP8_KV_CACHE
// Extern declarations for FP8 kernels (implemented in xqa_loader_fp16_fp8_impl.cuh via instantiation)
Status LaunchXQAFp8Kernel(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    void* output,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_seq_len,
    const float scale,
    const int local_window_size,
    const bool is_bsnh,
    const int* past_seq_lens,
    const float* kv_cache_scale,
    void* workspace,
    size_t workspace_size);
#endif

// ============================================================================
// Dispatcher Implementation
// ============================================================================

template <typename T>
Status LaunchXQAKernelImpl(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    void* output,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_seq_len,
    const float scale,
    const int local_window_size,
    const bool is_bsnh,
    const int* past_seq_lens,
    const float* attention_sinks,
    const float* kv_cache_scale,
    const XqaQuantType kv_quant_type,
    void* workspace,
    size_t workspace_size) {
  // Head size check is done in global dispatcher

  // Dispatch to INT8 path if requested
  if (kv_quant_type == XqaQuantType::kInt8) {
    ORT_RETURN_IF(attention_sinks != nullptr, "XQA attention sinks are not supported with INT8 KV cache.");
    if constexpr (std::is_same<T, half>::value) {
      return LaunchXQAInt8Kernel(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, max_seq_len, scale, local_window_size, is_bsnh, past_seq_lens, kv_cache_scale, workspace, workspace_size);
    } else {
      // BF16 case is handled in xqa_loader_bf16.cu via specialization
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "XQA INT8 path mismatch.");
    }
  }

#ifdef USE_FP8_KV_CACHE
  // Dispatch to FP8 path if requested
  if (kv_quant_type == XqaQuantType::kFp8) {
    ORT_RETURN_IF(attention_sinks != nullptr, "XQA attention sinks are not supported with FP8 KV cache.");
    if constexpr (std::is_same<T, half>::value) {
      return LaunchXQAFp8Kernel(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, max_seq_len, scale, local_window_size, is_bsnh, past_seq_lens, kv_cache_scale, workspace, workspace_size);
    } else {
      // BF16 case is handled in xqa_loader_bf16.cu via specialization
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "XQA FP8 path mismatch.");
    }
  }
#endif

  int group_size = num_heads / kv_num_heads;
  switch (group_size) {
    case 1:
      return grp1_fp16::Launch<T>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, max_seq_len, scale, is_bsnh, past_seq_lens, attention_sinks, kv_cache_scale, workspace, workspace_size, local_window_size);
    case 2:
      return grp2_fp16::Launch<T>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, max_seq_len, scale, is_bsnh, past_seq_lens, attention_sinks, kv_cache_scale, workspace, workspace_size, local_window_size);
    case 4:
      return grp4_fp16::Launch<T>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, max_seq_len, scale, is_bsnh, past_seq_lens, attention_sinks, kv_cache_scale, workspace, workspace_size, local_window_size);
    case 5:
      return grp5_fp16::Launch<T>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, max_seq_len, scale, is_bsnh, past_seq_lens, attention_sinks, kv_cache_scale, workspace, workspace_size, local_window_size);
    case 8:
      return grp8_fp16::Launch<T>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, max_seq_len, scale, is_bsnh, past_seq_lens, attention_sinks, kv_cache_scale, workspace, workspace_size, local_window_size);
    case 16:
      return grp16_fp16::Launch<T>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, max_seq_len, scale, is_bsnh, past_seq_lens, attention_sinks, kv_cache_scale, workspace, workspace_size, local_window_size);
    case 32:
      return grp32_fp16::Launch<T>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, max_seq_len, scale, is_bsnh, past_seq_lens, attention_sinks, kv_cache_scale, workspace, workspace_size, local_window_size);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "XQA supports group_size 1, 2, 4, 5, 8, 16, 32. Input has ", group_size);
  }
}

#ifndef GENERATE_CUBIN
// Returns the dynamic shared-memory (bytes) the non-quantized XQA kernel for this head dim and
// group size requests at launch (read from the device symbol, so it is accurate even for a PTX
// kernel JIT-compiled for the running SM). Used by the GQA dispatcher to skip XQA when the
// device's per-block opt-in shared memory is too small. The non-quantized kernel is an upper
// bound for the int8/fp8 variants (smaller cache element), so this single query also guards the
// quantized paths. Not marked inline: it is defined in exactly one TU per head-dim namespace
// (xqa_loader_fp16_{64,128,256}.cu) and called from the head-size dispatcher TU, so it needs an
// externally linkable definition.
size_t GetXQAKernelSmemBytes(int group_size) {
  switch (group_size) {
    case 1:
      return grp1_fp16::GetSmemSize();
    case 2:
      return grp2_fp16::GetSmemSize();
    case 4:
      return grp4_fp16::GetSmemSize();
    case 5:
      return grp5_fp16::GetSmemSize();
    case 8:
      return grp8_fp16::GetSmemSize();
    case 16:
      return grp16_fp16::GetSmemSize();
    case 32:
      return grp32_fp16::GetSmemSize();
    default:
      return 0;
  }
}
#endif  // GENERATE_CUBIN

// Instantiate template for half

}  // namespace HEAD_DIM_NAMESPACE
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
