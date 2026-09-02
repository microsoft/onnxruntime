// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Head-size / dtype / cache-dtype dispatcher for the paged-KV XQA decode kernels. The kernels
// themselves live in xqa_paged_<query>_<cache>_<head>.cu (each of which instantiates the five
// supported query/KV group sizes through xqa_paged_loader_impl.cuh).

#include "contrib_ops/cuda/bert/xqa/xqa_paged_loader.h"

#include <cuda_bf16.h>
#include <type_traits>

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Signature shared by every per-TU entry point.
#define XQA_PAGED_DECL(fn)               \
  Status fn(                             \
      const cudaDeviceProp& device_prop, \
      cudaStream_t stream,               \
      const void* query,                 \
      const void* key_cache,             \
      const void* value_cache,           \
      void* output,                      \
      const int* page_table,             \
      const int batch_size,              \
      const int num_heads,               \
      const int kv_num_heads,            \
      const int head_size,               \
      const int max_pages_per_seq,       \
      const float scale,                 \
      const int local_window_size,       \
      const int* past_seq_lens,          \
      const float* attention_sinks,      \
      const float* k_cache_scale,        \
      const float* v_cache_scale,        \
      void* workspace,                   \
      size_t workspace_size);            \
  size_t fn##_SmemSize(const int num_heads, const int kv_num_heads)

#define XQA_PAGED_ARGS                                                                           \
  device_prop, stream, query, key_cache, value_cache, output, page_table, batch_size, num_heads, \
      kv_num_heads, head_size, max_pages_per_seq, scale, local_window_size, past_seq_lens,       \
      attention_sinks, k_cache_scale, v_cache_scale, workspace, workspace_size

namespace H64 {
XQA_PAGED_DECL(LaunchXQAPagedInt8Kernel);
XQA_PAGED_DECL(LaunchXQAPagedInt8KernelBF16);
#ifdef USE_FP8_KV_CACHE
XQA_PAGED_DECL(LaunchXQAPagedFp8Kernel);
XQA_PAGED_DECL(LaunchXQAPagedFp8KernelBF16);
#endif
}  // namespace H64

namespace H128 {
XQA_PAGED_DECL(LaunchXQAPagedInt8Kernel);
XQA_PAGED_DECL(LaunchXQAPagedInt8KernelBF16);
#ifdef USE_FP8_KV_CACHE
XQA_PAGED_DECL(LaunchXQAPagedFp8Kernel);
XQA_PAGED_DECL(LaunchXQAPagedFp8KernelBF16);
#endif
}  // namespace H128

namespace H256 {
XQA_PAGED_DECL(LaunchXQAPagedFp16Kernel);
XQA_PAGED_DECL(LaunchXQAPagedInt8Kernel);
XQA_PAGED_DECL(LaunchXQAPagedInt8KernelBF16);
#ifdef USE_FP8_KV_CACHE
XQA_PAGED_DECL(LaunchXQAPagedFp8Kernel);
XQA_PAGED_DECL(LaunchXQAPagedFp8KernelBF16);
#endif
}  // namespace H256

Status LaunchXQAPagedKernel(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    void* output,
    const int* page_table,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_pages_per_seq,
    const float scale,
    const int local_window_size,
    const int* past_seq_lens,
    const float* attention_sinks,
    const float* k_cache_scale,
    const float* v_cache_scale,
    const XqaQuantType kv_quant_type,
    const bool is_bf16,
    void* workspace,
    size_t workspace_size) {
  if (device_prop.major < 8) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "XQA is only supported on Ampere (SM80) or newer GPUs.");
  }

  if (kv_quant_type == XqaQuantType::kNone) {
    if (head_size == 256 && !is_bf16) {
      return H256::LaunchXQAPagedFp16Kernel(XQA_PAGED_ARGS);
    }
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, FAIL,
        "Native-cache paged XQA only supports FP16 query/output with head_size 256. Input has ",
        is_bf16 ? "BF16" : "FP16", " query/output and head_size ", head_size);
  } else if (kv_quant_type == XqaQuantType::kInt8) {
    if (head_size == 64) {
      return is_bf16 ? H64::LaunchXQAPagedInt8KernelBF16(XQA_PAGED_ARGS)
                     : H64::LaunchXQAPagedInt8Kernel(XQA_PAGED_ARGS);
    } else if (head_size == 128) {
      return is_bf16 ? H128::LaunchXQAPagedInt8KernelBF16(XQA_PAGED_ARGS)
                     : H128::LaunchXQAPagedInt8Kernel(XQA_PAGED_ARGS);
    } else if (head_size == 256) {
      return is_bf16 ? H256::LaunchXQAPagedInt8KernelBF16(XQA_PAGED_ARGS)
                     : H256::LaunchXQAPagedInt8Kernel(XQA_PAGED_ARGS);
    }
  } else if (kv_quant_type == XqaQuantType::kFp8) {
#ifdef USE_FP8_KV_CACHE
    if (head_size == 64) {
      return is_bf16 ? H64::LaunchXQAPagedFp8KernelBF16(XQA_PAGED_ARGS)
                     : H64::LaunchXQAPagedFp8Kernel(XQA_PAGED_ARGS);
    } else if (head_size == 128) {
      return is_bf16 ? H128::LaunchXQAPagedFp8KernelBF16(XQA_PAGED_ARGS)
                     : H128::LaunchXQAPagedFp8Kernel(XQA_PAGED_ARGS);
    } else if (head_size == 256) {
      return is_bf16 ? H256::LaunchXQAPagedFp8KernelBF16(XQA_PAGED_ARGS)
                     : H256::LaunchXQAPagedFp8Kernel(XQA_PAGED_ARGS);
    }
#else
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Paged XQA was built without FP8 KV cache support.");
#endif
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Paged XQA does not support the requested KV cache type.");
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                         "Paged XQA only supports head_size 64, 128, or 256. Input has ", head_size);
}

size_t GetXQAPagedRequiredSharedMemoryBytes(
    const cudaDeviceProp& device_prop,
    int head_size,
    int num_heads,
    int kv_num_heads,
    XqaQuantType kv_quant_type,
    [[maybe_unused]] bool is_bf16) {
  if (device_prop.major < 8 || kv_num_heads <= 0) {
    return 0;
  }
  // FP16 and BF16 kernels have identical shared-memory footprints (both 2-byte elements), so the
  // FP16 instantiation is queried for both.
  if (kv_quant_type == XqaQuantType::kNone) {
    if (head_size == 256 && !is_bf16) {
      return H256::LaunchXQAPagedFp16Kernel_SmemSize(num_heads, kv_num_heads);
    }
  } else if (kv_quant_type == XqaQuantType::kInt8) {
    if (head_size == 64) {
      return H64::LaunchXQAPagedInt8Kernel_SmemSize(num_heads, kv_num_heads);
    } else if (head_size == 128) {
      return H128::LaunchXQAPagedInt8Kernel_SmemSize(num_heads, kv_num_heads);
    } else if (head_size == 256) {
      return H256::LaunchXQAPagedInt8Kernel_SmemSize(num_heads, kv_num_heads);
    }
  } else if (kv_quant_type == XqaQuantType::kFp8) {
#ifdef USE_FP8_KV_CACHE
    if (head_size == 64) {
      return H64::LaunchXQAPagedFp8Kernel_SmemSize(num_heads, kv_num_heads);
    } else if (head_size == 128) {
      return H128::LaunchXQAPagedFp8Kernel_SmemSize(num_heads, kv_num_heads);
    } else if (head_size == 256) {
      return H256::LaunchXQAPagedFp8Kernel_SmemSize(num_heads, kv_num_heads);
    }
#endif
  }
  return 0;
}

#undef XQA_PAGED_ARGS
#undef XQA_PAGED_DECL

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
