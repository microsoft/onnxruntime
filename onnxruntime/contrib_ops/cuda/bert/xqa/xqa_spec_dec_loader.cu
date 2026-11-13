// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/xqa/xqa_loader.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define XQA_SPEC_DEC_DECL(fn)                                                                                      \
  Status fn(const cudaDeviceProp& device_prop, cudaStream_t stream, const void* query, const void* key_cache,      \
            const void* value_cache, void* output, int batch_size, int sequence_length, int num_heads,             \
            int kv_num_heads, int max_seq_len, float scale, int local_window_size, bool is_bsnh,                   \
            const int* past_seq_lens, void* workspace, size_t workspace_size);                                     \
  size_t fn##_WorkspaceSize(const cudaDeviceProp& device_prop, int batch_size, int sequence_length, int num_heads, \
                            int kv_num_heads, int max_seq_len);                                                    \
  size_t fn##_SmemSize(int num_heads, int kv_num_heads);

namespace H128 {
XQA_SPEC_DEC_DECL(LaunchXQASpecDecBf16Kernel)
XQA_SPEC_DEC_DECL(LaunchXQASpecDecFp16Kernel)
}  // namespace H128

#undef XQA_SPEC_DEC_DECL

bool IsXQASpecDecSupported(int head_size, int num_heads, int kv_num_heads, int sequence_length) {
  if (head_size != 128 || kv_num_heads <= 0 || num_heads % kv_num_heads != 0) {
    return false;
  }
  const int group_size = num_heads / kv_num_heads;
  return (group_size == 4 || group_size == 8 || group_size == 16) && sequence_length >= 2 &&
         sequence_length <= kXqaSpecDecMaxSequenceLength;
}

Status LaunchXQASpecDecKernel(const cudaDeviceProp& device_prop, cudaStream_t stream, bool is_bf16, const void* query,
                              const void* key_cache, const void* value_cache, void* output, int batch_size,
                              int sequence_length, int num_heads, int kv_num_heads, int head_size, int max_seq_len,
                              float scale, int local_window_size, bool is_bsnh, const int* past_seq_lens,
                              void* workspace, size_t workspace_size) {
  ORT_RETURN_IF_NOT(IsXQASpecDecSupported(head_size, num_heads, kv_num_heads, sequence_length),
                    "Unsupported speculative XQA shape: head_size=", head_size, " num_heads=", num_heads,
                    " kv_num_heads=", kv_num_heads, " sequence_length=", sequence_length);
  const auto launch = is_bf16 ? H128::LaunchXQASpecDecBf16Kernel : H128::LaunchXQASpecDecFp16Kernel;
  return launch(device_prop, stream, query, key_cache, value_cache, output, batch_size, sequence_length, num_heads,
                kv_num_heads, max_seq_len, scale, local_window_size, is_bsnh, past_seq_lens, workspace,
                workspace_size);
}

size_t GetXQASpecDecWorkspaceSize(const cudaDeviceProp& device_prop, bool is_bf16, int batch_size,
                                  int sequence_length, int num_heads, int kv_num_heads, int head_size,
                                  int max_seq_len) {
  if (!IsXQASpecDecSupported(head_size, num_heads, kv_num_heads, sequence_length)) {
    return 0;
  }
  return is_bf16 ? H128::LaunchXQASpecDecBf16Kernel_WorkspaceSize(device_prop, batch_size, sequence_length,
                                                                  num_heads, kv_num_heads, max_seq_len)
                 : H128::LaunchXQASpecDecFp16Kernel_WorkspaceSize(device_prop, batch_size, sequence_length,
                                                                  num_heads, kv_num_heads, max_seq_len);
}

size_t GetXQASpecDecRequiredSharedMemoryBytes(bool is_bf16, int head_size, int num_heads, int kv_num_heads) {
  if (!IsXQASpecDecSupported(head_size, num_heads, kv_num_heads, 2)) {
    return 0;
  }
  return is_bf16 ? H128::LaunchXQASpecDecBf16Kernel_SmemSize(num_heads, kv_num_heads)
                 : H128::LaunchXQASpecDecFp16Kernel_SmemSize(num_heads, kv_num_heads);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
