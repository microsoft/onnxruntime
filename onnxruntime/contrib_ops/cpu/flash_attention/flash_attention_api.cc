#include "contrib_ops/cpu/flash_attention/flash_attention_api.h"

namespace onnxruntime::contrib {

namespace cpu_default {
void flash_attention_kernel_impl(
    Tensor& output,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    bool is_causal,
    const Tensor* attn_mask,
    double scale,
    concurrency::ThreadPool* thread_pool,
    AllocatorPtr allocator,
    bool is_q_bnsh,
    bool is_kv_bnsh);
}

void cpu_flash_attention(
    Tensor& output,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    bool is_causal,
    const Tensor* attn_mask,
    double scale,
    concurrency::ThreadPool* thread_pool,
    AllocatorPtr allocator,
    bool is_q_bnsh,
    bool is_kv_bnsh) {
  // TODO: dispatch to different kernels according to cpu capabilities (like AVX2, AVX512 etc.)
  return cpu_default::flash_attention_kernel_impl(
      output,
      query,
      key,
      value,
      is_causal,
      attn_mask,
      scale,
      thread_pool,
      allocator,
      is_q_bnsh,
      is_kv_bnsh);
}
}  // namespace onnxruntime::contrib
