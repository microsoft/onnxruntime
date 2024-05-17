#define CPU_CAPABILITY_AVX512 1
#define CPU_CAPABILITY AVX512
#include "contrib_ops/cpu/flash_attention/flash_attention.h"
#undef CPU_CAPABILITY
#undef CPU_CAPABILITY_AVX512

namespace onnxruntime {
namespace contrib {
void flash_attention_kernel_impl_avx512(
    TensorWrapper& output,
    TensorWrapper& logsumexp,
    const TensorWrapper& query,
    const TensorWrapper& key,
    const TensorWrapper& value,
    bool is_causal,
    const TensorWrapper& attn_mask,
    double scale,
    concurrency::ThreadPool* thread_pool,
    AllocatorPtr allocator,
    bool is_q_bnsh,
    bool is_kv_bnsh) {
  return flash_attention_kernel_impl(
      output,
      logsumexp,
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

}  // namespace contrib
}  // namespace onnxruntime
