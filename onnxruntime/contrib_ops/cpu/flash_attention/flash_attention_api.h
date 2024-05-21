#pragma once

#include "core/common/common.h"
#include "core/providers/common.h"
#include "core/platform/threadpool.h"

namespace onnxruntime::contrib {

void cpu_flash_attention(
    Tensor& output,
    Tensor& logsumexp,
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

}  // namespace onnxruntime::contrib
