// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include "gtest/gtest.h"

#include "test/contrib_ops/paged_attention_op_test_helper.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/webgpu/attention_test_utils.h"

namespace onnxruntime::test {

TEST(PagedAttention, WebGpu_CooperativePrefill_RaggedDenseFallback) {
  if (!WebGpuSupportsFp16()) {
    GTEST_SKIP() << "WebGPU shader-f16 support is unavailable";
  }
  for (bool causal : {false, true}) {
    for (int window : {-1, 17}) {
      SCOPED_TRACE(causal);
      SCOPED_TRACE(window);
      PagedAttentionTestCase c{};
      c.batch_size = 2;
      c.token_count = 230;
      c.num_heads = 6;
      c.kv_num_heads = 2;
      c.head_size = 128;
      c.block_size = 16;  // Forces dense fallback instead of fused paged prefill.
      c.num_blocks = 20;
      c.max_num_blocks_per_seq = 10;
      c.is_causal = causal;
      c.local_window_size = window;
      c.cumulative_seqlens_q = {0, 129, 230};
      c.past_seqlens = {17, 7};
      c.block_table = {18, 2, 16, 4, 14, 6, 12, 8, 10, 0,
                       19, 1, 17, 3, 15, 5, 13, 7, 11, 9};
      RunPagedAttentionTest(c, DefaultWebGpuExecutionProvider(), /*output_abs_error=*/1e-4f);
    }
  }
}

}  // namespace onnxruntime::test

#endif  // !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)
