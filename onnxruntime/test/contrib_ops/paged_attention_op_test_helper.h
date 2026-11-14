// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace onnxruntime {
class IExecutionProvider;

namespace test {

struct PagedAttentionTestCase {
  int batch_size = 1;
  int token_count = 1;
  int num_heads = 1;
  int kv_num_heads = 1;
  int head_size = 8;
  int block_size = 256;
  int num_blocks = 2;
  int max_num_blocks_per_seq = 1;
  float scale = 0.0f;  // 0 means kernel default = 1/sqrt(head_size)
  bool is_causal = true;
  int local_window_size = -1;
  std::vector<int32_t> cumulative_seqlens_q;
  std::vector<int32_t> past_seqlens;
  std::vector<int32_t> block_table;
};

void RunPagedAttentionTest(const PagedAttentionTestCase& c,
                           std::unique_ptr<IExecutionProvider> execution_provider,
                           float output_abs_error = 2e-2f);

}  // namespace test
}  // namespace onnxruntime
