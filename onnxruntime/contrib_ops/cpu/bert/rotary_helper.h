// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {
namespace rotary_helper {

template <typename T>
Status PackVIntoRotaryQKV(concurrency::ThreadPool* tp,
                          int batch_size,
                          int sequence_length,
                          int num_heads,
                          int kv_num_heads,
                          int head_size,
                          const T* input,
                          T* output) {
  int seq_stride = head_size;
  int head_stride = sequence_length * seq_stride;
  int batch_stride = (num_heads + 2 * kv_num_heads) * head_stride;

  const int loop_len = batch_size * sequence_length * kv_num_heads;
  const double cost = static_cast<double>(head_size);
  ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    for (std::ptrdiff_t ptr = begin; ptr != end; ++ptr) {
      const int b = static_cast<int>((ptr / kv_num_heads) / sequence_length);
      const int s = static_cast<int>((ptr / kv_num_heads) % sequence_length);
      const int n = static_cast<int>(ptr % kv_num_heads);
      const int block_offset = b * batch_stride + s * seq_stride + n * head_stride;
      const T* input_data = input + block_offset;
      T* output_data = output + block_offset;
      for (int i = 0; i < head_size; i++) {
        output_data[i] = input_data[i];
      }
    }
  });
  return Status::OK();
}

}  // namespace rotary_helper
}  // namespace contrib
}  // namespace onnxruntime
