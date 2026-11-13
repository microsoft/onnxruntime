// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <vector>

namespace onnxruntime::test {

std::vector<float> RunGQAReference(
    int batch_size,
    int sequence_length,
    int num_heads,
    int kv_num_heads,
    int head_size,
    const std::vector<float>& query_data,
    const std::vector<float>& key_data,
    const std::vector<float>& value_data,
    bool do_rotary,
    bool use_fp16 = false,
    bool rotary_interleaved = false);

void ExpectBlockQuantInt8Close(const std::vector<float>& reference,
                               const std::vector<float>& actual,
                               float max_relative_rmse,
                               float max_absolute_error);

void RunFp16HighMagnitudeAttention(int sequence_length, uint32_t bit_width, int head_size = 128);

}  // namespace onnxruntime::test
