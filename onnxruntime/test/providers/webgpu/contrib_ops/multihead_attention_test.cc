// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)

#include <algorithm>
#include <cmath>
#include <limits>

#include "gtest/gtest.h"

#include "core/common/inlined_containers.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/webgpu/attention_test_utils.h"

namespace onnxruntime::test {

namespace {

void RunNonCausalPrefill(bool verify_dispatch) {
  if (!WebGpuSupportsFp16()) {
    GTEST_SKIP() << "WebGPU shader-f16 support is unavailable";
  }
  constexpr int sequence_length = 129;
  constexpr int num_heads = 2;
  constexpr int head_size = 128;
  constexpr int hidden_size = num_heads * head_size;
  for (int kv_length : {83, 145}) {
    for (bool bnsh_query : {false, true}) {
      SCOPED_TRACE(testing::Message() << "kv_length=" << kv_length << " bnsh_query=" << bnsh_query);
      InlinedVector<MLFloat16> query(sequence_length * hidden_size);
      InlinedVector<MLFloat16> key(kv_length * hidden_size);
      InlinedVector<MLFloat16> value(key.size());
      InlinedVector<MLFloat16> bias(sequence_length * kv_length);
      InlinedVector<MLFloat16> expected(query.size());
      for (int row = 0; row < sequence_length; ++row) {
        for (int head = 0; head < num_heads; ++head) {
          for (int d = 0; d < head_size; ++d) {
            query[row * hidden_size + head * head_size + d] =
                MLFloat16(((row * 7 + head * 11 + d * 3) % 29 - 14) * 0.125f);
          }
        }
      }
      for (int k = 0; k < kv_length; ++k) {
        for (int d = 0; d < hidden_size; ++d) {
          key[k * hidden_size + d] =
              MLFloat16(((k * 11 + (d / head_size) * 5 + (d % head_size) * 7) % 31 - 15) * 0.125f);
          value[k * hidden_size + d] = MLFloat16((k % 11) * 0.125f + (d % 7) * 0.0625f);
        }
      }
      for (int row = 0; row < sequence_length; ++row) {
        for (int k = 0; k < kv_length; ++k) {
          bias[row * kv_length + k] = MLFloat16(((k + row) % 13) * 0.25f);
        }
        for (int head = 0; head < num_heads; ++head) {
          InlinedVector<double> scores(kv_length);
          double maximum = -std::numeric_limits<double>::infinity();
          for (int k = 0; k < kv_length; ++k) {
            double dot = 0.0;
            for (int d = 0; d < head_size; ++d) {
              dot += static_cast<double>(query[row * hidden_size + head * head_size + d].ToFloat()) *
                     key[k * hidden_size + head * head_size + d].ToFloat();
            }
            scores[k] = dot / std::sqrt(static_cast<double>(head_size)) + bias[row * kv_length + k].ToFloat();
            maximum = std::max(maximum, scores[k]);
          }
          double denominator = 0.0;
          for (int k = 0; k < kv_length; ++k) {
            scores[k] = std::exp(scores[k] - maximum);
            denominator += scores[k];
          }
          for (int d = 0; d < head_size; ++d) {
            double sum = 0.0;
            for (int k = 0; k < kv_length; ++k) {
              sum += scores[k] * value[k * hidden_size + head * head_size + d].ToFloat();
            }
            expected[row * hidden_size + head * head_size + d] = MLFloat16(static_cast<float>(sum / denominator));
          }
        }
      }
      OpTester test("MultiHeadAttention", 1, kMSDomain);
      test.AddAttribute<int64_t>("num_heads", num_heads);
      test.AddAttribute<int64_t>("unidirectional", 0);
      test.AddInput<MLFloat16>("query", {1, sequence_length, hidden_size}, query.data(), query.size());
      test.AddInput<MLFloat16>("key", {1, kv_length, hidden_size}, key.data(), key.size());
      test.AddInput<MLFloat16>("value", {1, kv_length, hidden_size}, value.data(), value.size());
      if (bnsh_query) {
        // Projection bias makes the EP transpose Q to BNSH before flash attention.
        const InlinedVector<MLFloat16> projection_bias(3 * hidden_size, MLFloat16(0.0f));
        test.AddInput<MLFloat16>("bias", {3 * hidden_size}, projection_bias.data(), projection_bias.size());
      } else {
        test.AddOptionalInputEdge<MLFloat16>();
      }
      test.AddOptionalInputEdge<int32_t>();  // key_padding_mask
      test.AddInput<MLFloat16>("attention_bias", {1, 1, sequence_length, kv_length}, bias.data(), bias.size());
      test.AddOutput<MLFloat16>("output", {1, sequence_length, hidden_size}, expected.data(), expected.size());
      test.SetOutputAbsErr("output", 0.001f);
      SessionOptions options;
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
      RunWithCooperativeDispatchCheck(verify_dispatch, [&] {
        test.Config(options).ConfigEp(DefaultWebGpuExecutionProvider()).RunWithConfig();
      });
    }
  }
}

}  // namespace

TEST(MultiHeadAttentionTest, WebGPU_FP16_CooperativePrefill_NonCausal) {
  RunNonCausalPrefill(false);
}

TEST(MultiHeadAttentionTest, WebGPU_FP16_CooperativePrefill_NonCausal_Dispatch) {
  if (!SupportsCooperativeDispatchObservation()) {
    GTEST_SKIP() << "Requires Qualcomm cooperative prefill and timestamp queries";
  }
  RunNonCausalPrefill(true);
}

}  // namespace onnxruntime::test

#endif  // !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_CONTRIB_OPS)
