// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <tuple>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/graph/model.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "default_providers.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime {
namespace test {

namespace {

constexpr int kHeadSize = 8;
constexpr int kBlockSize = 16;
constexpr int kCacheElementCount = kBlockSize * kHeadSize;
constexpr int kCacheElems = kCacheElementCount;

std::vector<MLFloat16> HalfVector(float value, int count = kHeadSize) {
  return std::vector<MLFloat16>(count, MLFloat16(value));
}

void AddAttributes(OpTester& tester) {
  tester.AddAttribute<int64_t>("num_heads", 1);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
}

void AddSingleTokenPrefix(OpTester& tester,
                          const std::vector<MLFloat16>& query,
                          const std::vector<MLFloat16>& key,
                          const std::vector<MLFloat16>& value,
                          const std::vector<int32_t>& selected_indices,
                          int32_t selected_count,
                          int32_t past_seqlen = 0,
                          int32_t block_id = 0,
                          int32_t slot = 0,
                          const std::vector<MLFloat16>& key_cache = HalfVector(0.0f, kCacheElementCount),
                          const std::vector<MLFloat16>& value_cache = HalfVector(0.0f, kCacheElementCount)) {
  AddAttributes(tester);
  tester.AddInput<MLFloat16>("query", {1, kHeadSize}, query);
  tester.AddInput<MLFloat16>("key", {1, kHeadSize}, key);
  tester.AddInput<MLFloat16>("value", {1, kHeadSize}, value);
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize}, key_cache);
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize}, value_cache);
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 1});
  tester.AddInput<int32_t>("past_seqlens", {1}, {past_seqlen});
  tester.AddInput<int32_t>("block_table", {1, 1}, {block_id});
  tester.AddInput<int32_t>("slot_mapping", {1}, {slot});
  tester.AddInput<int32_t>("selected_indices", {1, static_cast<int64_t>(selected_indices.size())},
                           selected_indices);
  tester.AddInput<int32_t>("selected_counts", {1}, {selected_count});
}

std::vector<MLFloat16> AddQuantizedInputs(OpTester& tester,
                                          const std::string& k_quant_type,
                                          const std::string& v_quant_type,
                                          const std::vector<float>& k_scales,
                                          const std::vector<float>& v_scales) {
  AddAttributes(tester);
  tester.AddAttribute<std::string>("k_quant_type", k_quant_type);
  tester.AddAttribute<std::string>("v_quant_type", v_quant_type);
  tester.AddAttribute<float>("scale", 1.0f);
  auto query = HalfVector(0.0f, 2 * kHeadSize);
  query[kHeadSize + 1] = MLFloat16(1.0f);
  auto key = HalfVector(0.0f, 2 * kHeadSize);
  key[kHeadSize + 1] = MLFloat16(1.5f);
  auto value = HalfVector(0.0f, 2 * kHeadSize);
  std::vector<MLFloat16> current_value;
  current_value.reserve(kHeadSize);
  for (int i = 0; i < kHeadSize; ++i) {
    const float scale = v_quant_type == "PER_CHANNEL" ? v_scales[i] : v_scales[0];
    current_value.emplace_back(scale * static_cast<float>(i + 1));
  }
  std::copy(current_value.begin(), current_value.end(), value.begin() + kHeadSize);
  tester.AddInput<MLFloat16>("query", {2, kHeadSize}, query);
  tester.AddInput<MLFloat16>("key", {2, kHeadSize}, key);
  tester.AddInput<MLFloat16>("value", {2, kHeadSize}, value);
  tester.AddInput<int8_t>("key_cache", {1, kBlockSize, 1, kHeadSize},
                          std::vector<int8_t>(kCacheElementCount, 0));
  tester.AddInput<int8_t>("value_cache", {1, kBlockSize, 1, kHeadSize},
                          std::vector<int8_t>(kCacheElementCount, 0));
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 2});
  tester.AddInput<int32_t>("past_seqlens", {1}, {0});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  tester.AddInput<int32_t>("slot_mapping", {2}, {0, 1});
  tester.AddInput<int32_t>("selected_indices", {2, 2}, {0, -1, 0, 1});
  tester.AddInput<int32_t>("selected_counts", {2}, {1, 2});
  tester.AddOptionalInputEdge<MLFloat16>();  // auxiliary_key
  tester.AddOptionalInputEdge<MLFloat16>();  // auxiliary_value
  tester.AddOptionalInputEdge<int32_t>();    // auxiliary_lengths
  tester.AddOptionalInputEdge<MLFloat16>();  // cos_cache
  tester.AddOptionalInputEdge<MLFloat16>();  // sin_cache
  tester.AddOptionalInputEdge<MLFloat16>();  // head_sink
  tester.AddOptionalInputEdge<MLFloat16>();  // q_norm_weight
  tester.AddOptionalInputEdge<MLFloat16>();  // k_norm_weight
  tester.AddInput<float>("k_scale", {static_cast<int64_t>(k_scales.size())}, k_scales);
  tester.AddInput<float>("v_scale", {static_cast<int64_t>(v_scales.size())}, v_scales);

  const float current_weight = std::exp(1.5f) / (1.0f + std::exp(1.5f));
  auto expected = HalfVector(0.0f);
  for (const MLFloat16 element : current_value) {
    expected.emplace_back(element.ToFloat() * current_weight);
  }
  return expected;
}

void RunCuda(OpTester& tester, const SessionOptions& options = {}) {
  auto cuda_ep = DefaultCudaExecutionProvider();
  ASSERT_NE(cuda_ep, nullptr);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  tester.Run(options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

void AddCommonInputs(OpTester& tester, float current_value,
                     const std::vector<int32_t>& selected_indices = {0, -1},
                     int32_t selected_count = 1) {
  tester.AddAttribute<int64_t>("num_heads", 1);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddInput<MLFloat16>("query", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("key", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(current_value)));
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize},
                             std::vector<MLFloat16>(kBlockSize * kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize},
                             std::vector<MLFloat16>(kBlockSize * kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 1});
  tester.AddInput<int32_t>("past_seqlens", {1}, {0});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  tester.AddInput<int32_t>("slot_mapping", {1}, {0});
  tester.AddInput<int32_t>("selected_indices", {1, static_cast<int64_t>(selected_indices.size())},
                           selected_indices);
  tester.AddInput<int32_t>("selected_counts", {1}, {selected_count});
}

void RunWebGpu(OpTester& tester,
               OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
               const std::string& expected_failure_string = "") {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  ASSERT_NE(webgpu_ep, nullptr);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  tester.Run(expect_result, expected_failure_string, {}, nullptr, &execution_providers);
}

void AddCacheProbeInputs(OpTester& tester,
                         float current_value,
                         float cached_value_at_slot0,
                         int32_t slot_mapping_value) {
  std::vector<MLFloat16> value_cache_data(kCacheElems, MLFloat16(0.0f));
  for (int i = 0; i < kHeadSize; ++i) {
    value_cache_data[i] = MLFloat16(cached_value_at_slot0);
  }

  tester.AddAttribute<int64_t>("num_heads", 1);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddInput<MLFloat16>("query", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("key", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(current_value)));
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize},
                             std::vector<MLFloat16>(kCacheElems, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize}, value_cache_data);
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 1});
  tester.AddInput<int32_t>("past_seqlens", {1}, {0});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  tester.AddInput<int32_t>("slot_mapping", {1}, {slot_mapping_value});
  tester.AddInput<int32_t>("selected_indices", {1, 2}, {0, -1});
  tester.AddInput<int32_t>("selected_counts", {1}, {1});
}

void SetCacheRow(std::vector<MLFloat16>& cache, int slot, int head_size,
                 const std::vector<float>& row) {
  for (int i = 0; i < head_size; ++i) {
    cache[slot * head_size + i] =
        MLFloat16(i < static_cast<int>(row.size()) ? row[i] : 0.0f);
  }
}

void SetConstantCacheRow(std::vector<MLFloat16>& cache, int slot, int head_size, float value) {
  for (int i = 0; i < head_size; ++i) {
    cache[slot * head_size + i] = MLFloat16(value);
  }
}

float JointSoftmax(const std::vector<float>& logits, const std::vector<float>& values) {
  const float max_logit = *std::max_element(logits.begin(), logits.end());
  double numerator = 0.0;
  double denominator = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    const double weight = std::exp(static_cast<double>(logits[i] - max_logit));
    numerator += weight * values[i];
    denominator += weight;
  }
  return static_cast<float>(numerator / denominator);
}

}  // namespace

TEST(SparsePagedAttention, Cuda_GroupedSessionOption) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  const auto run = [](const char* config_value, const char* expected_error = nullptr,
                      const char* config_key = "ep.cuda.sparse_paged_attention_grouped") {
    OpTester tester("SparsePagedAttention", 1, kMSDomain);
    AddSingleTokenPrefix(tester, HalfVector(0.0f), HalfVector(0.0f), HalfVector(1.0f), {0}, 1);
    tester.AddOutput<MLFloat16>("output", {1, kHeadSize}, HalfVector(1.0f));
    SessionOptions options;
    if (config_value != nullptr) {
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry(config_key, config_value));
    }
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    tester.Run(options,
               expected_error != nullptr ? OpTester::ExpectResult::kExpectFailure : OpTester::ExpectResult::kExpectSuccess,
               expected_error != nullptr ? expected_error : "", {}, nullptr, &providers);
  };
  for (const char* env_value : {"", "0", "1", "invalid"}) {
    SCOPED_TRACE(env_value);
    ScopedEnvironmentVariables env(EnvVarMap{{"ORT_SPARSE_PAGED_ATTENTION_GROUPED", env_value}});
    run("0");
    run("1");
    run(nullptr, std::string(env_value) == "invalid" ? "Failed to parse environment variable" : nullptr);
  }
  ScopedEnvironmentVariables env(EnvVarMap{{"ORT_SPARSE_PAGED_ATTENTION_GROUPED", "0"}});
  for (const char* config_value : {"", "2", "true", "-1"}) {
    SCOPED_TRACE(config_value);
    run(config_value, "ep.cuda.sparse_paged_attention_grouped must be 0 or 1");
  }
  constexpr const char* kTileConfig = "ep.cuda.sparse_paged_attention_grouped_tile_size";
  run("8", nullptr, kTileConfig);
  for (const char* tile : {"16", "", "0", "32", "eight"}) {
    run(tile, "only supports 8", kTileConfig);
  }
  for (const char* config : {"ep.cuda.sparse_paged_attention_grouped_decode",
                             "ep.cuda.sparse_paged_attention_grouped_decode_splits",
                             "ep.cuda.sparse_paged_attention_warp_reduction"}) {
    run("0", nullptr, config);
    for (const char* value : {"1", "8", "32", "", "-1", "33", "1suffix", "999999999999"}) {
      run(value, "only supports 0", config);
    }
  }
  constexpr const char* kVectorizedConfig = "ep.cuda.sparse_paged_attention_grouped_vectorized";
  run("0", nullptr, kVectorizedConfig);
  run("1", nullptr, kVectorizedConfig);
  run("2", "must be 0 or 1", kVectorizedConfig);
}

TEST(SparsePagedAttention, Cuda_GroupedProductionGeometryMatchesReference) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  constexpr int kHeads = 24;
  constexpr int kKvHeads = 2;
  constexpr int kChannels = 256;
  constexpr int kBlocks = 5;
  constexpr int kSelected = 257;
  constexpr int kAuxiliaryCapacity = 259;
  std::vector<std::vector<MLFloat16>> baseline_outputs;
  for (const int arm : {0, 1, 2, 3, 4}) {
    SCOPED_TRACE(arm);
    ScopedEnvironmentVariables scoped_env(EnvVarMap{{"ORT_SPARSE_PAGED_ATTENTION_GROUPED", arm == 1 || arm == 3 ? "1" : "0"}});
    SessionOptions options;
    if (arm >= 2) {
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry("ep.cuda.sparse_paged_attention_grouped", arm == 3 ? "0" : "1"));
    }
    if (arm == 4) {
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry("ep.cuda.sparse_paged_attention_grouped_vectorized", "1"));
    }
    size_t case_index = 0;
    for (const int rows : {1, 2, 3, 4, 6, 8, 32}) {
      for (const bool auxiliary : {false, true}) {
        for (const bool shared_auxiliary : {false, true}) {
          if (!auxiliary && shared_auxiliary) {
            continue;
          }
          for (const bool local : {false, true}) {
            const int batches = rows == 1 ? 1 : 2;
            const std::vector<int32_t> cumulative = rows == 1 ? std::vector<int32_t>{0, 1}
                                                              : std::vector<int32_t>{0, rows - 1, rows};
            const std::vector<int32_t> past(batches, 15);
            const std::vector<int32_t> table = rows == 1 ? std::vector<int32_t>{2, 0, 4}
                                                         : std::vector<int32_t>{2, 0, 4, 1, 3, -1};
            std::vector<int32_t> slots(rows);
            std::vector<int32_t> indices(rows * kSelected, -1);
            std::vector<int32_t> counts(rows, kSelected);
            std::vector<MLFloat16> query(rows * kHeads * kChannels);
            std::vector<MLFloat16> key(rows * kKvHeads * kChannels);
            std::vector<MLFloat16> value(key.size());
            std::vector<MLFloat16> cached_key(kBlocks * kBlockSize * kKvHeads * kChannels);
            std::vector<MLFloat16> cached_value(cached_key.size());
            std::vector<MLFloat16> auxiliary_key(batches * kAuxiliaryCapacity * kKvHeads * kChannels);
            std::vector<MLFloat16> auxiliary_value(auxiliary_key.size());
            const auto fill = [](std::vector<MLFloat16>& data, int modulus, float scale) {
              for (size_t element = 0; element < data.size(); ++element) {
                data[element] = MLFloat16((static_cast<int>(element % modulus) - modulus / 2) * scale);
              }
            };
            fill(query, 17, 0.0625f);
            fill(key, 13, 0.125f);
            fill(value, 23, 0.125f);
            fill(cached_key, 19, 0.0625f);
            fill(cached_value, 29, 0.125f);
            fill(auxiliary_key, 31, 0.0625f);
            fill(auxiliary_value, 37, 0.125f);
            for (int row = 0; row < rows; ++row) {
              const int batch = batches == 2 && row == rows - 1 ? 1 : 0;
              const int position = past[batch] + row - cumulative[batch];
              slots[row] = table[batch * 3 + position / kBlockSize] * kBlockSize + position % kBlockSize;
              for (int selected = 0; selected < kSelected; ++selected) {
                indices[row * kSelected + selected] = auxiliary ? (selected * 67 + row) % kAuxiliaryCapacity
                                                                : (selected * 7 + row) % (position + 2);
              }
              indices[row * kSelected] = -1;
              indices[row * kSelected + 1] = 1000000;
              if (row % 3 == 1) {
                counts[row] = 9;
              }
              if (row % 3 == 2) {
                counts[row] = 0;
              }
            }
            std::vector<MLFloat16> expected(query.size());
            for (int row = 0; row < rows; ++row) {
              const int batch = batches == 2 && row == rows - 1 ? 1 : 0;
              const int position = past[batch] + row - cumulative[batch];
              const int local_begin = std::max(0, position - 3);
              for (int head = 0; head < kHeads; ++head) {
                const int kv_head = head / 12;
                std::vector<float> logits;
                std::vector<const MLFloat16*> values;
                const auto add_candidate = [&](int logical, bool from_auxiliary) {
                  if (logical < 0 || (from_auxiliary ? logical >= kAuxiliaryCapacity - 2 : logical > position)) {
                    return;
                  }
                  const MLFloat16* key_row;
                  const MLFloat16* value_row;
                  if (from_auxiliary) {
                    const int base = ((batch * kAuxiliaryCapacity + logical) * kKvHeads + kv_head) * kChannels;
                    key_row = auxiliary_key.data() + base;
                    value_row = shared_auxiliary ? key_row : auxiliary_value.data() + base;
                  } else if (logical >= past[batch]) {
                    const int base = ((cumulative[batch] + logical - past[batch]) * kKvHeads + kv_head) * kChannels;
                    key_row = key.data() + base;
                    value_row = value.data() + base;
                  } else {
                    const int slot = table[batch * 3 + logical / kBlockSize] * kBlockSize + logical % kBlockSize;
                    const int base = (slot * kKvHeads + kv_head) * kChannels;
                    key_row = cached_key.data() + base;
                    value_row = cached_value.data() + base;
                  }
                  float dot = 0.0f;
                  for (int channel = 0; channel < kChannels; ++channel) {
                    dot += query[(row * kHeads + head) * kChannels + channel].ToFloat() * key_row[channel].ToFloat();
                  }
                  logits.push_back(std::tanh(dot * 0.0625f / 2.0f) * 2.0f);
                  values.push_back(value_row);
                };
                if (local) {
                  for (int logical = local_begin; logical <= position; ++logical) {
                    add_candidate(logical, false);
                  }
                }
                for (int selected = 0; selected < counts[row]; ++selected) {
                  const int logical = indices[row * kSelected + selected];
                  if (!auxiliary && local && logical >= local_begin && logical <= position) {
                    continue;
                  }
                  add_candidate(logical, auxiliary);
                }
                float max_logit = 0.25f;
                for (const float logit : logits) {
                  max_logit = std::max(max_logit, logit);
                }
                double denominator = std::exp(0.25f - max_logit);
                for (const float logit : logits) {
                  denominator += std::exp(logit - max_logit);
                }
                for (int channel = 0; channel < kChannels; ++channel) {
                  double numerator = 0.0;
                  for (size_t candidate = 0; candidate < logits.size(); ++candidate) {
                    numerator += std::exp(logits[candidate] - max_logit) * values[candidate][channel].ToFloat();
                  }
                  expected[(row * kHeads + head) * kChannels + channel] = MLFloat16(static_cast<float>(numerator / denominator));
                }
              }
            }
            OpTester tester("SparsePagedAttention", 1, kMSDomain);
            tester.AddAttribute<int64_t>("num_heads", kHeads);
            tester.AddAttribute<int64_t>("kv_num_heads", kKvHeads);
            tester.AddAttribute<float>("softcap", 2.0f);
            if (local) {
              tester.AddAttribute<std::string>("attention_mode", "local_plus_selected");
              tester.AddAttribute<int64_t>("local_window_size", 4);
            }
            if (auxiliary) {
              tester.AddAttribute<std::string>("selected_kv_source", "auxiliary");
              tester.AddAttribute<int64_t>("auxiliary_kv_shared", shared_auxiliary ? 1 : 0);
            }
            tester.AddInput<MLFloat16>("query", {rows, kHeads * kChannels}, query);
            tester.AddInput<MLFloat16>("key", {rows, kKvHeads * kChannels}, key);
            tester.AddInput<MLFloat16>("value", {rows, kKvHeads * kChannels}, value);
            tester.AddInput<MLFloat16>("key_cache", {kBlocks, kBlockSize, kKvHeads, kChannels}, cached_key);
            tester.AddInput<MLFloat16>("value_cache", {kBlocks, kBlockSize, kKvHeads, kChannels}, cached_value);
            tester.AddInput<int32_t>("cumulative_sequence_length", {batches + 1}, cumulative);
            tester.AddInput<int32_t>("past_seqlens", {batches}, past);
            tester.AddInput<int32_t>("block_table", {batches, 3}, table);
            tester.AddInput<int32_t>("slot_mapping", {rows}, slots);
            tester.AddInput<int32_t>("selected_indices", {rows, kSelected}, indices);
            tester.AddInput<int32_t>("selected_counts", {rows}, counts);
            if (auxiliary) {
              tester.AddInput<MLFloat16>("auxiliary_key", {batches, kAuxiliaryCapacity, kKvHeads, kChannels}, auxiliary_key);
              if (shared_auxiliary) {
                tester.AddOptionalInputEdge<MLFloat16>();
              } else {
                tester.AddInput<MLFloat16>("auxiliary_value", {batches, kAuxiliaryCapacity, kKvHeads, kChannels}, auxiliary_value);
              }
              tester.AddInput<int32_t>("auxiliary_lengths", {batches}, std::vector<int32_t>(batches, kAuxiliaryCapacity - 2));
            } else {
              tester.AddOptionalInputEdge<MLFloat16>();
              tester.AddOptionalInputEdge<MLFloat16>();
              tester.AddOptionalInputEdge<int32_t>();
            }
            tester.AddOptionalInputEdge<MLFloat16>();
            tester.AddOptionalInputEdge<MLFloat16>();
            tester.AddInput<MLFloat16>("head_sink", {kHeads}, HalfVector(0.25f, kHeads));
            tester.AddOutput<MLFloat16>("output", {rows, kHeads * kChannels}, expected);
            tester.SetOutputTolerance(0.002f, 0.002f);
            bool verified = false;
            tester.SetCustomOutputVerifier([&](const std::vector<OrtValue>& fetches, const std::string&) {
              verified = true;
              ASSERT_EQ(fetches.size(), 1u);
              const auto* actual = fetches[0].Get<Tensor>().Data<MLFloat16>();
              for (size_t channel = 0; channel < expected.size(); ++channel) {
                ASSERT_NEAR(actual[channel].ToFloat(), expected[channel].ToFloat(),
                            0.002f + 0.002f * std::abs(expected[channel].ToFloat()));
              }
              if (arm == 0) {
                baseline_outputs.emplace_back(actual, actual + expected.size());
              } else {
                ASSERT_LT(case_index, baseline_outputs.size());
                for (size_t channel = 0; channel < expected.size(); ++channel) {
                  ASSERT_EQ(actual[channel].val, baseline_outputs[case_index][channel].val)
                      << "case=" << case_index << " output element=" << channel;
                }
              }
            });
            RunCuda(tester, options);
            ASSERT_TRUE(verified);
            ASSERT_FALSE(::testing::Test::HasFailure());
            ++case_index;
          }
        }
      }
    }
  }
}

static void RunGroupedAliasedCacheAppendTest(int rows, bool unaligned = false) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }
  const int kRows = rows;
  constexpr int kHeads = 24;
  constexpr int kKvHeads = 2;
  constexpr int kChannels = 256;
  constexpr int kBlocks = 4;
  constexpr int kPast = 15;
  const std::vector<int32_t> table{2, 0, 3};
  const std::vector<int64_t> cache_shape{kBlocks, kBlockSize, kKvHeads, kChannels};
  const std::vector<MLFloat16> query(kRows * kHeads * kChannels, MLFloat16(0.0f));
  std::vector<MLFloat16> key(kRows * kKvHeads * kChannels);
  std::vector<MLFloat16> value(key.size());
  std::vector<MLFloat16> expected_output(query.size());
  const std::vector<MLFloat16> cached_key(kBlocks * kBlockSize * kKvHeads * kChannels, MLFloat16(-0.5f));
  const std::vector<MLFloat16> cached_value(cached_key.size(), MLFloat16(-0.25f));
  auto expected_key = cached_key;
  auto expected_value = cached_value;
  std::vector<int32_t> slots(kRows);
  const int selected_count = unaligned ? 2 : 1;
  std::vector<int32_t> selected(kRows * selected_count);
  for (int row = 0; row < kRows; ++row) {
    const int position = kPast + row;
    slots[row] = table[position / kBlockSize] * kBlockSize + position % kBlockSize;
    selected[row * selected_count] = position;
    if (unaligned) {
      selected[row * selected_count + 1] = kPast - 1;
    }
    for (int head = 0; head < kKvHeads; ++head) {
      for (int channel = 0; channel < kChannels; ++channel) {
        const int source = (row * kKvHeads + head) * kChannels + channel;
        const int destination = (slots[row] * kKvHeads + head) * kChannels + channel;
        key[source] = MLFloat16(0.5f + row * 0.03125f + channel * 0.00390625f);
        value[source] = MLFloat16(1.0f + row * 0.03125f + head * 0.25f);
        expected_key[destination] = key[source];
        expected_value[destination] = value[source];
      }
    }
    for (int head = 0; head < kHeads; ++head) {
      std::copy_n(value.begin() + (row * kKvHeads + head / 12) * kChannels, kChannels,
                  expected_output.begin() + (row * kHeads + head) * kChannels);
      if (unaligned) {
        for (int channel = 0; channel < kChannels; ++channel) {
          auto& element = expected_output[(row * kHeads + head) * kChannels + channel];
          element = MLFloat16((element.ToFloat() - 0.25f) * 0.5f);
        }
      }
    }
  }
  for (const auto& [grouped, vectorized] :
       std::vector<std::tuple<bool, bool>>{{false, false}, {true, false}, {true, true}}) {
    ScopedEnvironmentVariables scoped_env(EnvVarMap{{"ORT_SPARSE_PAGED_ATTENTION_GROUPED", grouped ? "1" : "0"}});
    OpTester tester("SparsePagedAttention", 1, kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", kHeads);
    tester.AddAttribute<int64_t>("kv_num_heads", kKvHeads);
    tester.AddInput<MLFloat16>("query", {kRows, kHeads * kChannels}, query);
    tester.AddInput<MLFloat16>("key", {kRows, kKvHeads * kChannels}, key);
    tester.AddInput<MLFloat16>("value", {kRows, kKvHeads * kChannels}, value);
    tester.AddInput<MLFloat16>("key_cache", cache_shape, cached_key);
    tester.AddInput<MLFloat16>("value_cache", cache_shape, cached_value);
    tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, kRows});
    tester.AddInput<int32_t>("past_seqlens", {1}, {kPast});
    tester.AddInput<int32_t>("block_table", {1, 3}, table);
    tester.AddInput<int32_t>("slot_mapping", {kRows}, slots);
    tester.AddInput<int32_t>("selected_indices", {kRows, selected_count}, selected);
    tester.AddInput<int32_t>("selected_counts", {kRows}, std::vector<int32_t>(kRows, selected_count));
    tester.AddOutput<MLFloat16>("output", {kRows, kHeads * kChannels}, expected_output);
    tester.AddOutput<MLFloat16>("key_cache_out", cache_shape, expected_key);
    tester.AddOutput<MLFloat16>("value_cache_out", cache_shape, expected_value);
    std::string serialized;
    auto& model = tester.BuildModel();
    ASSERT_STATUS_OK(model.MainGraph().Resolve());
    ASSERT_TRUE(model.ToProto().SerializeToString(&serialized));
    std::stringstream model_stream(serialized);
    SessionOptions options;
    options.session_logid = "SparsePagedAttentionAliasedCacheTest";
    if (vectorized) {
      ASSERT_STATUS_OK(options.config_options.AddConfigEntry("ep.cuda.sparse_paged_attention_grouped_vectorized", "1"));
    }
    InferenceSession session(options, GetEnvironment());
    auto provider = DefaultCudaExecutionProvider();
    ASSERT_NE(provider, nullptr);
    auto* provider_ptr = provider.get();
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(provider)));
    const auto allocators = provider_ptr->CreatePreferredAllocators();
    const OrtMemoryInfo* device_info = nullptr;
    for (const auto& allocator : allocators) {
      if (allocator->Info().device.Type() == OrtDevice::GPU && allocator->Info().mem_type == OrtMemTypeDefault) {
        device_info = &allocator->Info();
      }
    }
    ASSERT_NE(device_info, nullptr);
    ASSERT_STATUS_OK(session.Load(model_stream));
    ASSERT_STATUS_OK(session.Initialize());
    auto device_allocator = session.GetAllocator(*device_info);
    ASSERT_NE(device_allocator, nullptr);
    auto cpu_allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
    std::vector<OrtValue> backing_allocations;
    auto make_gpu = [&](const auto& data, const TensorShape& shape) {
      using Element = typename std::decay_t<decltype(data)>::value_type;
      if constexpr (std::is_same_v<Element, MLFloat16>) {
        if (unaligned) {
          std::vector<Element> padded(data.size() + 1, MLFloat16(-8.0f));
          std::copy(data.begin(), data.end(), padded.begin() + 1);
          const TensorShape allocation_shape({shape.Size() + 1});
          Tensor cpu_tensor(DataTypeImpl::GetType<Element>(), allocation_shape, padded.data(), cpu_allocator->Info());
          Tensor gpu_tensor(DataTypeImpl::GetType<Element>(), allocation_shape, device_allocator);
          ORT_THROW_IF_ERROR(provider_ptr->GetDataTransfer()->CopyTensor(cpu_tensor, gpu_tensor));
          auto* pointer = gpu_tensor.MutableData<Element>() + 1;
          ORT_ENFORCE(reinterpret_cast<std::uintptr_t>(pointer) % 4 == 2);
          Tensor view(DataTypeImpl::GetType<Element>(), shape, pointer, device_allocator->Info());
          OrtValue allocation;
          Tensor::InitOrtValue(std::move(gpu_tensor), allocation);
          backing_allocations.push_back(std::move(allocation));
          OrtValue result;
          Tensor::InitOrtValue(std::move(view), result);
          return result;
        }
      }
      Tensor cpu_tensor(DataTypeImpl::GetType<Element>(), shape, const_cast<Element*>(data.data()), cpu_allocator->Info());
      Tensor gpu_tensor(DataTypeImpl::GetType<Element>(), shape, device_allocator);
      ORT_THROW_IF_ERROR(provider_ptr->GetDataTransfer()->CopyTensor(cpu_tensor, gpu_tensor));
      OrtValue result;
      Tensor::InitOrtValue(std::move(gpu_tensor), result);
      return result;
    };
    std::unique_ptr<IOBinding> binding;
    ASSERT_STATUS_OK(session.NewIOBinding(&binding));
    auto key_cache = make_gpu(cached_key, TensorShape(cache_shape));
    auto value_cache = make_gpu(cached_value, TensorShape(cache_shape));
    ASSERT_STATUS_OK(binding->BindInput("query", make_gpu(query, TensorShape({kRows, kHeads * kChannels}))));
    ASSERT_STATUS_OK(binding->BindInput("key", make_gpu(key, TensorShape({kRows, kKvHeads * kChannels}))));
    ASSERT_STATUS_OK(binding->BindInput("value", make_gpu(value, TensorShape({kRows, kKvHeads * kChannels}))));
    ASSERT_STATUS_OK(binding->BindInput("key_cache", key_cache));
    ASSERT_STATUS_OK(binding->BindInput("value_cache", value_cache));
    ASSERT_STATUS_OK(binding->BindInput("cumulative_sequence_length", make_gpu(std::vector<int32_t>{0, kRows}, TensorShape({2}))));
    ASSERT_STATUS_OK(binding->BindInput("past_seqlens", make_gpu(std::vector<int32_t>{kPast}, TensorShape({1}))));
    ASSERT_STATUS_OK(binding->BindInput("block_table", make_gpu(table, TensorShape({1, 3}))));
    ASSERT_STATUS_OK(binding->BindInput("slot_mapping", make_gpu(slots, TensorShape({kRows}))));
    ASSERT_STATUS_OK(binding->BindInput("selected_indices", make_gpu(selected, TensorShape({kRows, selected_count}))));
    ASSERT_STATUS_OK(binding->BindInput("selected_counts", make_gpu(std::vector<int32_t>(kRows, selected_count), TensorShape({kRows}))));
    ASSERT_STATUS_OK(binding->BindOutput("output", device_info->device));
    ASSERT_STATUS_OK(binding->BindOutput("key_cache_out", key_cache));
    ASSERT_STATUS_OK(binding->BindOutput("value_cache_out", value_cache));
    ASSERT_STATUS_OK(session.Run(RunOptions(), *binding));
    ASSERT_STATUS_OK(binding->SynchronizeOutputs());
    const auto& outputs = binding->GetOutputs();
    ASSERT_EQ(outputs.size(), 3u);
    ASSERT_EQ(outputs[1].Get<Tensor>().Data<MLFloat16>(), key_cache.Get<Tensor>().Data<MLFloat16>());
    ASSERT_EQ(outputs[2].Get<Tensor>().Data<MLFloat16>(), value_cache.Get<Tensor>().Data<MLFloat16>());
    const std::vector<const std::vector<MLFloat16>*> expected{&expected_output, &expected_key, &expected_value};
    for (size_t output_index = 0; output_index < outputs.size(); ++output_index) {
      const auto& actual = outputs[output_index].Get<Tensor>();
      Tensor host(DataTypeImpl::GetType<MLFloat16>(), actual.Shape(), cpu_allocator);
      ASSERT_STATUS_OK(provider_ptr->GetDataTransfer()->CopyTensor(actual, host));
      ASSERT_EQ(host.Shape().Size(), static_cast<int64_t>(expected[output_index]->size()));
      for (size_t element = 0; element < expected[output_index]->size(); ++element) {
        ASSERT_EQ(host.Data<MLFloat16>()[element].val, (*expected[output_index])[element].val)
            << "grouped=" << grouped << " vectorized=" << vectorized
            << " output=" << output_index << " element=" << element;
      }
    }
  }
}

TEST(SparsePagedAttention, Cuda_GroupedAliasedCacheAppendMatchesReference) {
  for (const int rows : {1, 2, 3, 4, 6, 8, 32}) {
    SCOPED_TRACE(rows);
    RunGroupedAliasedCacheAppendTest(rows);
  }
}

TEST(SparsePagedAttention, Cuda_GroupedUnalignedAliasedCacheAppendMatchesReference) {
  RunGroupedAliasedCacheAppendTest(32, true);
}

TEST(SparsePagedAttention, Cuda_SelectedMainWritesAndReadsPagedCache) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddSingleTokenPrefix(tester, HalfVector(0.0f), HalfVector(0.0f), HalfVector(2.0f), {0, -1}, 1);
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize}, HalfVector(2.0f));
  RunCuda(tester);
}

TEST(SparsePagedAttention, Cuda_CurrentPrefillReadsContiguousKeyValue) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddAttributes(tester);
  tester.AddInput<MLFloat16>("query", {2, kHeadSize}, HalfVector(0.0f, 2 * kHeadSize));
  tester.AddInput<MLFloat16>("key", {2, kHeadSize}, HalfVector(0.0f, 2 * kHeadSize));
  auto values = HalfVector(2.0f);
  const auto second_value = HalfVector(4.0f);
  values.insert(values.end(), second_value.begin(), second_value.end());
  tester.AddInput<MLFloat16>("value", {2, kHeadSize}, values);
  auto cached_values = HalfVector(0.0f, kCacheElementCount);
  std::fill_n(cached_values.begin(), kHeadSize, MLFloat16(9.0f));
  std::fill_n(cached_values.begin() + kHeadSize, kHeadSize, MLFloat16(8.0f));
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize},
                             HalfVector(0.0f, kCacheElementCount));
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize}, cached_values);
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 2});
  tester.AddInput<int32_t>("past_seqlens", {1}, {0});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  tester.AddInput<int32_t>("slot_mapping", {2}, {2, 3});
  tester.AddInput<int32_t>("selected_indices", {2, 2}, {0, -1, 0, 1});
  tester.AddInput<int32_t>("selected_counts", {2}, {1, 2});
  auto expected = HalfVector(2.0f);
  const auto second_expected = HalfVector(3.0f);
  expected.insert(expected.end(), second_expected.begin(), second_expected.end());
  tester.AddOutput<MLFloat16>("output", {2, kHeadSize}, expected);
  RunCuda(tester);
}

TEST(SparsePagedAttention, Cuda_LocalAndSharedAuxiliaryUseJointSoftmax) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  std::vector<MLFloat16> query = HalfVector(0.0f);
  query[0] = MLFloat16(1.0f);
  std::vector<MLFloat16> auxiliary_key = HalfVector(0.0f);
  auxiliary_key[0] = MLFloat16(std::log(3.0f));

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddSingleTokenPrefix(tester, query, HalfVector(0.0f), HalfVector(1.0f), {0}, 1);
  tester.AddAttribute<float>("scale", 1.0f);
  tester.AddAttribute<std::string>("attention_mode", "local_plus_selected");
  tester.AddAttribute<std::string>("selected_kv_source", "auxiliary");
  tester.AddAttribute<int64_t>("auxiliary_kv_shared", 1);
  tester.AddAttribute<int64_t>("local_window_size", 1);
  tester.AddInput<MLFloat16>("auxiliary_key", {1, 1, 1, kHeadSize}, auxiliary_key);
  tester.AddOptionalInputEdge<MLFloat16>();  // auxiliary_value
  tester.AddInput<int32_t>("auxiliary_lengths", {1}, {1});
  auto expected = HalfVector(0.25f);
  expected[0] = MLFloat16((1.0f + 3.0f * std::log(3.0f)) / 4.0f);
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize}, expected, false, 0.002f, 0.002f);
  RunCuda(tester);
}

TEST(SparsePagedAttention, Cuda_NonSharedAuxiliaryUsesSeparateValue) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  std::vector<MLFloat16> query = HalfVector(0.0f);
  query[0] = MLFloat16(1.0f);
  std::vector<MLFloat16> auxiliary_key = HalfVector(0.0f);
  auxiliary_key[0] = MLFloat16(std::log(3.0f));

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddSingleTokenPrefix(tester, query, HalfVector(0.0f), HalfVector(1.0f), {0}, 1);
  tester.AddAttribute<float>("scale", 1.0f);
  tester.AddAttribute<std::string>("attention_mode", "local_plus_selected");
  tester.AddAttribute<std::string>("selected_kv_source", "auxiliary");
  tester.AddAttribute<int64_t>("local_window_size", 1);
  tester.AddInput<MLFloat16>("auxiliary_key", {1, 1, 1, kHeadSize}, auxiliary_key);
  tester.AddInput<MLFloat16>("auxiliary_value", {1, 1, 1, kHeadSize}, HalfVector(5.0f));
  tester.AddInput<int32_t>("auxiliary_lengths", {1}, {1});
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize}, HalfVector(4.0f), false, 0.002f, 0.002f);
  RunCuda(tester);
}

TEST(SparsePagedAttention, Cuda_HeadSinkContributesOnlyToSoftmaxDenominator) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddSingleTokenPrefix(tester, HalfVector(0.0f), HalfVector(0.0f), HalfVector(2.0f), {0}, 1);
  tester.AddOptionalInputEdge<MLFloat16>();  // auxiliary_key
  tester.AddOptionalInputEdge<MLFloat16>();  // auxiliary_value
  tester.AddOptionalInputEdge<int32_t>();    // auxiliary_lengths
  tester.AddOptionalInputEdge<MLFloat16>();  // cos_cache
  tester.AddOptionalInputEdge<MLFloat16>();  // sin_cache
  tester.AddInput<MLFloat16>("head_sink", {1}, {MLFloat16(std::log(3.0f))});
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize}, HalfVector(0.5f), false, 0.002f, 0.002f);
  RunCuda(tester);
}

TEST(SparsePagedAttention, Cuda_RejectsNumHeadsAboveIntMax) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(std::numeric_limits<int>::max()) + 1);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddInput<MLFloat16>("query", {1, kHeadSize}, HalfVector(0.0f));
  tester.AddInput<MLFloat16>("key", {1, kHeadSize}, HalfVector(0.0f));
  tester.AddInput<MLFloat16>("value", {1, kHeadSize}, HalfVector(1.0f));
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize},
                             HalfVector(0.0f, kCacheElementCount));
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize},
                             HalfVector(0.0f, kCacheElementCount));
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 1});
  tester.AddInput<int32_t>("past_seqlens", {1}, {0});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  tester.AddInput<int32_t>("slot_mapping", {1}, {0});
  tester.AddInput<int32_t>("selected_indices", {1, 1}, {0});
  tester.AddInput<int32_t>("selected_counts", {1}, {1});
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize}, HalfVector(0.0f));

  auto cuda_ep = DefaultCudaExecutionProvider();
  ASSERT_NE(cuda_ep, nullptr);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  tester.Run(OpTester::ExpectResult::kExpectFailure, "num_heads must not exceed INT_MAX",
             {}, nullptr, &execution_providers);
}

TEST(SparsePagedAttention, Cuda_RejectsNonDivisibleQueryWidth) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", 2);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddInput<MLFloat16>("query", {1, 2 * kHeadSize + 1}, HalfVector(0.0f, 2 * kHeadSize + 1));
  tester.AddInput<MLFloat16>("key", {1, kHeadSize}, HalfVector(0.0f));
  tester.AddInput<MLFloat16>("value", {1, kHeadSize}, HalfVector(0.0f));
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize},
                             HalfVector(0.0f, kCacheElementCount));
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize},
                             HalfVector(0.0f, kCacheElementCount));
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 1});
  tester.AddInput<int32_t>("past_seqlens", {1}, {0});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  tester.AddInput<int32_t>("slot_mapping", {1}, {0});
  tester.AddInput<int32_t>("selected_indices", {1, 1}, {0});
  tester.AddInput<int32_t>("selected_counts", {1}, {1});
  tester.AddOutput<MLFloat16>("output", {1, 2 * kHeadSize + 1}, HalfVector(0.0f, 2 * kHeadSize + 1));

  auto cuda_ep = DefaultCudaExecutionProvider();
  ASSERT_NE(cuda_ep, nullptr);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "Input 'query' hidden size must be a multiple of num_heads",
             {}, nullptr, &execution_providers);
}

TEST(SparsePagedAttention, Cuda_RejectsNonDivisiblePackedQkvWidth) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddAttributes(tester);
  tester.AddInput<MLFloat16>("query", {1, 3 * kHeadSize + 1}, HalfVector(0.0f, 3 * kHeadSize + 1));
  tester.AddOptionalInputEdge<MLFloat16>();  // key
  tester.AddOptionalInputEdge<MLFloat16>();  // value
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize},
                             HalfVector(0.0f, kCacheElementCount));
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize},
                             HalfVector(0.0f, kCacheElementCount));
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 1});
  tester.AddInput<int32_t>("past_seqlens", {1}, {0});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  tester.AddInput<int32_t>("slot_mapping", {1}, {0});
  tester.AddInput<int32_t>("selected_indices", {1, 1}, {0});
  tester.AddInput<int32_t>("selected_counts", {1}, {1});
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize}, HalfVector(0.0f));

  auto cuda_ep = DefaultCudaExecutionProvider();
  ASSERT_NE(cuda_ep, nullptr);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(cuda_ep));
  tester.Run(OpTester::ExpectResult::kExpectFailure,
             "Hidden size must be divisible by (num_heads + 2 * kv_num_heads)",
             {}, nullptr, &execution_providers);
}

TEST(SparsePagedAttention, Cuda_InvalidSelectedMainIndicesAreIgnored) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  for (const auto& test_case : std::vector<std::pair<int32_t, int32_t>>{{-1, 0}, {kBlockSize, 0}, {0, -1}}) {
    OpTester tester("SparsePagedAttention", 1, kMSDomain);
    AddSingleTokenPrefix(tester, HalfVector(0.0f), HalfVector(0.0f), HalfVector(7.0f),
                         {test_case.first}, 1, 0, test_case.second);
    tester.AddOutput<MLFloat16>("output", {1, kHeadSize}, HalfVector(0.0f));
    RunCuda(tester);
  }
}

TEST(SparsePagedAttention, Cuda_NonCausalSelectedMainIndexIsIgnored) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddAttributes(tester);
  tester.AddInput<MLFloat16>("query", {2, kHeadSize}, HalfVector(0.0f, 2 * kHeadSize));
  tester.AddInput<MLFloat16>("key", {2, kHeadSize}, HalfVector(0.0f, 2 * kHeadSize));
  auto values = HalfVector(1.0f);
  const auto future_values = HalfVector(9.0f);
  values.insert(values.end(), future_values.begin(), future_values.end());
  tester.AddInput<MLFloat16>("value", {2, kHeadSize}, values);
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize},
                             HalfVector(0.0f, kCacheElementCount));
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize},
                             HalfVector(0.0f, kCacheElementCount));
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 2});
  tester.AddInput<int32_t>("past_seqlens", {1}, {0});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  tester.AddInput<int32_t>("slot_mapping", {2}, {0, 1});
  tester.AddInput<int32_t>("selected_indices", {2, 2}, {1, 0, 1, -1});
  tester.AddInput<int32_t>("selected_counts", {2}, {2, 1});
  auto expected = HalfVector(1.0f);
  expected.insert(expected.end(), future_values.begin(), future_values.end());
  tester.AddOutput<MLFloat16>("output", {2, kHeadSize}, expected);
  RunCuda(tester);
}

TEST(SparsePagedAttention, Cuda_LocalWindowAndSelectedMainUseSetUnion) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  auto cached_values = HalfVector(0.0f, kCacheElementCount);
  std::fill_n(cached_values.begin(), kHeadSize, MLFloat16(3.0f));
  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddSingleTokenPrefix(tester, HalfVector(0.0f), HalfVector(0.0f), HalfVector(1.0f),
                       {1, 0}, 2, 1, 0, 1, HalfVector(0.0f, kCacheElementCount), cached_values);
  tester.AddAttribute<std::string>("attention_mode", "local_plus_selected");
  tester.AddAttribute<int64_t>("local_window_size", 1);
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize}, HalfVector(2.0f));
  RunCuda(tester);
}

TEST(SparsePagedAttention, Cuda_MergesMultipleSelectedMainSplits) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  constexpr int kPastLength = 128;
  constexpr int kSelectedCount = kPastLength + 1;
  constexpr int kNumBlocks = (kSelectedCount + kBlockSize - 1) / kBlockSize;
  std::vector<int32_t> selected_indices(kSelectedCount);
  std::iota(selected_indices.begin(), selected_indices.end(), 0);
  std::vector<int32_t> block_table(kNumBlocks);
  std::iota(block_table.begin(), block_table.end(), 0);

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddAttributes(tester);
  tester.AddInput<MLFloat16>("query", {1, kHeadSize}, HalfVector(0.0f));
  tester.AddInput<MLFloat16>("key", {1, kHeadSize}, HalfVector(0.0f));
  tester.AddInput<MLFloat16>("value", {1, kHeadSize}, HalfVector(3.0f));
  tester.AddInput<MLFloat16>("key_cache", {kNumBlocks, kBlockSize, 1, kHeadSize},
                             HalfVector(0.0f, kNumBlocks * kCacheElementCount));
  tester.AddInput<MLFloat16>("value_cache", {kNumBlocks, kBlockSize, 1, kHeadSize},
                             HalfVector(1.0f, kNumBlocks * kCacheElementCount));
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 1});
  tester.AddInput<int32_t>("past_seqlens", {1}, {kPastLength});
  tester.AddInput<int32_t>("block_table", {1, kNumBlocks}, block_table);
  tester.AddInput<int32_t>("slot_mapping", {1}, {kPastLength});
  tester.AddInput<int32_t>("selected_indices", {1, kSelectedCount}, selected_indices);
  tester.AddInput<int32_t>("selected_counts", {1}, {kSelectedCount});
  const float expected_value = static_cast<float>(kPastLength + 3) / kSelectedCount;
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize}, HalfVector(expected_value),
                              false, 0.002f, 0.002f);
  RunCuda(tester);
}

class SparsePagedAttentionInt8Test : public ::testing::TestWithParam<std::pair<std::string, std::string>> {};

TEST_P(SparsePagedAttentionInt8Test, Cuda_QuantizedMainCache) {
  if (DefaultCudaExecutionProvider() == nullptr) {
    GTEST_SKIP() << "CUDA EP not available.";
  }

  const auto& [k_quant_type, v_quant_type] = GetParam();
  std::vector<float> k_scales(k_quant_type == "PER_CHANNEL" ? kHeadSize : 1, 0.5f);
  std::vector<float> v_scales(v_quant_type == "PER_CHANNEL" ? kHeadSize : 1, 0.5f);
  if (k_quant_type == "PER_CHANNEL") {
    for (int i = 0; i < kHeadSize; ++i) {
      k_scales[i] = 0.25f * static_cast<float>(i + 1);
    }
  }
  if (v_quant_type == "PER_CHANNEL") {
    for (int i = 0; i < kHeadSize; ++i) {
      v_scales[i] = 0.25f * static_cast<float>(i + 1);
    }
  }
  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  const auto expected = AddQuantizedInputs(tester, k_quant_type, v_quant_type, k_scales, v_scales);
  tester.AddOutput<MLFloat16>("output", {2, kHeadSize}, expected, false, 0.002f, 0.002f);
  RunCuda(tester);
}

INSTANTIATE_TEST_SUITE_P(
    QuantizationGranularity, SparsePagedAttentionInt8Test,
    ::testing::Values(std::make_pair("PER_TENSOR", "PER_TENSOR"),
                      std::make_pair("PER_CHANNEL", "PER_CHANNEL"),
                      std::make_pair("PER_CHANNEL", "PER_TENSOR")));

TEST(SparsePagedAttention, WebGpu_SelectedMainWritesAndReadsPagedCache) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddCommonInputs(tester, 2.0f);
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize},
                              std::vector<MLFloat16>(kHeadSize, MLFloat16(2.0f)));
  tester.SetOutputTolerance(0.01f);
  RunWebGpu(tester);
}

TEST(SparsePagedAttention, WebGpu_SelectedAuxiliaryWritesDirectOutput) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddCommonInputs(tester, 0.0f, {0, 1}, 2);
  tester.AddAttribute<std::string>("selected_kv_source", "auxiliary");
  tester.AddInput<MLFloat16>("auxiliary_key", {1, 2, 1, kHeadSize}, HalfVector(0.0f, 2 * kHeadSize));
  auto auxiliary_value = HalfVector(3.0f);
  const auto second_value = HalfVector(7.0f);
  auxiliary_value.insert(auxiliary_value.end(), second_value.begin(), second_value.end());
  tester.AddInput<MLFloat16>("auxiliary_value", {1, 2, 1, kHeadSize}, auxiliary_value);
  tester.AddInput<int32_t>("auxiliary_lengths", {1}, {2});
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize}, HalfVector(5.0f));
  tester.SetOutputTolerance(0.01f);
  RunWebGpu(tester);
}

TEST(SparsePagedAttention, WebGpu_PackedQkvRotaryAndBlockTableScatter) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  constexpr int kNumBlocks = 2;
  constexpr int kRotaryHeadSize = 16;
  constexpr int kRotaryCacheElems = kBlockSize * kRotaryHeadSize;
  constexpr int kPackedSize = 3 * kRotaryHeadSize;
  std::vector<MLFloat16> packed_qkv(kPackedSize, MLFloat16(0.0f));
  std::fill_n(packed_qkv.begin() + 2 * kRotaryHeadSize, kRotaryHeadSize, MLFloat16(4.0f));
  std::vector<MLFloat16> expected_value_cache(kNumBlocks * kRotaryCacheElems, MLFloat16(0.0f));
  SetConstantCacheRow(expected_value_cache, kBlockSize, kRotaryHeadSize, 4.0f);

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddAttributes(tester);
  tester.AddAttribute<int64_t>("do_rotary", 1);
  tester.AddInput<MLFloat16>("query", {1, kPackedSize}, packed_qkv);
  tester.AddOptionalInputEdge<MLFloat16>();  // key is packed with query
  tester.AddOptionalInputEdge<MLFloat16>();  // value is packed with query
  tester.AddInput<MLFloat16>("key_cache", {kNumBlocks, kBlockSize, 1, kRotaryHeadSize},
                             HalfVector(0.0f, kNumBlocks * kRotaryCacheElems));
  tester.AddInput<MLFloat16>("value_cache", {kNumBlocks, kBlockSize, 1, kRotaryHeadSize},
                             HalfVector(0.0f, kNumBlocks * kRotaryCacheElems));
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 1});
  tester.AddInput<int32_t>("past_seqlens", {1}, {0});
  tester.AddInput<int32_t>("block_table", {1, 1}, {1});
  tester.AddOptionalInputEdge<int32_t>();  // derive the cache slot from block_table
  tester.AddInput<int32_t>("selected_indices", {1, 1}, {0});
  tester.AddInput<int32_t>("selected_counts", {1}, {1});
  tester.AddOptionalInputEdge<MLFloat16>();  // auxiliary_key
  tester.AddOptionalInputEdge<MLFloat16>();  // auxiliary_value
  tester.AddOptionalInputEdge<int32_t>();    // auxiliary_lengths
  tester.AddInput<MLFloat16>("cos_cache", {1, kRotaryHeadSize / 2},
                             HalfVector(1.0f, kRotaryHeadSize / 2));
  tester.AddInput<MLFloat16>("sin_cache", {1, kRotaryHeadSize / 2},
                             HalfVector(0.0f, kRotaryHeadSize / 2));
  tester.AddOutput<MLFloat16>("output", {1, kRotaryHeadSize},
                              HalfVector(4.0f, kRotaryHeadSize));
  tester.AddOutput<MLFloat16>("key_cache_out", {kNumBlocks, kBlockSize, 1, kRotaryHeadSize},
                              HalfVector(0.0f, kNumBlocks * kRotaryCacheElems));
  tester.AddOutput<MLFloat16>("value_cache_out", {kNumBlocks, kBlockSize, 1, kRotaryHeadSize},
                              expected_value_cache);
  tester.SetOutputTolerance(0.01f);
  RunWebGpu(tester);
}

TEST(SparsePagedAttention, WebGpu_SinkSoftcapAndNonCausalSelection) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  std::vector<MLFloat16> query(2 * kHeadSize, MLFloat16(0.0f));
  query[0] = MLFloat16(1.0f);
  query[kHeadSize] = MLFloat16(1.0f);
  std::vector<MLFloat16> key(2 * kHeadSize, MLFloat16(0.0f));
  key[kHeadSize] = MLFloat16(2.0f);
  std::vector<MLFloat16> value;
  value.insert(value.end(), kHeadSize, MLFloat16(1.0f));
  value.insert(value.end(), kHeadSize, MLFloat16(3.0f));

  const float capped_future_logit = std::tanh(2.0f);
  const float denominator = 1.0f + std::exp(capped_future_logit) + std::exp(0.5f);
  const float expected_value = (1.0f + 3.0f * std::exp(capped_future_logit)) / denominator;

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddAttributes(tester);
  tester.AddAttribute<float>("scale", 1.0f);
  tester.AddAttribute<float>("softcap", 1.0f);
  tester.AddAttribute<int64_t>("is_causal", 0);
  tester.AddInput<MLFloat16>("query", {2, kHeadSize}, query);
  tester.AddInput<MLFloat16>("key", {2, kHeadSize}, key);
  tester.AddInput<MLFloat16>("value", {2, kHeadSize}, value);
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize},
                             HalfVector(0.0f, kCacheElems));
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize},
                             HalfVector(0.0f, kCacheElems));
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 2});
  tester.AddInput<int32_t>("past_seqlens", {1}, {0});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  tester.AddInput<int32_t>("slot_mapping", {2}, {0, 1});
  tester.AddInput<int32_t>("selected_indices", {2, 2}, {0, 1, 0, 1});
  tester.AddInput<int32_t>("selected_counts", {2}, {2, 2});
  tester.AddOptionalInputEdge<MLFloat16>();  // auxiliary_key
  tester.AddOptionalInputEdge<MLFloat16>();  // auxiliary_value
  tester.AddOptionalInputEdge<int32_t>();    // auxiliary_lengths
  tester.AddOptionalInputEdge<MLFloat16>();  // cos_cache
  tester.AddOptionalInputEdge<MLFloat16>();  // sin_cache
  tester.AddInput<MLFloat16>("head_sink", {1}, {MLFloat16(0.5f)});
  tester.AddOutput<MLFloat16>("output", {2, kHeadSize},
                              HalfVector(expected_value, 2 * kHeadSize));
  tester.SetOutputTolerance(0.01f);
  RunWebGpu(tester);
}

// local_plus_selected + selected_kv_source='auxiliary' with a shared auxiliary
// K/V tensor. The local window contributes the main-cache value (1.0) and the
// selection contributes the auxiliary value (3.0). Because both partial states
// are merged into a single softmax, the result is their mean (2.0). Two
// independent softmaxes averaged afterwards would give the same number only by
// coincidence here, so the CUDA reference test uses the same configuration and
// both kernels must agree.
TEST(SparsePagedAttention, WebGpu_LocalAndSharedAuxiliaryUseJointSoftmax) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddCommonInputs(tester, 1.0f);
  tester.AddAttribute<std::string>("attention_mode", "local_plus_selected");
  tester.AddAttribute<std::string>("selected_kv_source", "auxiliary");
  tester.AddAttribute<int64_t>("auxiliary_kv_shared", 1);
  tester.AddAttribute<int64_t>("local_window_size", 1);
  tester.AddInput<MLFloat16>("auxiliary_key", {1, 1, 1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(3.0f)));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<int32_t>("auxiliary_lengths", {1}, {1});
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize},
                              std::vector<MLFloat16>(kHeadSize, MLFloat16(2.0f)));
  tester.SetOutputTolerance(0.01f);
  RunWebGpu(tester);
}

// A non-negative slot_mapping entry stores the current K/V and the cache
// outputs must observe that store.
TEST(SparsePagedAttention, WebGpu_SlotMappingWritesCacheOutputs) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  std::vector<MLFloat16> expected_value_cache(kCacheElems, MLFloat16(0.0f));
  for (int i = 0; i < kHeadSize; ++i) {
    expected_value_cache[i] = MLFloat16(7.0f);
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddCacheProbeInputs(tester, /*current_value*/ 7.0f, /*cached_value_at_slot0*/ 5.0f,
                      /*slot_mapping_value*/ 0);
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize},
                              std::vector<MLFloat16>(kHeadSize, MLFloat16(7.0f)));
  tester.AddOutput<MLFloat16>("key_cache_out", {1, kBlockSize, 1, kHeadSize},
                              std::vector<MLFloat16>(kCacheElems, MLFloat16(0.0f)));
  tester.AddOutput<MLFloat16>("value_cache_out", {1, kBlockSize, 1, kHeadSize},
                              expected_value_cache);
  tester.SetOutputTolerance(0.01f);
  RunWebGpu(tester);
}

// slot_mapping == -1 suppresses the store. The previously cached value must
// survive in value_cache_out and must be what the selection reads back, proving
// the suppression happens before the write rather than after it.
TEST(SparsePagedAttention, WebGpu_NegativeSlotMappingSuppressesCacheWrite) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  std::vector<MLFloat16> expected_value_cache(kCacheElems, MLFloat16(0.0f));
  for (int i = 0; i < kHeadSize; ++i) {
    expected_value_cache[i] = MLFloat16(5.0f);
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddCacheProbeInputs(tester, /*current_value*/ 7.0f, /*cached_value_at_slot0*/ 5.0f,
                      /*slot_mapping_value*/ -1);
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize},
                              std::vector<MLFloat16>(kHeadSize, MLFloat16(5.0f)));
  tester.AddOutput<MLFloat16>("key_cache_out", {1, kBlockSize, 1, kHeadSize},
                              std::vector<MLFloat16>(kCacheElems, MLFloat16(0.0f)));
  tester.AddOutput<MLFloat16>("value_cache_out", {1, kBlockSize, 1, kHeadSize},
                              expected_value_cache);
  tester.SetOutputTolerance(0.01f);
  RunWebGpu(tester);
}

// Validation: selected_indices must be 2-D (token_count, max_selected_entries).
// A rank-1 tensor is type-correct, so only the kernel can reject it, and it must
// do so explicitly instead of reinterpreting the buffer.
TEST(SparsePagedAttention, WebGpu_RejectsMalformedSelectedIndices) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", 1);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddInput<MLFloat16>("query", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("key", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(1.0f)));
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize},
                             std::vector<MLFloat16>(kCacheElems, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize},
                             std::vector<MLFloat16>(kCacheElems, MLFloat16(0.0f)));
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 1});
  tester.AddInput<int32_t>("past_seqlens", {1}, {0});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  tester.AddInput<int32_t>("slot_mapping", {1}, {0});
  tester.AddInput<int32_t>("selected_indices", {2}, {0, -1});
  tester.AddInput<int32_t>("selected_counts", {1}, {1});
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize},
                              std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  RunWebGpu(tester, OpTester::ExpectResult::kExpectFailure,
            "selected_indices must have shape");
}

// Validation: auxiliary inputs are meaningless when the selection addresses the
// main cache, and are rejected rather than silently ignored.
TEST(SparsePagedAttention, WebGpu_RejectsAuxiliaryInputsWithMainSelection) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  AddCommonInputs(tester, 1.0f);
  tester.AddAttribute<std::string>("selected_kv_source", "main");
  tester.AddInput<MLFloat16>("auxiliary_key", {1, 1, 1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(3.0f)));
  tester.AddOptionalInputEdge<MLFloat16>();
  tester.AddInput<int32_t>("auxiliary_lengths", {1}, {1});
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize},
                              std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  RunWebGpu(tester, OpTester::ExpectResult::kExpectFailure,
            "Auxiliary inputs must be absent");
}

// The distinguishing property of this op is that the main-cache candidates and
// the auxiliary candidates share *one* softmax. This test makes that observable:
// the logits are non-zero and non-uniform, the two sources contribute a
// different number of candidates (3 local vs. 2 auxiliary), and the source with
// the larger logits carries the smaller values. Averaging two independent
// softmaxes would land far away from the joint result, and the test asserts that
// the two references really do differ so it cannot silently stop discriminating.
TEST(SparsePagedAttention, WebGpu_JointSoftmaxDiffersFromIndependentAveraging) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  // scale=1 and a one-hot query turn every logit into the key's first element,
  // so the expected distribution is exact rather than a floating-point artifact
  // of 1/sqrt(head_size).
  std::vector<MLFloat16> query(kHeadSize, MLFloat16(0.0f));
  query[0] = MLFloat16(1.0f);

  std::vector<MLFloat16> key_cache(kCacheElems, MLFloat16(0.0f));
  SetCacheRow(key_cache, 0, kHeadSize, {2.0f});
  SetCacheRow(key_cache, 1, kHeadSize, {1.0f});
  SetCacheRow(key_cache, 2, kHeadSize, {0.0f});

  std::vector<MLFloat16> value_cache(kCacheElems, MLFloat16(0.0f));
  SetConstantCacheRow(value_cache, 0, kHeadSize, 1.0f);
  SetConstantCacheRow(value_cache, 1, kHeadSize, 2.0f);
  SetConstantCacheRow(value_cache, 2, kHeadSize, 3.0f);

  constexpr int kAuxCapacity = 4;
  std::vector<MLFloat16> auxiliary_key(kAuxCapacity * kHeadSize, MLFloat16(0.0f));
  SetCacheRow(auxiliary_key, 0, kHeadSize, {-1.0f});
  SetCacheRow(auxiliary_key, 1, kHeadSize, {-2.0f});
  std::vector<MLFloat16> auxiliary_value(kAuxCapacity * kHeadSize, MLFloat16(0.0f));
  SetConstantCacheRow(auxiliary_value, 0, kHeadSize, 4.0f);
  SetConstantCacheRow(auxiliary_value, 1, kHeadSize, 5.0f);

  const std::vector<float> main_logits{2.0f, 1.0f, 0.0f};
  const std::vector<float> main_values{1.0f, 2.0f, 3.0f};
  const std::vector<float> aux_logits{-1.0f, -2.0f};
  const std::vector<float> aux_values{4.0f, 5.0f};
  std::vector<float> all_logits(main_logits);
  all_logits.insert(all_logits.end(), aux_logits.begin(), aux_logits.end());
  std::vector<float> all_values(main_values);
  all_values.insert(all_values.end(), aux_values.begin(), aux_values.end());

  const float expected = JointSoftmax(all_logits, all_values);
  const float independent_average =
      0.5f * (JointSoftmax(main_logits, main_values) + JointSoftmax(aux_logits, aux_values));
  ASSERT_GT(std::abs(expected - independent_average), 0.5f)
      << "the configuration no longer separates a joint softmax from independent averaging";

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", 1);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddAttribute<float>("scale", 1.0f);
  tester.AddAttribute<std::string>("attention_mode", "local_plus_selected");
  tester.AddAttribute<std::string>("selected_kv_source", "auxiliary");
  tester.AddAttribute<int64_t>("local_window_size", 3);
  tester.AddInput<MLFloat16>("query", {1, kHeadSize}, query);
  tester.AddInput<MLFloat16>("key", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize}, key_cache);
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize}, value_cache);
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 1});
  // The token sits at position 2, so the causal local window of 3 covers the
  // pre-populated cache slots 0..2.
  tester.AddInput<int32_t>("past_seqlens", {1}, {2});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  // -1 keeps the current K/V out of the cache so the attended rows are exactly
  // the ones written above.
  tester.AddInput<int32_t>("slot_mapping", {1}, {-1});
  tester.AddInput<int32_t>("selected_indices", {1, 3}, {0, 1, -1});
  tester.AddInput<int32_t>("selected_counts", {1}, {2});
  tester.AddInput<MLFloat16>("auxiliary_key", {1, kAuxCapacity, 1, kHeadSize}, auxiliary_key);
  tester.AddInput<MLFloat16>("auxiliary_value", {1, kAuxCapacity, 1, kHeadSize}, auxiliary_value);
  tester.AddInput<int32_t>("auxiliary_lengths", {1}, {2});
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize},
                              std::vector<MLFloat16>(kHeadSize, MLFloat16(expected)));
  tester.SetOutputTolerance(0.01f);
  RunWebGpu(tester);
}

// Two requests packed into one call, two query heads sharing one KV head, and a
// per-token selection. The two heads read the same cache rows through different
// query vectors, so a broken GQA head mapping or a broken request lookup shows
// up as a wrong value rather than as a shape error.
TEST(SparsePagedAttention, WebGpu_GroupedQueryMultiRequestSelection) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  constexpr int kGqaHeadSize = 8;
  constexpr int kNumHeads = 2;
  constexpr int kNumBlocks = 2;
  constexpr int kGqaCacheElems = kNumBlocks * kBlockSize * kGqaHeadSize;

  // head 0 reads key element 0, head 1 reads key element 1.
  const std::vector<MLFloat16> query{
      MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f),
      MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f),  // token 0, head 0
      MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(0.0f),
      MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f),  // token 0, head 1
      MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f),
      MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f),  // token 1, head 0
      MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(0.0f),
      MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f),  // token 1, head 1
      MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f),
      MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f),  // token 2, head 0
      MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(0.0f),
      MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f)};  // token 2, head 1

  // The current K/V of all three tokens is stored through slot_mapping and then
  // read back through the selection, so the scatter is part of what is verified.
  const std::vector<MLFloat16> key{
      MLFloat16(1.0f), MLFloat16(0.5f), MLFloat16(0.0f), MLFloat16(0.0f),
      MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f),
      MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(0.0f), MLFloat16(0.0f),
      MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f),
      MLFloat16(0.5f), MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f),
      MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f)};
  const std::vector<MLFloat16> value{
      MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(1.0f),
      MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(1.0f),
      MLFloat16(2.0f), MLFloat16(2.0f), MLFloat16(2.0f), MLFloat16(2.0f),
      MLFloat16(2.0f), MLFloat16(2.0f), MLFloat16(2.0f), MLFloat16(2.0f),
      MLFloat16(3.0f), MLFloat16(3.0f), MLFloat16(3.0f), MLFloat16(3.0f),
      MLFloat16(3.0f), MLFloat16(3.0f), MLFloat16(3.0f), MLFloat16(3.0f)};

  // Request 1 owns block 1; its position 0 is pre-populated, its position 3 is
  // the token stored below at flat slot 1 * kBlockSize + 3.
  std::vector<MLFloat16> key_cache(kGqaCacheElems, MLFloat16(0.0f));
  SetCacheRow(key_cache, kBlockSize, kGqaHeadSize, {2.0f, 1.0f});
  std::vector<MLFloat16> value_cache(kGqaCacheElems, MLFloat16(0.0f));
  SetConstantCacheRow(value_cache, kBlockSize, kGqaHeadSize, 4.0f);

  const float token0_head0 = JointSoftmax({1.0f}, {1.0f});
  const float token0_head1 = JointSoftmax({0.5f}, {1.0f});
  const float token1_head0 = JointSoftmax({1.0f, 0.0f}, {1.0f, 2.0f});
  const float token1_head1 = JointSoftmax({0.5f, 1.0f}, {1.0f, 2.0f});
  const float token2_head0 = JointSoftmax({2.0f, 0.5f}, {4.0f, 3.0f});
  const float token2_head1 = JointSoftmax({1.0f, 0.0f}, {4.0f, 3.0f});
  ASSERT_NE(token1_head0, token1_head1);
  ASSERT_NE(token2_head0, token2_head1);

  std::vector<MLFloat16> expected;
  for (float head_value : {token0_head0, token0_head1, token1_head0, token1_head1, token2_head0,
                           token2_head1}) {
    expected.insert(expected.end(), kGqaHeadSize, MLFloat16(head_value));
  }

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", kNumHeads);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddAttribute<float>("scale", 1.0f);
  tester.AddInput<MLFloat16>("query", {3, kNumHeads * kGqaHeadSize}, query);
  tester.AddInput<MLFloat16>("key", {3, kGqaHeadSize}, key);
  tester.AddInput<MLFloat16>("value", {3, kGqaHeadSize}, value);
  tester.AddInput<MLFloat16>("key_cache", {kNumBlocks, kBlockSize, 1, kGqaHeadSize}, key_cache);
  tester.AddInput<MLFloat16>("value_cache", {kNumBlocks, kBlockSize, 1, kGqaHeadSize}, value_cache);
  // Request 0 owns tokens 0..1 at positions 0..1; request 1 owns token 2 at
  // position 3 (three cached positions precede it).
  tester.AddInput<int32_t>("cumulative_sequence_length", {3}, {0, 2, 3});
  tester.AddInput<int32_t>("past_seqlens", {2}, {0, 3});
  tester.AddInput<int32_t>("block_table", {2, 1}, {0, 1});
  tester.AddInput<int32_t>("slot_mapping", {3}, {0, 1, kBlockSize + 3});
  tester.AddInput<int32_t>("selected_indices", {3, 2}, {0, -1, 0, 1, 0, 3});
  tester.AddInput<int32_t>("selected_counts", {3}, {1, 2, 2});
  tester.AddOutput<MLFloat16>("output", {3, kNumHeads * kGqaHeadSize}, expected);
  tester.SetOutputTolerance(0.01f);
  RunWebGpu(tester);
}

// past_seqlens is caller-owned device data that the op never reads on the host.
// An extreme value must not overflow the i32 metadata arithmetic or turn into an
// unbounded local-window walk: the sanitized lengths are clamped to the
// positions the block table can address, here the 16 slots of the single block.
// With a zero query every logit is 0, so the result is the plain mean of those
// 16 cached value rows. Without the clamp this configuration asks the shader for
// ~2^31 candidates.
TEST(SparsePagedAttention, WebGpu_ExtremePastSeqlenClampsToCacheCapacity) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  std::vector<MLFloat16> value_cache(kCacheElems, MLFloat16(0.0f));
  SetConstantCacheRow(value_cache, 0, kHeadSize, 8.0f);
  const float expected = 8.0f / static_cast<float>(kBlockSize);

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", 1);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddAttribute<std::string>("attention_mode", "local_plus_selected");
  // An unbounded local window, so the candidate count is decided purely by the
  // sanitized lengths.
  tester.AddAttribute<int64_t>("local_window_size", -1);
  tester.AddInput<MLFloat16>("query", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("key", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize},
                             std::vector<MLFloat16>(kCacheElems, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize}, value_cache);
  tester.AddInput<int32_t>("cumulative_sequence_length", {2}, {0, 1});
  tester.AddInput<int32_t>("past_seqlens", {1},
                           std::vector<int32_t>{std::numeric_limits<int32_t>::max()});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  tester.AddInput<int32_t>("slot_mapping", {1}, {-1});
  // Position 0 is already inside the local window, so the de-duplication rule
  // must keep it from being weighted twice.
  tester.AddInput<int32_t>("selected_indices", {1, 2}, {0, -1});
  tester.AddInput<int32_t>("selected_counts", {1}, {1});
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize},
                              std::vector<MLFloat16>(kHeadSize, MLFloat16(expected)));
  tester.SetOutputTolerance(0.01f);
  RunWebGpu(tester);
}

// The same guarantee for cumulative_sequence_length: a length far beyond the
// packed token count must not overflow main_length. Sanitization clamps the
// cumulative range to the token count, so the single token keeps position 0 and
// attends the one selected cache row.
TEST(SparsePagedAttention, WebGpu_ExtremeCumulativeSequenceLengthIsSanitized) {
  if (DefaultWebGpuExecutionProvider() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available.";
  }

  std::vector<MLFloat16> value_cache(kCacheElems, MLFloat16(0.0f));
  SetConstantCacheRow(value_cache, 0, kHeadSize, 6.0f);

  OpTester tester("SparsePagedAttention", 1, kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", 1);
  tester.AddAttribute<int64_t>("kv_num_heads", 1);
  tester.AddInput<MLFloat16>("query", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("key", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value", {1, kHeadSize},
                             std::vector<MLFloat16>(kHeadSize, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("key_cache", {1, kBlockSize, 1, kHeadSize},
                             std::vector<MLFloat16>(kCacheElems, MLFloat16(0.0f)));
  tester.AddInput<MLFloat16>("value_cache", {1, kBlockSize, 1, kHeadSize}, value_cache);
  tester.AddInput<int32_t>(
      "cumulative_sequence_length", {2},
      std::vector<int32_t>{0, std::numeric_limits<int32_t>::max()});
  tester.AddInput<int32_t>("past_seqlens", {1},
                           std::vector<int32_t>{std::numeric_limits<int32_t>::max()});
  tester.AddInput<int32_t>("block_table", {1, 1}, {0});
  tester.AddInput<int32_t>("slot_mapping", {1}, {-1});
  tester.AddInput<int32_t>("selected_indices", {1, 2}, {0, -1});
  tester.AddInput<int32_t>("selected_counts", {1}, {1});
  tester.AddOutput<MLFloat16>("output", {1, kHeadSize},
                              std::vector<MLFloat16>(kHeadSize, MLFloat16(6.0f)));
  tester.SetOutputTolerance(0.01f);
  RunWebGpu(tester);
}

}  // namespace test
}  // namespace onnxruntime
