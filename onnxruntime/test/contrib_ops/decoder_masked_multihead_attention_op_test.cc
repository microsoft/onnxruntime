// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/env_var_utils.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/scoped_env_vars.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "test/contrib_ops/attention_op_test_helper.h"
#include <limits>

namespace onnxruntime {
using contrib::AttentionMaskType;

namespace test {

template <typename T>
static std::vector<T> CreateOnes(int size) {
  std::vector<T> f;
  f.reserve(size);

  for (int i = 0; i < size; ++i) {
    f.push_back(T(1));
  }

  return f;
}

template <typename T>
static std::vector<T> CreateValues(int size, int val) {
  std::vector<T> f;
  f.reserve(size);

  for (int i = 0; i < size; ++i) {
    f.push_back(T(val));
  }

  return f;
}

template <typename T>
static std::vector<T> CreateRandom(int size) {
  std::vector<T> f;
  f.reserve(size);

  for (int i = 0; i < size; ++i) {
    if ((i % 9) == 0) {
      f.push_back((T)0.09);
    } else if ((i % 8) == 0) {
      f.push_back((T)0.08);
    } else if ((i % 7) == 0) {
      f.push_back((T)0.07);
    } else if ((i % 6) == 0) {
      f.push_back((T)0.06);
    } else if ((i % 5) == 0) {
      f.push_back((T)0.05);
    } else if ((i % 4) == 0) {
      f.push_back((T)0.04);
    } else if ((i % 3) == 0) {
      f.push_back((T)0.03);
    } else if ((i % 2) == 0) {
      f.push_back((T)0.02);
    } else {
      f.push_back((T)0.01);
    }
  }

  return f;
}

template <typename T>
static std::vector<T> QKV(std::vector<T>& input, std::vector<T>& weights, std::vector<T>& bias,
                          int batch_size, int sequence_length, int hidden_size) {
  std::vector<T> qkv;
  qkv.resize(batch_size * sequence_length * 3 * hidden_size, 0);

  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < sequence_length; ++i) {
      for (int j = 0; j < 3 * hidden_size; ++j) {
        T sum = 0;

        for (int k = 0; k < hidden_size; ++k) {
          sum += input[b * sequence_length * hidden_size + i * hidden_size + k] * weights[k * 3 * hidden_size + j];
        }

        qkv[b * sequence_length * 3 * hidden_size + i * 3 * hidden_size + j] = sum + bias[j];
      }
    }
  }

  return qkv;
}

// Reorder from [B, N, S, H] to [B, N, H/x, S, x]
// where x = (sizeof(T) / 16);
template <typename T>
static std::vector<T> ReorderKCache(std::vector<T>& unordered_k_cache,
                                    int batch_size, int num_heads, int sequence_length,
                                    int head_size, int max_sequence_length) {
  std::vector<T> ordered = unordered_k_cache;

  int num_inner_elements = 16 / sizeof(T);
  int num_iter = head_size / num_inner_elements;

  for (int b = 0; b < batch_size; ++b) {
    for (int h = 0; h < num_heads; ++h) {
      for (int i = 0; i < num_iter; ++i) {
        for (int s = 0; s < sequence_length; ++s) {
          int base_offset = (b * num_heads * max_sequence_length * head_size) +
                            (h * max_sequence_length * head_size);

          int input_base_offset = base_offset + (s * head_size) + (i * num_inner_elements);
          int output_base_offset = base_offset + (i * max_sequence_length * num_inner_elements) + (s * num_inner_elements);

          for (int e = 0; e < num_inner_elements; ++e) {
            ordered[output_base_offset + e] = unordered_k_cache[input_base_offset + e];
          }
        }
      }
    }
  }

  return ordered;
}

// Merge [B, N, H/x, max_sequence_length (S), x] with [B, N, H/x, 1, x]
// and create [B, N, H/x, max_sequence_length(S+1), x]
template <typename T>
static std::vector<T> MergeReorderedKCacheWithK(std::vector<T>& ordered_k_cache,
                                                T* k,
                                                int batch_size, int num_heads,
                                                int past_sequence_length, int max_sequence_length,
                                                int head_size) {
  std::vector<T> merged = ordered_k_cache;

  int total_seq_length = past_sequence_length + 1;

  int chunk_size = 16 / sizeof(T);
  int num_chunks = head_size / chunk_size;

  for (int b = 0; b < batch_size; ++b) {
    for (int n = 0; n < num_heads; ++n) {
      for (int c = 0; c < num_chunks; ++c) {
        for (int s = 0; s < total_seq_length; ++s) {
          for (int h = 0; h < chunk_size; ++h) {
            T input_value = 0;

            if (s < past_sequence_length) {
              int input_offset = (b * num_heads * num_chunks * max_sequence_length * chunk_size) +
                                 (n * num_chunks * max_sequence_length * chunk_size) +
                                 (c * max_sequence_length * chunk_size) +
                                 (s * chunk_size) +
                                 h;

              input_value = ordered_k_cache[input_offset];
            } else {
              int hidden_size = num_heads * head_size;
              int input_offset = (b * 3 * hidden_size) +
                                 (n * num_chunks * chunk_size) +
                                 (c * chunk_size) +
                                 h;
              input_value = k[input_offset];
            }

            int output_offset = (b * num_heads * num_chunks * max_sequence_length * chunk_size) +
                                (n * num_chunks * max_sequence_length * chunk_size) +
                                (c * max_sequence_length * chunk_size) +
                                (s * chunk_size) +
                                h;

            merged[output_offset] = input_value;
          }
        }
      }
    }
  }

  return merged;
}

template <typename T>
static void MergeVCacheWithV(T* v_cache,
                             T* v,
                             int batch_size, int num_heads,
                             int past_sequence_length, int max_sequence_length,
                             int head_size) {
  int output_iter = past_sequence_length * head_size;

  for (int b = 0; b < batch_size; ++b) {
    for (int n = 0; n < num_heads; ++n) {
      int hidden_size = num_heads * head_size;
      int input_iter = (b * 3 * hidden_size) + (n * head_size);

      for (int h = 0; h < head_size; ++h) {
        v_cache[output_iter + h] = v[input_iter + h];
      }

      output_iter += max_sequence_length * head_size;
    }
  }
}

template <typename T>
static std::pair<std::vector<float>, std::vector<float>> MergePastKWithPresentKAndTranspose(float* past_k, float* present_k,
                                                                                            int num_batch, int num_heads,
                                                                                            int past_sequence_length, int max_sequence_length,
                                                                                            int head_size) {
  int total_seq_length = (past_sequence_length + 1);
  std::vector<T> merged_k(num_batch * num_heads * total_seq_length * head_size, 0);
  std::vector<T> transposed_merged_k(num_batch * num_heads * total_seq_length * head_size, 0);

  for (int b = 0; b < num_batch; ++b) {
    for (int n = 0; n < num_heads; ++n) {
      for (int s = 0; s < total_seq_length; ++s) {
        for (int h = 0; h < head_size; ++h) {
          float input_value = 0.f;

          if (s < past_sequence_length) {
            int input_offset = b * num_heads * max_sequence_length * head_size + (n * max_sequence_length * head_size) + (s * head_size) + h;
            input_value = past_k[input_offset];
          } else {
            int hidden_size = num_heads * head_size;
            // Offset by 3* hidden_size because QKV data contains Q, K, and V per batch
            int input_offset = (b * 3 * hidden_size) + (n * head_size) + h;
            input_value = present_k[input_offset];
          }

          int output_offset = b * num_heads * total_seq_length * head_size + (n * total_seq_length * head_size) + (s * head_size) + h;

          merged_k[output_offset] = input_value;
        }
      }
    }
  }

  for (int b = 0; b < num_batch; ++b) {
    for (int n = 0; n < num_heads; ++n) {
      int base_offset = (b * num_heads * total_seq_length * head_size) +
                        (n * total_seq_length * head_size);

      for (int s = 0; s < total_seq_length; ++s) {
        for (int h = 0; h < head_size; ++h) {
          int input_offset = base_offset + (s * head_size) + h;
          int output_offset = base_offset + (h * total_seq_length) + s;
          transposed_merged_k[output_offset] = merged_k[input_offset];
        }
      }
    }
  }

  return std::make_pair(merged_k, transposed_merged_k);
}

template <typename T>
void ValidateReorderedMergedKWithK(T* k, T* k_cache, int batch_size, int num_heads,
                                   int total_sequence_length, int max_sequence_length, int head_size) {
  // k -> B N S H
  // k_cache -> B N H/c S c

  int chunk_size = 16 / sizeof(T);

  for (int b = 0; b < batch_size; ++b) {
    for (int n = 0; n < num_heads; ++n) {
      for (int s = 0; s < total_sequence_length; ++s) {
        for (int h = 0; h < head_size; ++h) {
          int offset_0 = (b * num_heads * total_sequence_length * head_size) +
                         (n * total_sequence_length * head_size) +
                         (s * head_size) +
                         h;

          int chunk = h / chunk_size;

          int offset_1 = (b * num_heads * max_sequence_length * head_size) +
                         (n * max_sequence_length * head_size) +
                         (chunk * max_sequence_length * chunk_size) +
                         (s * chunk_size) + (h % chunk_size);

          if (k[offset_0] != k_cache[offset_1]) {
            throw std::runtime_error("Not good");
          }
        }
      }
    }
  }
}

template <typename T>
std::vector<T> QK_Transpose(float* q_matrix, float* k_transpose_matrix,
                            int batch_size, int num_heads, int total_sequence_length, int head_size) {
  int hidden_size = num_heads * head_size;

  std::vector<T> qk_transpose;
  qk_transpose.resize(batch_size * num_heads * total_sequence_length, 0);

  for (int b = 0; b < batch_size; ++b) {
    for (int n = 0; n < num_heads; ++n) {
      int input_1_base_offset = (b * 3 * hidden_size) +
                                (n * head_size);

      int input_2_base_offset = (b * num_heads * total_sequence_length * head_size) +
                                (n * total_sequence_length * head_size);

      int output_base_offset = (b * num_heads * total_sequence_length) +
                               (n * total_sequence_length);

      // sequence_length == 1
      for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < total_sequence_length; ++j) {
          T sum = 0;
          for (int k = 0; k < head_size; ++k) {
            float val = q_matrix[input_1_base_offset + i * head_size + k];
            sum += (q_matrix[input_1_base_offset + i * head_size + k] *
                    k_transpose_matrix[input_2_base_offset + k * total_sequence_length + j]);
          }

          float scale = 1 / sqrt(static_cast<float>(head_size));
          qk_transpose[output_base_offset + i * total_sequence_length + j] = scale * sum;
        }
      }
    }
  }

  return qk_transpose;
}

template <typename T>
std::vector<T> Softmax_QK_Transpose(float* qk_transpose_matrix,
                                    int batch_size, int num_heads, int sequence_length, int total_sequence_length, int head_size) {
  if (sequence_length != 1) {
    throw std::exception("Not supported");
  }

  std::vector<T> softmax_qk_transpose;
  softmax_qk_transpose.resize(batch_size * num_heads * sequence_length * total_sequence_length, 0);

  for (int b = 0; b < batch_size; ++b) {
    for (int n = 0; n < num_heads; ++n) {
      int base_offset = (b * num_heads * sequence_length * total_sequence_length) +
                        (n * sequence_length * total_sequence_length);

      T max = std::numeric_limits<T>::min();
      for (int s = 0; s < total_sequence_length; ++s) {
        auto val = qk_transpose_matrix[base_offset + s];
        if (val > max) {
          max = val;
        }
      }

      T denom = 0;
      for (int s = 0; s < total_sequence_length; ++s) {
        auto val = qk_transpose_matrix[base_offset + s];
        denom += std::exp(val - max);
      }

      for (int s = 0; s < total_sequence_length; ++s) {
        auto val = qk_transpose_matrix[base_offset + s];
        softmax_qk_transpose[base_offset + s] = std::exp(val - max) / (denom + (T)0.000001);
      }
    }
  }

  return softmax_qk_transpose;
}

template <typename T>
std::vector<T> Softmax_QK_Transpose_V(float* softmax_qk_transpose_matrix,
                                      T* v_matrix,
                                      int batch_size, int num_heads, int sequence_length,
                                      int total_sequence_length, int max_sequence_length,
                                      int head_size) {
  if (sequence_length != 1) {
    throw std::exception("Not supported");
  }

  std::vector<T> output;
  output.resize(batch_size * sequence_length * num_heads * head_size, 0);

  for (int b = 0; b < batch_size; ++b) {
    for (int n = 0; n < num_heads; ++n) {
      int input_1_base_offset = (b * num_heads * sequence_length * total_sequence_length) +
                                (n * sequence_length * total_sequence_length);

      int input_2_base_offset = (b * num_heads * max_sequence_length * head_size) +
                                (n * max_sequence_length * head_size);

      int output_base_offset = (b * num_heads * sequence_length * head_size) +
                               (n * sequence_length * head_size);

      for (int i = 0; i < sequence_length; ++i) {
        for (int j = 0; j < head_size; ++j) {
          T sum = 0;

          for (int k = 0; k < total_sequence_length; ++k) {
            sum += (softmax_qk_transpose_matrix[input_1_base_offset + i * total_sequence_length + k] *
                    v_matrix[input_2_base_offset + k * head_size + j]);
          }

          output[output_base_offset + i * head_size + j] = sum;
        }
      }
    }
  }

  return output;
}
TEST(DecoderMaskedSelfAttentionTest, MediumSequences_fp32) {
  for (int batch_size = 1; batch_size <= 5; batch_size += 2) {
    for (int past_sequence_length = 1; past_sequence_length <= 3000; past_sequence_length += 150) {
      int sequence_length = 1;
      int hidden_size = 768;
      int number_of_heads = 12;
      int head_size = (hidden_size / number_of_heads);
      int total_sequence_length = sequence_length + past_sequence_length;
      int max_sequence_length = past_sequence_length + 1;  // Always keep >  past_sequence_length

      OpTester tester("DecoderMaskedSelfAttention", 1, onnxruntime::kMSDomain);
      tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));
      tester.AddAttribute<int64_t>("past_present_share_buffer", static_cast<int64_t>(1));

      std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
      std::vector<int64_t> weights_dims = {hidden_size, 3 * hidden_size};
      std::vector<int64_t> bias_dims = {3 * hidden_size};
      std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size};

      auto input = CreateRandom<float>(batch_size * sequence_length * hidden_size);
      tester.AddInput<float>("input", input_dims, input);

      auto weight = CreateRandom<float>(hidden_size * 3 * hidden_size);
      tester.AddInput<float>("weight", weights_dims, weight);

      auto bias = CreateRandom<float>(3 * hidden_size);
      tester.AddInput<float>("bias", bias_dims, bias);

      // Mask
      tester.AddOptionalInputEdge<int32_t>();

      // Past
      std::vector<int64_t> past_dims = {2, batch_size, number_of_heads, max_sequence_length, head_size};
      int past_present_size = 2 * batch_size * number_of_heads * max_sequence_length * head_size;

      auto kv_cache = CreateRandom<float>(past_present_size);

      auto reordered_kv_cache = ReorderKCache(kv_cache, batch_size,
                                              number_of_heads, past_sequence_length, head_size, max_sequence_length);

      tester.AddInput<float>("past", past_dims, reordered_kv_cache);

      // Rel
      tester.AddOptionalInputEdge<float>();

      // Past sequence length
      std::vector<int32_t> arr_past_sequence_len(1, past_sequence_length);
      tester.AddInput<int32_t>("past_sequence_length", {1}, arr_past_sequence_len);

      // QKV MatMul
      auto qkv = QKV(input, weight, bias, batch_size, sequence_length, hidden_size);
      auto* qkv_matrix = qkv.data();

      auto pair = MergePastKWithPresentKAndTranspose<float>(kv_cache.data(), qkv_matrix + hidden_size, batch_size,
                                                            number_of_heads, past_sequence_length,
                                                            max_sequence_length, head_size);

      auto k_merged = pair.first;
      auto k_transpose = pair.second;

      auto qk_transpose = QK_Transpose<float>(qkv_matrix, k_transpose.data(), batch_size, number_of_heads,
                                              total_sequence_length, head_size);

      auto softmax_qk_transpose = Softmax_QK_Transpose<float>(qk_transpose.data(), batch_size, number_of_heads,
                                                              sequence_length, total_sequence_length, head_size);

      auto present = MergeReorderedKCacheWithK(reordered_kv_cache, qkv_matrix + hidden_size, batch_size,
                                               number_of_heads, past_sequence_length, max_sequence_length, head_size);

      ValidateReorderedMergedKWithK<float>(k_merged.data(), present.data(), batch_size, number_of_heads, total_sequence_length, max_sequence_length, head_size);

      auto k_cache_size = past_present_size / 2;

      MergeVCacheWithV<float>(present.data() + k_cache_size, qkv_matrix + 2 * hidden_size, batch_size,
                              number_of_heads, past_sequence_length, max_sequence_length, head_size);

      auto output = Softmax_QK_Transpose_V(softmax_qk_transpose.data(), present.data() + k_cache_size,
                                           batch_size, number_of_heads,
                                           sequence_length, total_sequence_length,
                                           max_sequence_length, head_size);

      // Output(s)
      tester.AddOutput<float>("output", input_dims, output);

      tester.AddOutput<float>("present", past_dims, present);

      // Run
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
}

}  // namespace test
}  // namespace onnxruntime