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

namespace test {

template <typename T>
static std::vector<T> CreateOnes(int size) {
  std::vector<T> f;
  f.reserve(size);

  for (int i = 0; i < size; ++i) {
    f.push_back(T(1.0f));
  }

  return f;
}

template <typename T>
static std::vector<T> CreateValues(int size, float val) {
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
      f.push_back((T)0.09f);
    } else if ((i % 8) == 0) {
      f.push_back((T)0.08f);
    } else if ((i % 7) == 0) {
      f.push_back((T)0.07f);
    } else if ((i % 6) == 0) {
      f.push_back((T)0.06f);
    } else if ((i % 5) == 0) {
      f.push_back((T)0.05f);
    } else if ((i % 4) == 0) {
      f.push_back((T)0.04f);
    } else if ((i % 3) == 0) {
      f.push_back((T)0.03f);
    } else if ((i % 2) == 0) {
      f.push_back((T)0.02f);
    } else {
      f.push_back((T)0.01f);
    }
  }

  return f;
}

template <typename T>
float ToFloat(T val);

template <>
constexpr float ToFloat(float val) {
  return val;
}

template <>
float ToFloat(MLFloat16 val) {
  return val.ToFloat();
}

// QKV
template <typename T>
static std::vector<T> QKV(std::vector<T>& input, std::vector<T>& weights, std::vector<T>& bias,
                          int batch_size, int sequence_length, int hidden_size) {
  std::vector<T> qkv;
  qkv.resize(batch_size * sequence_length * 3 * hidden_size, static_cast<T>(0.f));

  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < sequence_length; ++i) {
      for (int j = 0; j < 3 * hidden_size; ++j) {
        float sum = 0;

        for (int k = 0; k < hidden_size; ++k) {
          sum += ToFloat(input[b * sequence_length * hidden_size + i * hidden_size + k]) *
                 ToFloat(weights[k * 3 * hidden_size + j]);
        }

        qkv[b * sequence_length * 3 * hidden_size + i * 3 * hidden_size + j] = static_cast<T>(sum + ToFloat(bias[j]));
      }
    }
  }

  return qkv;
}

// Transpose [B, N, S, H/x, x] -> [B, N, H/x, S, x]
// where `num_chunks` = H/x
template <typename T>
std::vector<T> Transpose(T* data, int batch_size, int num_heads,
                         int num_chunks, int max_sequence_length, int virtual_head_size) {
  std::vector<T> transposed(batch_size * num_heads * num_chunks * max_sequence_length * virtual_head_size, T{0.f});

  for (int b = 0; b < batch_size; ++b) {
    for (int n = 0; n < num_heads; ++n) {
      int base_offset = (b * num_heads * num_chunks * max_sequence_length * virtual_head_size) +
                        (n * num_chunks * max_sequence_length * virtual_head_size);

      for (int c = 0; c < num_chunks; ++c) {
        for (int s = 0; s < max_sequence_length; ++s) {
          int input_offset = base_offset + s * num_chunks * virtual_head_size + c * virtual_head_size;
          int output_offset = base_offset + c * max_sequence_length * virtual_head_size + s * virtual_head_size;

          for (int h = 0; h < virtual_head_size; ++h) {
            transposed[output_offset + h] = data[input_offset + h];
          }
        }
      }
    }
  }

  return transposed;
}

// Given two buffers of shapes [B, N, c, M_s, c_size]
// check for equality of the first sequence_length elements alone
template <typename T>
void CheckEquality(T* data_1, T* data_2, int batch_size, int num_heads, int num_chunks,
                   int max_sequence_length, int sequence_length, int virtual_head_size) {
  for (int b = 0; b < batch_size; ++b) {
    for (int n = 0; n < num_heads; ++n) {
      for (int c = 0; c < num_chunks; ++c) {
        int base_offset = (b * num_heads * num_chunks * max_sequence_length * virtual_head_size) +
                          (n * num_chunks * max_sequence_length * virtual_head_size) +
                          (c * max_sequence_length * virtual_head_size);

        for (int s = 0; s < sequence_length; ++s) {
          for (int h = 0; h < virtual_head_size; ++h) {
            auto val_1 = data_1[base_offset + s * virtual_head_size + h];
            auto val_2 = data_2[base_offset + s * virtual_head_size + h];
            if (val_1 != val_2) {
              throw std::runtime_error("Equality check failed");
            }
          }
        }
      }
    }
  }
}

// Reorder 'K' from [B, N, S, H] to [B, N, H/x, S, x] where x = (sizeof(T) / 16);
// Copy 'V' over as is
template <typename T>
static std::vector<T> ReorderKVCache(const std::vector<T>& unordered_k_cache,
                                     int batch_size, int num_heads, int sequence_length,
                                     int head_size, int max_sequence_length, bool merge_past_kv = true) {
  std::vector<T> ordered(unordered_k_cache.size(), T{0.f});

  // Copy V over
  if (merge_past_kv) {
    size_t v_start = unordered_k_cache.size() / 2;
    for (size_t i = v_start; i < unordered_k_cache.size(); ++i) {
      ordered[i] = unordered_k_cache[i];
    }
  }

  // Now let us re-order K and copy it over to the final buffer
  int num_inner_elements = 16 / sizeof(T);
  int chunks = head_size / num_inner_elements;

  for (int b = 0; b < batch_size; ++b) {
    for (int h = 0; h < num_heads; ++h) {
      for (int c = 0; c < chunks; ++c) {
        for (int s = 0; s < sequence_length; ++s) {
          int base_offset = (b * num_heads * max_sequence_length * head_size) +
                            (h * max_sequence_length * head_size);

          int input_base_offset = base_offset + (s * head_size) + (c * num_inner_elements);
          int output_base_offset = base_offset + (c * max_sequence_length * num_inner_elements) +
                                   (s * num_inner_elements);

          for (int e = 0; e < num_inner_elements; ++e) {
            ordered[output_base_offset + e] = unordered_k_cache[input_base_offset + e];
          }
        }
      }
    }
  }

  return ordered;
}

// For K: Merge [B, N, H/x, max_sequence_length (S), x] with [B, N, H/x, 1, x]
// and create [B, N, H/x, max_sequence_length(S+1), x]
// For V: Keep as is
template <typename T>
static std::vector<T> MergeReorderedKVCacheWithK(std::vector<T>& ordered_k_cache,
                                                 T* k,
                                                 int batch_size, int num_heads,
                                                 int past_sequence_length, int max_sequence_length,
                                                 int head_size, bool merge_past_kv = true) {
  std::vector<T> merged = ordered_k_cache;

  int total_seq_length = past_sequence_length + 1;

  int chunk_size = 16 / sizeof(T);
  int num_chunks = head_size / chunk_size;

  for (int b = 0; b < batch_size; ++b) {
    for (int n = 0; n < num_heads; ++n) {
      for (int c = 0; c < num_chunks; ++c) {
        for (int s = 0; s < total_seq_length; ++s) {
          for (int h = 0; h < chunk_size; ++h) {
            T input_value{0.f};

            if (s < past_sequence_length) {
              int input_offset = (b * num_heads * num_chunks * max_sequence_length * chunk_size) +
                                 (n * num_chunks * max_sequence_length * chunk_size) +
                                 (c * max_sequence_length * chunk_size) +
                                 (s * chunk_size) +
                                 h;

              input_value = ordered_k_cache[input_offset];
            } else {
              int hidden_size = num_heads * head_size;
              int input_offset = merge_past_kv ? ((b * 3 * hidden_size) +
                                                  (n * num_chunks * chunk_size) +
                                                  (c * chunk_size) +
                                                  h)
                                               : ((b * hidden_size) + n * head_size + c * chunk_size + h);
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

// Given a pointer to the 'V' component of the past cache, we will merge it
// with current 'V' in-place
template <typename T>
static void MergeReorderedKVCacheWithV(T* v_cache,
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
static std::pair<std::vector<T>, std::vector<T>> MergePastKWithPresentKAndTranspose(T* past_k, T* present_k,
                                                                                    int num_batch, int num_heads,
                                                                                    int past_sequence_length,
                                                                                    int max_sequence_length,
                                                                                    int head_size) {
  int total_seq_length = (past_sequence_length + 1);
  std::vector<T> merged_k(num_batch * num_heads * total_seq_length * head_size, T{0.f});
  std::vector<T> transposed_merged_k(num_batch * num_heads * total_seq_length * head_size, T{0.f});

  for (int b = 0; b < num_batch; ++b) {
    for (int n = 0; n < num_heads; ++n) {
      for (int s = 0; s < total_seq_length; ++s) {
        for (int h = 0; h < head_size; ++h) {
          T input_value{0.f};

          if (s < past_sequence_length) {
            int input_offset = b * num_heads * max_sequence_length * head_size +
                               (n * max_sequence_length * head_size) + (s * head_size) + h;
            input_value = past_k[input_offset];
          } else {
            int hidden_size = num_heads * head_size;
            // Offset by 3 * hidden_size because QKV data contains Q, K, and V per batch
            int input_offset = (b * 3 * hidden_size) + (n * head_size) + h;
            input_value = present_k[input_offset];
          }

          int output_offset = b * num_heads * total_seq_length * head_size +
                              (n * total_seq_length * head_size) + (s * head_size) + h;

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
  // k_cache -> B N H/chunk_size max_sequence_length chunk_size

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
            throw std::runtime_error("Validation failed");
          }
        }
      }
    }
  }
}

// QK_Transpose
template <typename T>
std::vector<T> QK_Transpose(T* q_matrix, T* k_transpose_matrix,
                            int batch_size, int num_heads, int total_sequence_length, int head_size) {
  int hidden_size = num_heads * head_size;

  std::vector<T> qk_transpose;
  qk_transpose.resize(batch_size * num_heads * total_sequence_length, static_cast<T>(0.f));

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
          float sum = 0;
          for (int k = 0; k < head_size; ++k) {
            sum += (ToFloat(q_matrix[input_1_base_offset + i * head_size + k]) *
                    ToFloat(k_transpose_matrix[input_2_base_offset + k * total_sequence_length + j]));
          }

          float scale = 1 / sqrt(static_cast<float>(head_size));
          qk_transpose[output_base_offset + i * total_sequence_length + j] = static_cast<T>(scale * sum);
        }
      }
    }
  }

  return qk_transpose;
}

// Softmax_QK_Transpose
template <typename T>
std::vector<T> Softmax_QK_Transpose(T* qk_transpose_matrix, int batch_size, int num_heads,
                                    int sequence_length, int total_sequence_length) {
  if (sequence_length != 1) {
    throw std::runtime_error("Not supported");
  }

  std::vector<T> softmax_qk_transpose;
  softmax_qk_transpose.resize(static_cast<size_t>(batch_size) * num_heads * sequence_length * total_sequence_length,
                              static_cast<T>(0.f));

  for (int b = 0; b < batch_size; ++b) {
    for (int n = 0; n < num_heads; ++n) {
      int base_offset = (b * num_heads * sequence_length * total_sequence_length) +
                        (n * sequence_length * total_sequence_length);

      float max = std::numeric_limits<float>::lowest();
      for (int s = 0; s < total_sequence_length; ++s) {
        auto val = ToFloat(qk_transpose_matrix[base_offset + s]);
        if (val > max) {
          max = val;
        }
      }

      float denom = 0;
      for (int s = 0; s < total_sequence_length; ++s) {
        auto val = ToFloat(qk_transpose_matrix[base_offset + s]);
        denom += std::exp(val - max);
      }

      for (int s = 0; s < total_sequence_length; ++s) {
        auto val = ToFloat(qk_transpose_matrix[base_offset + s]);
        softmax_qk_transpose[base_offset + s] = static_cast<T>(std::exp(val - max) / (denom + (float)0.000001));
      }
    }
  }

  return softmax_qk_transpose;
}

// Softmax_QK_Transpose_V
template <typename T>
std::vector<T> Softmax_QK_Transpose_V(T* softmax_qk_transpose_matrix,
                                      T* v_matrix,
                                      int batch_size, int num_heads, int sequence_length,
                                      int total_sequence_length, int max_sequence_length,
                                      int head_size) {
  if (sequence_length != 1) {
    throw std::runtime_error("Not supported");
  }

  std::vector<T> output;
  output.resize(batch_size * sequence_length * num_heads * head_size, static_cast<T>(0.f));

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
          float sum = 0;

          for (int k = 0; k < total_sequence_length; ++k) {
            sum += (ToFloat(softmax_qk_transpose_matrix[input_1_base_offset + i * total_sequence_length + k]) *
                    ToFloat(v_matrix[input_2_base_offset + k * head_size + j]));
          }

          output[output_base_offset + i * head_size + j] = static_cast<T>(sum);
        }
      }
    }
  }

  return output;
}

// Currently we only support CUDA for DecoderMaskedSelfAttention
#ifdef USE_CUDA

template <typename T>
static void TestDecoderMaskedSelfAttention() {
  // The kernel is only supported on CC 5.3 or higher GPUs
  if (NeedSkipIfCudaArchLowerThan(530)) {
    return;
  }

  // Buckets for test data:
  // batch_size: 1, >=2
  // past_sequence_length 0~30, 31~2046, >=2047 (so that total_sequence_length: 1~31, 32~2047, >=2048)
  // head_size: 32, 64, 128
  struct MyTestCase {
    int batch_size;
    int past_sequence_length;
    int hidden_size;
  } test_cases[] = {
      {1, 0, 768},
      {1, 1, 384},
      {2, 30, 768},
      {3, 31, 1536},
      {4, 512, 384},
      {1, 1024, 768},
      {1, 2046, 1536},
      {2, 2047, 384},
      {3, 3000, 768},
  };

  constexpr int sequence_length = 1;
  constexpr int num_heads = 12;

  for (MyTestCase test_case : test_cases) {
    int batch_size = test_case.batch_size;
    int past_sequence_length = test_case.past_sequence_length;
    int hidden_size = test_case.hidden_size;

    int head_size = (hidden_size / num_heads);
    int total_sequence_length = sequence_length + past_sequence_length;
    int max_sequence_length = past_sequence_length + 1;  // Always keep > past_sequence_length

    OpTester tester("DecoderMaskedSelfAttention", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
    tester.AddAttribute<int64_t>("past_present_share_buffer", static_cast<int64_t>(1));

    std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
    std::vector<int64_t> weights_dims = {hidden_size, 3 * hidden_size};
    std::vector<int64_t> bias_dims = {3 * hidden_size};
    std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size};

    auto input = CreateRandom<T>(batch_size * sequence_length * hidden_size);
    tester.AddInput<T>("input", input_dims, input);

    auto weight = CreateRandom<T>(hidden_size * 3 * hidden_size);
    tester.AddInput<T>("weight", weights_dims, weight);

    auto bias = CreateRandom<T>(3 * hidden_size);
    tester.AddInput<T>("bias", bias_dims, bias);

    // Mask
    tester.AddOptionalInputEdge<int32_t>();

    // Past
    std::vector<int64_t> past_dims = {2, batch_size, num_heads, max_sequence_length, head_size};
    int past_present_size = 2 * batch_size * num_heads * max_sequence_length * head_size;

    auto kv_cache = CreateRandom<T>(past_present_size);

    auto reordered_kv_cache = ReorderKVCache<T>(kv_cache, batch_size,
                                                num_heads, past_sequence_length, head_size, max_sequence_length);

    // Validate if reordering went well - by transposing and checking equality
    int chunk_size = 16 / sizeof(T);
    int num_chunks = head_size / chunk_size;
    auto transposed = Transpose<T>(kv_cache.data(), batch_size, num_heads, num_chunks, max_sequence_length, chunk_size);
    CheckEquality<T>(transposed.data(), reordered_kv_cache.data(), batch_size, num_heads, num_chunks,
                     max_sequence_length, past_sequence_length, chunk_size);

    tester.AddInput<T>("past", past_dims, reordered_kv_cache);

    // Rel
    tester.AddOptionalInputEdge<T>();

    // Past sequence length
    std::vector<int32_t> arr_past_sequence_len(1, past_sequence_length);
    tester.AddInput<int32_t>("past_sequence_length", {1}, arr_past_sequence_len);

    // QKV MatMul
    auto qkv = QKV(input, weight, bias, batch_size, sequence_length, hidden_size);
    auto* qkv_matrix = qkv.data();

    auto pair = MergePastKWithPresentKAndTranspose<T>(kv_cache.data(), qkv_matrix + hidden_size, batch_size, num_heads,
                                                      past_sequence_length, max_sequence_length, head_size);

    auto k_merged = pair.first;
    auto k_transpose = pair.second;

    auto qk_transpose = QK_Transpose<T>(qkv_matrix, k_transpose.data(), batch_size, num_heads,
                                        total_sequence_length, head_size);

    auto softmax_qk_transpose = Softmax_QK_Transpose<T>(qk_transpose.data(), batch_size, num_heads,
                                                        sequence_length, total_sequence_length);

    auto present = MergeReorderedKVCacheWithK<T>(reordered_kv_cache, qkv_matrix + hidden_size, batch_size,
                                                 num_heads, past_sequence_length, max_sequence_length, head_size);

    // Validate our test logic
    // We want to validate if our merged "unordered" K is the same as
    // the merged "ordered" K so that the QKT we do in our test code
    // is equivalent to the QKT we do in the kernel
    ValidateReorderedMergedKWithK<T>(k_merged.data(), present.data(), batch_size, num_heads, total_sequence_length,
                                     max_sequence_length, head_size);

    MergeReorderedKVCacheWithV<T>(present.data() + (past_present_size / 2), qkv_matrix + 2 * hidden_size, batch_size,
                                  num_heads, past_sequence_length, max_sequence_length, head_size);

    auto output = Softmax_QK_Transpose_V<T>(softmax_qk_transpose.data(), present.data() + (past_present_size / 2),
                                            batch_size, num_heads, sequence_length, total_sequence_length,
                                            max_sequence_length, head_size);

    // Output(s)
    tester.AddOutput<T>("output", input_dims, output);
    tester.AddOutput<T>("present", past_dims, present);

    if (std::is_same<T, MLFloat16>::value) {
      tester.SetOutputTolerance(0.005f);
    } else {
      tester.SetOutputTolerance(0.001f, 0.001f);
    }

    // Run - Regular kernel execution path
    {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    // Test alternate kernel path of loading more KV data "in flight"
    {
      ScopedEnvironmentVariables scoped_env_vars{
          EnvVarMap{{onnxruntime::contrib::attention::kDecoderMaskedAttentionLoadKVDataInFlight, "1"}}};

      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());

      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
}

#endif  // USE_CUDA

template <typename T>
static std::vector<T> CalculateOutputQK(const std::vector<T>& q, const std::vector<T>& k,
                                        const std::vector<int32_t>& mask_index, const std::vector<T>& attention_bias,
                                        int batch_size, int num_heads,
                                        int sequence_length, int max_sequence_length, int head_size) {
  // q (B, 1, NH), k (B, N, L(M), H) -> qk (B, N, 1, L)
  // mask_index (B, L), (optional) attention_bias (1, 1, 1, L)
  float scale = 1 / sqrt(static_cast<float>(head_size));
  std::vector<T> output_qk;
  output_qk.resize(static_cast<size_t>(batch_size) * num_heads * sequence_length, static_cast<T>(0.f));
  for (int b = 0; b < batch_size; ++b) {
    for (int n = 0; n < num_heads; ++n) {
      for (int s = 0; s < sequence_length; ++s) {
        float mask_value = (mask_index[b * sequence_length + s] == 0) ? -10000.f : 0.f;
        float bias_value = (attention_bias.empty()) ? 0.f : ToFloat(attention_bias[s]);
        float sum = 0;
        for (int h = 0; h < head_size; ++h) {
          sum += ToFloat(q[b * num_heads * head_size + n * head_size + h]) *
                 ToFloat(k[b * num_heads * max_sequence_length * head_size +
                           n * max_sequence_length * head_size + s * head_size + h]);
        }

        output_qk[b * num_heads * sequence_length + n * sequence_length + s] =
            static_cast<T>(scale * sum + mask_value + bias_value);
      }
    }
  }

  return output_qk;
}

template <typename T>
static std::vector<T> CalculateOutput(const std::vector<T>& softmax, const std::vector<T>& v, int batch_size,
                                      int num_heads, int sequence_length, int max_sequence_length, int head_size) {
  // softmax (B, N, 1, L) v (B, N, L(M), H) -> output (B, N, 1, H)
  std::vector<T> output;
  output.resize(static_cast<size_t>(batch_size) * num_heads * head_size, static_cast<T>(0.f));
  for (int b = 0; b < batch_size; ++b) {
    for (int n = 0; n < num_heads; ++n) {
      for (int h = 0; h < head_size; ++h) {
        float sum = 0;
        for (int s = 0; s < sequence_length; ++s) {
          sum += ToFloat(softmax[b * num_heads * sequence_length + n * sequence_length + s]) *
                 ToFloat(v[b * num_heads * max_sequence_length * head_size +
                           n * max_sequence_length * head_size + s * head_size + h]);
        }

        output[b * num_heads * head_size + n * head_size + h] = static_cast<T>(sum);
      }
    }
  }

  return output;
}

template <typename T>
static std::vector<T> MergePast(const std::vector<T>& past, const std::vector<T>& current, int batch_size,
                                int num_heads, int past_seq_len, int max_seq_len, int head_size) {
  // past (B, N, S(M), H), current (B, 1, NH) -> merged (B, N, S+1(M), H)
  std::vector<T> merged = past;
  for (int b = 0; b < batch_size; ++b) {
    for (int n = 0; n < num_heads; ++n) {
      for (int h = 0; h < head_size; ++h) {
        merged[b * num_heads * max_seq_len * head_size + n * max_seq_len * head_size + past_seq_len * head_size + h] =
            current[b * num_heads * head_size + n * head_size + h];
      }
    }
  }

  return merged;
}

template <typename T>
static std::vector<T> ReorderKVByCacheIndirection(const std::vector<T>& key_or_value,
                                                  const int32_t* cache_indirection,
                                                  int batch_size, int beam_width, int max_sequence_length,
                                                  int num_heads, int head_size, int past_sequence_length) {
  std::vector<T> reordered = key_or_value;

  for (int b = 0; b < batch_size; ++b) {
    int beam_batch_index = b / beam_width;
    const int* beam_indices = cache_indirection + b * max_sequence_length;
    for (int n = 0; n < num_heads; ++n) {
      for (int s = 0; s < past_sequence_length; ++s) {
        int beam_offset = beam_indices[s] * num_heads * max_sequence_length * head_size;
        int beam_batch_offset = (beam_batch_index * beam_width * num_heads + n) * max_sequence_length * head_size;
        for (int h = 0; h < head_size; ++h) {
          reordered[b * num_heads * max_sequence_length * head_size +
                    n * max_sequence_length * head_size + s * head_size + h] =
              key_or_value[beam_offset + beam_batch_offset + s * head_size + h];
        }
      }
    }
  }

  return reordered;
}

template <typename T>
static void TestDecoderMaskedMultiHeadAttention(bool is_cross_attn = true, bool use_cuda = true) {
  int batch_size = 8;
  int past_sequence_length = 2;
  int kv_sequence_length = 16;
  int head_size = 32;
  int num_heads = 12;
  int beam_width = 4;
  int hidden_size = head_size * num_heads;

  OpTester tester("DecoderMaskedMultiHeadAttention", 1, onnxruntime::kMSDomain);
  FixedPatternValueGenerator generator{};
  RandomValueGenerator random{123};

  // Attributes
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
  tester.AddAttribute<int64_t>("past_present_share_buffer", static_cast<int64_t>(!is_cross_attn));
  // Output scaled Q * K^T by default for cross-attention
  tester.AddAttribute<int64_t>("output_qk", static_cast<int64_t>(is_cross_attn));

  // Inputs and outputs
  auto query = CreateRandom<T>(batch_size * 1 * hidden_size);
  tester.AddInput<T>("query", {batch_size, 1, hidden_size}, query);

  if (is_cross_attn) {
    auto key = CreateRandom<T>(batch_size * num_heads * kv_sequence_length * head_size);
    std::vector<T> reordered_key;
    if (use_cuda) {
      reordered_key = ReorderKVCache<T>(key, batch_size, num_heads,
                                        kv_sequence_length, head_size, kv_sequence_length, false);
    }
    auto value = CreateRandom<T>(batch_size * num_heads * kv_sequence_length * head_size);
    tester.AddInput<T>("key", {batch_size, num_heads, kv_sequence_length, head_size}, (use_cuda ? reordered_key : key));
    tester.AddInput<T>("value", {batch_size, num_heads, kv_sequence_length, head_size},
                       CreateRandom<T>(batch_size * num_heads * kv_sequence_length * head_size));

    const std::vector<int64_t> mask_index_dims = {batch_size, kv_sequence_length};
    auto mask_index = generator.Discrete<int32_t>(mask_index_dims, AsSpan({0, 1}));
    tester.AddInput<int32_t>("mask_index", {batch_size, kv_sequence_length}, mask_index);

    // Calculate Softmax(Q * K^T + (Optional) mask) * V
    std::vector<T> empty_attention_bias;
    auto output_qk = CalculateOutputQK(query, key, mask_index, empty_attention_bias, batch_size, num_heads,
                                       kv_sequence_length, kv_sequence_length, head_size);
    std::vector<float> output_qk_float(output_qk.size());
    for (size_t i = 0; i < output_qk.size(); ++i) {
      output_qk_float[i] = static_cast<float>(output_qk[i]);
    }
    auto softmax = Softmax_QK_Transpose<T>(output_qk.data(), batch_size, num_heads, 1, kv_sequence_length);
    auto output = CalculateOutput<T>(softmax, value, batch_size, num_heads,
                                     kv_sequence_length, kv_sequence_length, head_size);

    tester.AddOutput<T>("output", {batch_size, 1, hidden_size}, output);
    tester.AddOptionalOutputEdge<T>();  // optional present_key
    tester.AddOptionalOutputEdge<T>();  // optional present_value
    tester.AddOutput<float>("qk", {batch_size, num_heads, 1, kv_sequence_length}, output_qk_float);
  } else {
    int max_sequence_length = past_sequence_length + 10;
    int total_sequence_length = past_sequence_length + 1;

    auto key = CreateRandom<T>(batch_size * hidden_size);
    auto value = CreateRandom<T>(batch_size * hidden_size);
    tester.AddInput<T>("key", {batch_size, 1, hidden_size}, key);
    tester.AddInput<T>("value", {batch_size, 1, hidden_size}, value);

    const std::vector<int64_t> mask_index_dims = {batch_size, total_sequence_length};
    auto mask_index = generator.Discrete<int32_t>(mask_index_dims, AsSpan({0, 1}));
    tester.AddInput<int32_t>("mask_index", {batch_size, total_sequence_length}, mask_index);
    std::vector<int64_t> attention_bias_dims = {1, 1, 1, total_sequence_length};
    auto attention_bias_float = random.Gaussian<float>(attention_bias_dims, 0.0f, 0.3f);
    std::vector<T> attention_bias(attention_bias_float.size());
    for (size_t i = 0; i < attention_bias.size(); ++i) {
      attention_bias[i] = static_cast<T>(attention_bias_float[i]);
    }
    tester.AddInput<T>("attention_bias", {1, 1, 1, total_sequence_length}, attention_bias);

    auto past_key = CreateRandom<T>(batch_size * num_heads * max_sequence_length * head_size);
    auto past_value = CreateRandom<T>(batch_size * num_heads * max_sequence_length * head_size);

    std::vector<T> reordered_past_key;  // For CUDA, we need to reorder past key
    if (use_cuda) {
      reordered_past_key = ReorderKVCache<T>(past_key, batch_size, num_heads,
                                             past_sequence_length, head_size, max_sequence_length, false);
    }

    tester.AddInput<T>("past_key", {batch_size, num_heads, max_sequence_length, head_size},
                       (use_cuda ? reordered_past_key : past_key));
    tester.AddInput<T>("past_value", {batch_size, num_heads, max_sequence_length, head_size}, past_value);

    // merge past key and value with current key and value
    auto merged_key = MergePast<T>(past_key, key, batch_size, num_heads,
                                   past_sequence_length, max_sequence_length, head_size);
    std::vector<T> merged_reordered_key;
    if (use_cuda) {
      merged_reordered_key = MergeReorderedKVCacheWithK<T>(reordered_past_key, key.data(), batch_size, num_heads,
                                                           past_sequence_length, max_sequence_length, head_size, false);
    }
    auto merged_value = MergePast<T>(past_value, value, batch_size, num_heads,
                                     past_sequence_length, max_sequence_length, head_size);

    tester.AddInput<int32_t>("past_sequence_length", {1}, {past_sequence_length});

    std::vector<T> mod_merged_key, mod_merged_value;
    if (beam_width > 1) {
      tester.AddInput<int32_t>("beam_width", {1}, {beam_width});

      const std::vector<int64_t> cache_indir_dims = {batch_size, beam_width, max_sequence_length};
      auto value_candidates = ValueRange<int32_t>(beam_width);
      auto cache_indir = generator.Discrete<int32_t>(cache_indir_dims, value_candidates);
      tester.AddInput<int32_t>("cache_indirection", cache_indir_dims, cache_indir);

      // Modify merged_key and merged_value according to cache_indirection
      mod_merged_key = ReorderKVByCacheIndirection<T>(merged_key, cache_indir.data(),
                                                      batch_size, beam_width, max_sequence_length,
                                                      num_heads, head_size, past_sequence_length);
      mod_merged_value = ReorderKVByCacheIndirection<T>(merged_value, cache_indir.data(),
                                                        batch_size, beam_width, max_sequence_length,
                                                        num_heads, head_size, past_sequence_length);
    }

    // Calculate Softmax(Q * K^T + (Optional) mask) * V
    auto output_qk = CalculateOutputQK<T>(query, (beam_width > 1 ? mod_merged_key : merged_key),
                                          mask_index, attention_bias,
                                          batch_size, num_heads, total_sequence_length, max_sequence_length, head_size);
    auto softmax = Softmax_QK_Transpose<T>(output_qk.data(), batch_size, num_heads, 1, total_sequence_length);
    auto output = CalculateOutput<T>(softmax, (beam_width > 1 ? mod_merged_value : merged_value),
                                     batch_size, num_heads, total_sequence_length, max_sequence_length, head_size);

    tester.AddOutput<T>("output", {batch_size, 1, hidden_size}, output);
    tester.AddOutput<T>("present_key", {batch_size, num_heads, max_sequence_length, head_size},
                        (use_cuda ? merged_reordered_key : merged_key));
    tester.AddOutput<T>("present_value", {batch_size, num_heads, max_sequence_length, head_size}, merged_value);
  }

  if (std::is_same<T, MLFloat16>::value) {
    tester.SetOutputTolerance(0.02f);
  } else {
    tester.SetOutputTolerance(0.0001f, 0.0001f);
  }

  {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    if (use_cuda) {
      execution_providers.push_back(DefaultCudaExecutionProvider());
    } else {
      execution_providers.push_back(DefaultCpuExecutionProvider());
    }
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

#ifdef USE_CUDA

TEST(DecoderMaskedSelfAttentionTest, Test_fp32) {
  TestDecoderMaskedSelfAttention<float>();
}

TEST(DecoderMaskedSelfAttentionTest, Test_fp16) {
  TestDecoderMaskedSelfAttention<MLFloat16>();
}

TEST(DecoderMaskedMultiHeadAttentionTest, cuda_cross_attn_fp32) {
  TestDecoderMaskedMultiHeadAttention<float>();
}

TEST(DecoderMaskedMultiHeadAttentionTest, cuda_cross_attn_fp16) {
  TestDecoderMaskedMultiHeadAttention<MLFloat16>();
}

TEST(DecoderMaskedMultiHeadAttentionTest, cuda_self_attn_fp32) {
  TestDecoderMaskedMultiHeadAttention<float>(/* is_cross_attn = */ false);
}

TEST(DecoderMaskedMultiHeadAttentionTest, cuda_self_attn_fp16) {
  TestDecoderMaskedMultiHeadAttention<MLFloat16>(/* is_cross_attn = */ false);
}

#endif

TEST(DecoderMaskedMultiHeadAttentionTest, cpu_cross_attn_fp32) {
  TestDecoderMaskedMultiHeadAttention<float>(/* is_cross_attn = */ true, /* use_cuda = */ false);
}

TEST(DecoderMaskedMultiHeadAttentionTest, cpu_self_attn_fp32) {
  TestDecoderMaskedMultiHeadAttention<float>(/* is_cross_attn = */ false, /* use_cuda = */ false);
}

}  // namespace test
}  // namespace onnxruntime
