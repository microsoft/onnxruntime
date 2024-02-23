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

// This op is currently only supported on CUDA- so test it only for CUDA
#ifdef USE_CUDA

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

// QKV
template <typename T>
static std::vector<T> QKV(std::vector<T>& input, std::vector<T>& weights, std::vector<T>& bias,
                          int batch_size, int sequence_length, int hidden_size);

template <>
std::vector<float> QKV(std::vector<float>& input, std::vector<float>& weights, std::vector<float>& bias,
                       int batch_size, int sequence_length, int hidden_size) {
  std::vector<float> qkv;
  qkv.resize(batch_size * sequence_length * 3 * hidden_size, 0);

  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < sequence_length; ++i) {
      for (int j = 0; j < 3 * hidden_size; ++j) {
        float sum = 0;

        for (int k = 0; k < hidden_size; ++k) {
          sum += input[b * sequence_length * hidden_size + i * hidden_size + k] * weights[k * 3 * hidden_size + j];
        }

        qkv[b * sequence_length * 3 * hidden_size + i * 3 * hidden_size + j] = sum + bias[j];
      }
    }
  }

  return qkv;
}

template <>
std::vector<MLFloat16> QKV(std::vector<MLFloat16>& input, std::vector<MLFloat16>& weights, std::vector<MLFloat16>& bias,
                           int batch_size, int sequence_length, int hidden_size) {
  std::vector<MLFloat16> qkv;
  qkv.resize(batch_size * sequence_length * 3 * hidden_size, static_cast<MLFloat16>(0.f));

  for (int b = 0; b < batch_size; ++b) {
    for (int i = 0; i < sequence_length; ++i) {
      for (int j = 0; j < 3 * hidden_size; ++j) {
        float sum = 0;

        for (int k = 0; k < hidden_size; ++k) {
          sum += input[b * sequence_length * hidden_size + i * hidden_size + k].ToFloat() * weights[k * 3 * hidden_size + j].ToFloat();
        }

        qkv[b * sequence_length * 3 * hidden_size + i * 3 * hidden_size + j] = static_cast<MLFloat16>(sum + bias[j].ToFloat());
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
static std::vector<T> ReorderKVCache(std::vector<T>& unordered_k_cache,
                                     int batch_size, int num_heads, int sequence_length,
                                     int head_size, int max_sequence_length) {
  std::vector<T> ordered(unordered_k_cache.size(), T{0.f});

  // Copy V over
  size_t v_start = unordered_k_cache.size() / 2;
  for (size_t i = v_start; i < unordered_k_cache.size(); ++i) {
    ordered[i] = unordered_k_cache[i];
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
          int output_base_offset = base_offset + (c * max_sequence_length * num_inner_elements) + (s * num_inner_elements);

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

// GIven a pointer to the 'V' component of the past cache, we will merge it
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
                                                                                    int past_sequence_length, int max_sequence_length,
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
                            int batch_size, int num_heads, int total_sequence_length, int head_size);

template <>
std::vector<float> QK_Transpose(float* q_matrix, float* k_transpose_matrix,
                                int batch_size, int num_heads, int total_sequence_length, int head_size) {
  int hidden_size = num_heads * head_size;

  std::vector<float> qk_transpose;
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
          float sum = 0;
          for (int k = 0; k < head_size; ++k) {
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

template <>
std::vector<MLFloat16> QK_Transpose(MLFloat16* q_matrix, MLFloat16* k_transpose_matrix,
                                    int batch_size, int num_heads, int total_sequence_length, int head_size) {
  int hidden_size = num_heads * head_size;

  std::vector<MLFloat16> qk_transpose;
  qk_transpose.resize(batch_size * num_heads * total_sequence_length, MLFloat16(0.f));

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
            sum += (q_matrix[input_1_base_offset + i * head_size + k].ToFloat() *
                    k_transpose_matrix[input_2_base_offset + k * total_sequence_length + j].ToFloat());
          }

          float scale = 1 / sqrt(static_cast<float>(head_size));
          qk_transpose[output_base_offset + i * total_sequence_length + j] = MLFloat16(scale * sum);
        }
      }
    }
  }

  return qk_transpose;
}

// Softmax_QK_Transpose
template <typename T>
std::vector<T> Softmax_QK_Transpose(T* qk_transpose_matrix,
                                    int batch_size, int num_heads, int sequence_length, int total_sequence_length, int head_size);

template <>
std::vector<float> Softmax_QK_Transpose(float* qk_transpose_matrix,
                                        int batch_size, int num_heads, int sequence_length, int total_sequence_length, int head_size) {
  if (sequence_length != 1) {
    throw std::runtime_error("Not supported");
  }

  std::vector<float> softmax_qk_transpose;
  softmax_qk_transpose.resize(batch_size * num_heads * sequence_length * total_sequence_length, 0);

  for (int b = 0; b < batch_size; ++b) {
    for (int n = 0; n < num_heads; ++n) {
      int base_offset = (b * num_heads * sequence_length * total_sequence_length) +
                        (n * sequence_length * total_sequence_length);

      float max = std::numeric_limits<float>::min();
      for (int s = 0; s < total_sequence_length; ++s) {
        auto val = qk_transpose_matrix[base_offset + s];
        if (val > max) {
          max = val;
        }
      }

      float denom = 0;
      for (int s = 0; s < total_sequence_length; ++s) {
        auto val = qk_transpose_matrix[base_offset + s];
        denom += std::exp(val - max);
      }

      for (int s = 0; s < total_sequence_length; ++s) {
        auto val = qk_transpose_matrix[base_offset + s];
        softmax_qk_transpose[base_offset + s] = std::exp(val - max) / (denom + (float)0.000001);
      }
    }
  }

  return softmax_qk_transpose;
}

template <>
std::vector<MLFloat16> Softmax_QK_Transpose(MLFloat16* qk_transpose_matrix,
                                            int batch_size, int num_heads, int sequence_length, int total_sequence_length, int head_size) {
  if (sequence_length != 1) {
    throw std::runtime_error("Not supported");
  }

  std::vector<MLFloat16> softmax_qk_transpose;
  softmax_qk_transpose.resize(batch_size * num_heads * sequence_length * total_sequence_length, MLFloat16(0.f));

  for (int b = 0; b < batch_size; ++b) {
    for (int n = 0; n < num_heads; ++n) {
      int base_offset = (b * num_heads * sequence_length * total_sequence_length) +
                        (n * sequence_length * total_sequence_length);

      float max = std::numeric_limits<float>::min();
      for (int s = 0; s < total_sequence_length; ++s) {
        auto val = qk_transpose_matrix[base_offset + s].ToFloat();
        if (val > max) {
          max = val;
        }
      }

      float denom = 0;
      for (int s = 0; s < total_sequence_length; ++s) {
        auto val = qk_transpose_matrix[base_offset + s].ToFloat();
        denom += std::exp(val - max);
      }

      for (int s = 0; s < total_sequence_length; ++s) {
        auto val = qk_transpose_matrix[base_offset + s].ToFloat();
        softmax_qk_transpose[base_offset + s] = MLFloat16(std::exp(val - max) / (denom + (float)0.000001));
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
                                      int head_size);
template <>
std::vector<float> Softmax_QK_Transpose_V(float* softmax_qk_transpose_matrix,
                                          float* v_matrix,
                                          int batch_size, int num_heads, int sequence_length,
                                          int total_sequence_length, int max_sequence_length,
                                          int head_size) {
  if (sequence_length != 1) {
    throw std::runtime_error("Not supported");
  }

  std::vector<float> output;
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
          float sum = 0;

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

template <>
std::vector<MLFloat16> Softmax_QK_Transpose_V(MLFloat16* softmax_qk_transpose_matrix,
                                              MLFloat16* v_matrix,
                                              int batch_size, int num_heads, int sequence_length,
                                              int total_sequence_length, int max_sequence_length,
                                              int head_size) {
  if (sequence_length != 1) {
    throw std::runtime_error("Not supported");
  }

  std::vector<MLFloat16> output;
  output.resize(batch_size * sequence_length * num_heads * head_size, MLFloat16(0.f));

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
            sum += (softmax_qk_transpose_matrix[input_1_base_offset + i * total_sequence_length + k].ToFloat() *
                    v_matrix[input_2_base_offset + k * head_size + j].ToFloat());
          }

          output[output_base_offset + i * head_size + j] = MLFloat16(sum);
        }
      }
    }
  }

  return output;
}
TEST(DecoderMaskedSelfAttentionTest, Test_fp32) {
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
  constexpr int number_of_heads = 12;

  for (MyTestCase test_case : test_cases) {
    int batch_size = test_case.batch_size;
    int past_sequence_length = test_case.past_sequence_length;
    int hidden_size = test_case.hidden_size;

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

    auto reordered_kv_cache = ReorderKVCache<float>(kv_cache, batch_size,
                                                    number_of_heads, past_sequence_length, head_size, max_sequence_length);

    // Validate if reordering went well - by transposing and checking equality
    int chunk_size = 16 / sizeof(float);
    int num_chunks = head_size / chunk_size;
    auto transposed = Transpose<float>(kv_cache.data(), batch_size, number_of_heads, num_chunks, max_sequence_length, chunk_size);
    CheckEquality<float>(transposed.data(), reordered_kv_cache.data(), batch_size, number_of_heads, num_chunks,
                         max_sequence_length, past_sequence_length, chunk_size);

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

    auto present = MergeReorderedKVCacheWithK<float>(reordered_kv_cache, qkv_matrix + hidden_size, batch_size,
                                                     number_of_heads, past_sequence_length, max_sequence_length, head_size);

    // Validate our test logic
    // We want to validate if our merged "unordered" K is the same as
    // the merged "ordered" K so that the QKT we do in our test code
    // is equivalent to the QKT we do in the kernel
    ValidateReorderedMergedKWithK<float>(k_merged.data(), present.data(), batch_size, number_of_heads, total_sequence_length, max_sequence_length, head_size);

    MergeReorderedKVCacheWithV<float>(present.data() + (past_present_size / 2), qkv_matrix + 2 * hidden_size, batch_size,
                                      number_of_heads, past_sequence_length, max_sequence_length, head_size);

    auto output = Softmax_QK_Transpose_V<float>(softmax_qk_transpose.data(), present.data() + (past_present_size / 2),
                                                batch_size, number_of_heads,
                                                sequence_length, total_sequence_length,
                                                max_sequence_length, head_size);

    // Output(s)
    tester.AddOutput<float>("output", input_dims, output);

    tester.AddOutput<float>("present", past_dims, present);

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

TEST(DecoderMaskedSelfAttentionTest, Test_fp16) {
  // The kernel is only supported on CC 5.3 or higher GPUs
  if (NeedSkipIfCudaArchLowerThan(530)) {
    return;
  }

  // Buckets for test data:
  // batch_size: 1, >=2
  // past_sequence_length 0, 1~30, 31~2046, >=2047 (so that total_sequence_length: 1, 2-31, 32~2047, >=2048)
  // head_size: 32, 64, 128
  struct MyTestCase {
    int batch_size;
    int past_sequence_length;
    int hidden_size;
  } test_cases[] = {
      {1, 0, 768},
      {1, 1, 768},
      {3, 30, 384},
      {8, 31, 1536},
      {4, 256, 384},
      {3, 1024, 768},
      {2, 2046, 1536},
      {1, 2047, 384},
      {2, 3000, 768},
  };

  constexpr int sequence_length = 1;
  constexpr int number_of_heads = 12;

  for (MyTestCase test_case : test_cases) {
    int batch_size = test_case.batch_size;
    int past_sequence_length = test_case.past_sequence_length;
    int hidden_size = test_case.hidden_size;

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

    auto input = CreateRandom<MLFloat16>(batch_size * sequence_length * hidden_size);
    tester.AddInput<MLFloat16>("input", input_dims, input);

    auto weight = CreateRandom<MLFloat16>(hidden_size * 3 * hidden_size);
    tester.AddInput<MLFloat16>("weight", weights_dims, weight);

    auto bias = CreateRandom<MLFloat16>(3 * hidden_size);
    tester.AddInput<MLFloat16>("bias", bias_dims, bias);

    // Mask
    tester.AddOptionalInputEdge<int32_t>();

    // Past
    std::vector<int64_t> past_dims = {2, batch_size, number_of_heads, max_sequence_length, head_size};
    int past_present_size = 2 * batch_size * number_of_heads * max_sequence_length * head_size;

    auto kv_cache = CreateRandom<MLFloat16>(past_present_size);

    auto reordered_kv_cache = ReorderKVCache<MLFloat16>(kv_cache, batch_size,
                                                        number_of_heads, past_sequence_length, head_size, max_sequence_length);

    // Validate if reordering went well - by transposing and checking equality
    int chunk_size = 16 / sizeof(MLFloat16);
    int num_chunks = head_size / chunk_size;
    auto transposed = Transpose<MLFloat16>(kv_cache.data(), batch_size, number_of_heads, num_chunks, max_sequence_length, chunk_size);
    CheckEquality<MLFloat16>(transposed.data(), reordered_kv_cache.data(), batch_size, number_of_heads, num_chunks,
                             max_sequence_length, past_sequence_length, chunk_size);

    tester.AddInput<MLFloat16>("past", past_dims, reordered_kv_cache);

    // Rel
    tester.AddOptionalInputEdge<MLFloat16>();

    // Past sequence length
    std::vector<int32_t> arr_past_sequence_len(1, past_sequence_length);
    tester.AddInput<int32_t>("past_sequence_length", {1}, arr_past_sequence_len);

    // QKV MatMul
    auto qkv = QKV(input, weight, bias, batch_size, sequence_length, hidden_size);
    auto* qkv_matrix = qkv.data();

    auto pair = MergePastKWithPresentKAndTranspose<MLFloat16>(kv_cache.data(), qkv_matrix + hidden_size, batch_size,
                                                              number_of_heads, past_sequence_length,
                                                              max_sequence_length, head_size);

    auto k_merged = pair.first;
    auto k_transpose = pair.second;

    auto qk_transpose = QK_Transpose<MLFloat16>(qkv_matrix, k_transpose.data(), batch_size, number_of_heads,
                                                total_sequence_length, head_size);

    auto softmax_qk_transpose = Softmax_QK_Transpose<MLFloat16>(qk_transpose.data(), batch_size, number_of_heads,
                                                                sequence_length, total_sequence_length, head_size);

    auto present = MergeReorderedKVCacheWithK<MLFloat16>(reordered_kv_cache, qkv_matrix + hidden_size, batch_size,
                                                         number_of_heads, past_sequence_length, max_sequence_length, head_size);

    // Validate our test logic
    // We want to validate if our merged "unordered" K is the same as
    // the merged "ordered" K so that the QKT we do in our test code
    // is equivalent to the QKT we do in the kernel
    ValidateReorderedMergedKWithK<MLFloat16>(k_merged.data(), present.data(), batch_size, number_of_heads, total_sequence_length, max_sequence_length, head_size);

    MergeReorderedKVCacheWithV<MLFloat16>(present.data() + (past_present_size / 2), qkv_matrix + 2 * hidden_size, batch_size,
                                          number_of_heads, past_sequence_length, max_sequence_length, head_size);

    auto output = Softmax_QK_Transpose_V(softmax_qk_transpose.data(), present.data() + (past_present_size / 2),
                                         batch_size, number_of_heads,
                                         sequence_length, total_sequence_length,
                                         max_sequence_length, head_size);

    // Output(s)
    tester.AddOutput<MLFloat16>("output", input_dims, output);

    tester.AddOutput<MLFloat16>("present", past_dims, present);

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

#endif

}  // namespace test
}  // namespace onnxruntime
