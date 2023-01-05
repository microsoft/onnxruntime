// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/env_var_utils.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/scoped_env_vars.h"
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
namespace test {
enum MaskIndexType {
  kMaskIndexEnd = 0,      // [batch_size]
  kMaskIndexEndAndStart,  // [2 * batch_size]
  kMaskRaw,               // [batch_size, total_sequence_length]
  kMask3D,                // [batch_size, sequence_length, total_sequence_length]
  kMaskDummy,             // Dummy mask with shape [1, 1] or [batch_size, 1]
  kMask4D                 // Megatron causal mask with shape [batch_size, 1, max_sequence_length, max_sequence_length]
};

template <typename T>
std::vector<T> ReorderToKvCache(
    const T* past_data,
    int batch_size,
    int effective_sequence_length,
    int max_sequence_length,
    int number_of_heads,
    int head_size) {
  std::vector<T> arr(2LL * batch_size * number_of_heads * max_sequence_length * head_size);
  const T* src = past_data;
  T* dst = arr.data();
  const int64_t para_size = static_cast<int64_t>(effective_sequence_length) * head_size;
  const int64_t large_para = static_cast<int64_t>(max_sequence_length) * head_size;
  for (int64_t ob = 0, obe = 2LL * batch_size * number_of_heads; ob < obe; ob++) {
    std::copy_n(src, para_size, dst);
    src += para_size;
    dst += large_para;
  }
  return arr;
}

static void RunAttentionTest(
    const std::vector<float>& input_data,         // input:      [batch_size, sequence_length, hidden_size]
    const std::vector<float>& weights_data,       // weights:    [hidden_size, 3 * hidden_size]
    bool is_weights_constant,                     // weights is constant
    const std::vector<float>& bias_data,          // bias:       [3 * hidden_size]
    const std::vector<int32_t>& mask_index_data,  // mask_index: see MaskIndexType for supported shape
    const std::vector<float>& output_data,        // output:     [batch_size, sequence_length, hidden_size]
    int batch_size,
    int sequence_length,
    int hidden_size,
    int number_of_heads,
    bool use_float16 = false,
    bool is_unidirectional = false,
    bool use_past_state = false,
    int past_sequence_length = 0,
    const std::vector<float>* past_data = nullptr,
    const std::vector<float>* present_data = nullptr,
    MaskIndexType mask_index_type = kMaskIndexEnd,
    int input_hidden_size = 0,
    int max_sequence_length = 0,
    const bool disable_cpu = false,
    const bool disable_cuda = false,
    const bool disable_rocm = false,
    std::vector<int32_t> qkv_sizes = {},
    const std::vector<float>& extra_add_data = {},
    int kv_sequence_length = 0,
    bool past_present_share_buffer = false) {
  input_hidden_size = (input_hidden_size == 0 ? hidden_size : input_hidden_size);  // By default, no pruning.
  kv_sequence_length = (kv_sequence_length == 0 ? sequence_length : kv_sequence_length);
  past_present_share_buffer = past_present_share_buffer && use_past_state;

  int min_cuda_architecture = use_float16 ? 530 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture) && !is_weights_constant && !disable_cuda;
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get()) && !is_weights_constant && !disable_rocm;
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get()) && !use_float16 && !disable_cpu;

  int head_size = hidden_size / number_of_heads;
  if (enable_cpu || enable_cuda || enable_rocm) {
    OpTester tester("Attention", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));
    tester.AddAttribute<int64_t>("unidirectional", static_cast<int64_t>(is_unidirectional ? 1 : 0));
    tester.AddAttribute<int64_t>("past_present_share_buffer", static_cast<int64_t>(past_present_share_buffer ? 1 : 0));

    int32_t qkv_hidden_size_sum;
    int32_t v_hidden_size;
    if (qkv_sizes.size() != 0) {
      qkv_hidden_size_sum = qkv_sizes[0] + qkv_sizes[1] + qkv_sizes[2];
      std::vector<int64_t> sizes_attribute{qkv_sizes[0], qkv_sizes[1], qkv_sizes[2]};
      tester.AddAttribute<std::vector<int64_t>>("qkv_hidden_sizes", sizes_attribute);
      v_hidden_size = qkv_sizes[2];
    } else {
      qkv_hidden_size_sum = 3 * hidden_size;
      v_hidden_size = hidden_size;
    }

    int64_t total_sequence_length = past_sequence_length + kv_sequence_length;

    std::vector<int64_t> input_dims = {batch_size, sequence_length, input_hidden_size};
    std::vector<int64_t> weights_dims = {input_hidden_size, qkv_hidden_size_sum};
    std::vector<int64_t> bias_dims = {qkv_hidden_size_sum};
    std::vector<int64_t> output_dims = {batch_size, sequence_length, v_hidden_size};
    if (use_float16) {
      tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input_data));
      tester.AddInput<MLFloat16>("weight", weights_dims, ToFloat16(weights_data), is_weights_constant);
      if (bias_data.size()) {
        tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
      } else {
        tester.AddOptionalInputEdge<MLFloat16>();
      }
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    } else {
      tester.AddInput<float>("input", input_dims, input_data);
      tester.AddInput<float>("weight", weights_dims, weights_data, is_weights_constant);
      if (bias_data.size()) {
        tester.AddInput<float>("bias", bias_dims, bias_data);
      } else {
        tester.AddOptionalInputEdge<float>();
      }
      tester.AddOutput<float>("output", output_dims, output_data);
    }

    std::vector<int64_t> mask_index_dims_1 = {batch_size};
    std::vector<int64_t> mask_index_dims_2 = {2 * batch_size};
    std::vector<int64_t> mask_index_dims_3 = {batch_size, total_sequence_length};
    std::vector<int64_t> mask_index_dims_4 = {batch_size, 1};
    std::vector<int64_t> mask_index_dims_5 = {batch_size, sequence_length, total_sequence_length};
    std::vector<int64_t> mask_index_dims_6 = {batch_size, 1, max_sequence_length, max_sequence_length};
    std::vector<int64_t> mask_index_dims;
    switch (mask_index_type) {
      case kMaskIndexEnd:
        mask_index_dims = mask_index_dims_1;
        break;
      case kMaskIndexEndAndStart:
        mask_index_dims = mask_index_dims_2;
        break;
      case kMaskRaw:
        mask_index_dims = mask_index_dims_3;
        break;
      case kMaskDummy:
        mask_index_dims = mask_index_dims_4;
        break;
      case kMask3D:
        mask_index_dims = mask_index_dims_5;
        break;
      case kMask4D:
        mask_index_dims = mask_index_dims_6;
        break;
      default:
        assert(0);  // shall not reach here.
        break;
    }
    if (mask_index_data.size() > 0) {  // mask index is optional.
      tester.AddInput<int32_t>("mask_index", mask_index_dims, mask_index_data);
    } else {
      tester.AddOptionalInputEdge<int32_t>();
    }

    if (use_past_state) {
      if (!past_present_share_buffer) {
        std::vector<int64_t> past_dims = {2, batch_size, number_of_heads, past_sequence_length, head_size};
        std::vector<int64_t> present_dims = {2, batch_size, number_of_heads, total_sequence_length, head_size};
        if (use_float16) {
          if (past_sequence_length > 0) {
            tester.AddInput<MLFloat16>("past", past_dims, ToFloat16(*past_data));
          }
          tester.AddOutput<MLFloat16>("present", present_dims, ToFloat16(*present_data));
        } else {
          if (past_sequence_length > 0) {
            tester.AddInput<float>("past", past_dims, *past_data);
          }
          tester.AddOutput<float>("present", present_dims, *present_data);
        }
      } else {  // past_present_share_buffer
        std::vector<int64_t> cache_dims = {2, batch_size, number_of_heads, max_sequence_length, head_size};
        if (use_float16) {
          auto past_cache = ReorderToKvCache(ToFloat16(*past_data).data(), batch_size, past_sequence_length,
                                             max_sequence_length, number_of_heads, head_size);
          auto present_cache = ReorderToKvCache(ToFloat16(*present_data).data(), batch_size, static_cast<int>(total_sequence_length),
                                                max_sequence_length, number_of_heads, head_size);

          tester.AddInput<MLFloat16>("past", cache_dims, past_cache);
          tester.AddOutput<MLFloat16>("present", cache_dims, present_cache);
        } else {
          auto past_cache = ReorderToKvCache(past_data->data(), batch_size, past_sequence_length,
                                             max_sequence_length, number_of_heads, head_size);
          auto present_cache = ReorderToKvCache(present_data->data(), batch_size, static_cast<int>(total_sequence_length),
                                                max_sequence_length, number_of_heads, head_size);
          tester.AddInput<float>("past", cache_dims, past_cache);
          tester.AddOutput<float>("present", cache_dims, present_cache);
        }
      }
    } else {
      if (use_float16) {
        tester.AddOptionalInputEdge<MLFloat16>();
      } else {
        tester.AddOptionalInputEdge<float>();
      }
    }

    std::vector<int64_t> extra_add_data_dims = {batch_size, number_of_heads, sequence_length, sequence_length};
    if (extra_add_data.size() > 0) {
      if (use_float16) {
        tester.AddInput<MLFloat16>("extra_add_qk", extra_add_data_dims, ToFloat16(extra_add_data));
      } else {
        tester.AddInput<float>("extra_add_qk", extra_add_data_dims, extra_add_data);
      }
    } else {
      if (use_float16) {
        tester.AddOptionalInputEdge<MLFloat16>();
      } else {
        tester.AddOptionalInputEdge<float>();
      }
    }

    if (past_present_share_buffer) {
      std::vector<int32_t> arr_past_sequence_len(1, past_sequence_length);
      tester.AddInput<int32_t>("past_sequence_length", {1}, arr_past_sequence_len);
    } else {
      tester.AddOptionalInputEdge<int32_t>();
    }

    if (enable_cuda) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_rocm) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultRocmExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_cpu) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCpuExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
}

static void RunAttentionTest(
    const std::vector<float>& input_data,         // input:      [batch_size, sequence_length, hidden_size]
    const std::vector<float>& weights_data,       // weights:    [hidden_size, 3 * hidden_size]
    const std::vector<float>& bias_data,          // bias:       [3 * hidden_size]
    const std::vector<int32_t>& mask_index_data,  // mask_index
    const std::vector<float>& output_data,        // output:     [batch_size, sequence_length, hidden_size]
    int batch_size,
    int sequence_length,
    int hidden_size,
    int number_of_heads,
    bool use_float16 = false,
    bool is_unidirectional = false,
    bool use_past_state = false,
    int past_sequence_length = 0,
    const std::vector<float>* past_data = nullptr,
    const std::vector<float>* present_data = nullptr,
    MaskIndexType mask_index_type = kMaskIndexEnd,
    int input_hidden_size = 0,
    int max_sequence_length = 0,
    const bool disable_cpu = false,
    const bool disable_cuda = false,
    const bool disable_rocm = false,
    const std::vector<int32_t> qkv_sizes = {},
    const std::vector<float>& extra_add_data = {},
    int kv_sequence_length = 0,
    bool past_present_share_buffer = false) {
  RunAttentionTest(input_data, weights_data, false, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length,
                   past_data, present_data, mask_index_type, input_hidden_size, max_sequence_length,
                   disable_cpu, disable_cuda, disable_rocm, qkv_sizes, extra_add_data,
                   kv_sequence_length, past_present_share_buffer);
  RunAttentionTest(input_data, weights_data, true, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length,
                   past_data, present_data, mask_index_type, input_hidden_size, max_sequence_length,
                   disable_cpu, disable_cuda, disable_rocm, qkv_sizes, extra_add_data,
                   kv_sequence_length, past_present_share_buffer);
}

TEST(AttentionTest, AttentionBatch1) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(AttentionTest, AttentionBatch1WithQKVAttr1) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<int32_t> qkv_sizes = {
      6, 6, 4};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,

      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f,

      0.3f, 0.2f, 4.0f, 2.2f, 2.4f, 3.3f, 2.1f, 4.2f, 0.5f, 0.1f, 0.4f, 1.6f,
      0.4f, 0.8f, 0.9f, 0.1f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f,
      0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f,
      0.5f, 0.7f, 0.2f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> output_data = {
      3.1967618465423584f, 0.51903456449508667f, 0.63051539659500122f, 2.9394614696502686f,
      0.65332180261611938f, 1.000949501991272f, 0.74175024032592773f, 2.8231701850891113f};

  constexpr bool disable_rocm = true;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   false, false, false, 0, nullptr, nullptr, kMaskIndexEnd, 0,
                   0, false, false, disable_rocm, qkv_sizes);
}

TEST(AttentionTest, AttentionBatch1WithQKVAttr2) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      -0.031707365f, 0.053643607f, 0.057394292f, -0.019800574f, 0.075466447f, -0.0034214978f, 0.012995008f, -0.019587509f};

  std::vector<int32_t> qkv_sizes = {
      6, 6, 2};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,

      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f,

      0.3f, 0.2f, 4.0f, 2.2f, 2.4f, 3.3f, 2.1f, 4.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f,
      0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f,
      0.5f, 0.7f};

  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> output_data = {
      0.64932525157928467f, 0.79390722513198853f, 0.64932847023010254f, 0.79375863075256348f};

  constexpr bool disable_rocm = true;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   false, false, false, 0, nullptr, nullptr, kMaskIndexEnd, 0,
                   0, false, false, disable_rocm, qkv_sizes);
}

TEST(AttentionTest, AttentionBatch1ExtraAdd) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<int32_t> qkv_sizes = {};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f,
      0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> extra_add_qk = {
      0.2f, -0.1f, 0.4f, 2.5f, 1.6f, -1.1f, 0.4f, -2.5f};

  std::vector<float> output_data = {
      4.066014289855957f, 0.068997815251350403f, 4.25f, 5.6499996185302734f,
      -1.8799558877944946f, 0.32488855719566345f, 4.25f, 5.6499996185302734f};

  constexpr bool disable_cpu = false;
  constexpr bool disable_cuda = false;
  constexpr bool disable_rocm = false;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   false, false, false, 0, nullptr, nullptr, kMaskIndexEnd, 0,
                   0, disable_cpu, disable_cuda, disable_rocm, qkv_sizes, extra_add_qk);
}

TEST(AttentionTest, AttentionBatch2ExtraAdd) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<int32_t> qkv_sizes = {};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f,
      0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L, 2L};

  std::vector<float> extra_add_qk = {
      0.2f, -0.1f, 0.4f, 2.5f, 1.6f, -1.1f, 0.4f, -2.5f,
      0.2f, -0.1f, 0.4f, 2.5f, 1.6f, -1.1f, 0.4f, -2.5f};

  std::vector<float> output_data = {
      4.066014289855957f, 0.068997815251350403f, 4.25f, 5.6499996185302734f,
      -1.8799558877944946f, 0.32488855719566345f, 4.25f, 5.6499996185302734f,
      4.066014289855957f, 0.068997815251350403f, 4.25f, 5.6499996185302734f,
      -1.8799558877944946f, 0.32488855719566345f, 4.25f, 5.6499996185302734f};

  constexpr bool disable_cpu = false;
  constexpr bool disable_cuda = false;
  constexpr bool disable_rocm = false;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   false, false, false, 0, nullptr, nullptr, kMaskIndexEnd, 0,
                   0, disable_cpu, disable_cuda, disable_rocm, qkv_sizes, extra_add_qk);
}

TEST(AttentionTest, AttentionBatch1_Float16) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> output_data = {
      3.154296875, 0.1082763671875, 4.25, 5.6484375,
      3.970703125, 0.072998046875, 4.25, 5.6484375};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads, true);
}

TEST(AttentionTest, AttentionBatch2) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L, 2L};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(AttentionTest, AttentionMaskPartialSequence) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask_index < sequence_length
  std::vector<int32_t> mask_index_data = {1L};

  std::vector<float> output_data = {
      8.6899995803833008f, -0.13000002503395081f, 4.25f, 5.6499996185302734f,
      8.6899995803833008f, -0.13000002503395081f, 4.2499995231628418f, 5.6499991416931152f};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(AttentionTest, AttentionMaskExceedSequence) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask_index > sequence_length
  std::vector<int32_t> mask_index_data = {3L};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(AttentionTest, AttentionNoMaskIndex) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // No mask_index
  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(AttentionTest, AttentionUnidirectional) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.091099896f, -0.018294459f, -0.36594841f, 0.28410032f,
      -0.12125026f, -0.0066160089f, 0.38809127f, -0.22455512f};

  std::vector<float> weight_data = {
      -0.2659236192703247f,
      0.02789675071835518f,
      0.07280516624450684f,
      0.050951678305864334f,
      0.020417947322130203f,
      -0.04751841351389885f,
      0.043815530836582184f,
      0.006015353370457888f,
      -0.11496957391500473f,
      -0.1773347705602646f,
      0.30928605794906616f,
      0.005648412741720676f,

      0.08960387855768204f,
      -0.27270448207855225f,
      0.14847396314144135f,
      -0.17960812151432037f,
      0.01788954995572567f,
      0.09993876516819f,
      0.03943513706326485f,
      -0.02484400011599064f,
      -0.12958766520023346f,
      0.220433309674263f,
      0.1720484346151352f,
      0.22024005651474f,

      0.059368450194597244f,
      0.1710093915462494f,
      -0.3967452347278595f,
      -0.1591450721025467f,
      0.1446179747581482f,
      -0.20505407452583313f,
      0.12749597430229187f,
      0.32139700651168823f,
      0.139958456158638f,
      -0.10619817674160004f,
      0.04528557509183884f,
      0.045598603785037994f,

      -0.007152545265853405f,
      0.109454445540905f,
      -0.1582530289888382f,
      -0.2646341919898987f,
      0.0920850858092308f,
      0.0701494812965393f,
      -0.19062495231628418f,
      -0.24360455572605133f,
      -0.09368397295475006f,
      0.07878211885690689f,
      0.2973634898662567f,
      0.11210034042596817f};

  std::vector<float> bias_data = {
      -0.0540979839861393f,
      -0.06444740295410156f,
      0.03112877532839775f,
      -0.08288222551345825f,
      0.07840359210968018f,
      0.039143580943346024f,
      -0.45591455698013306f,
      -0.11876055598258972f,
      0.3670335114002228f,
      0.028461361303925514f,
      -0.08913630992174149f,
      0.28048714995384216f};

  // No mask_index
  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      0.28109729f, 0.069518551f, 0.0038009658f, 0.29213354f, 0.3692801f, 0.029495837f, -0.084964074f, 0.28169215f};

  bool is_unidirectional = true;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional);
}

void RawAttentionEmptyPastState(bool past_present_share_buffer) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.091099896f, -0.018294459f, -0.36594841f, 0.28410032f,
      -0.12125026f, -0.0066160089f, 0.38809127f, -0.22455512f};

  std::vector<float> weight_data = {
      -0.2659236192703247f,
      0.02789675071835518f,
      0.07280516624450684f,
      0.050951678305864334f,
      0.020417947322130203f,
      -0.04751841351389885f,
      0.043815530836582184f,
      0.006015353370457888f,
      -0.11496957391500473f,
      -0.1773347705602646f,
      0.30928605794906616f,
      0.005648412741720676f,

      0.08960387855768204f,
      -0.27270448207855225f,
      0.14847396314144135f,
      -0.17960812151432037f,
      0.01788954995572567f,
      0.09993876516819f,
      0.03943513706326485f,
      -0.02484400011599064f,
      -0.12958766520023346f,
      0.220433309674263f,
      0.1720484346151352f,
      0.22024005651474f,

      0.059368450194597244f,
      0.1710093915462494f,
      -0.3967452347278595f,
      -0.1591450721025467f,
      0.1446179747581482f,
      -0.20505407452583313f,
      0.12749597430229187f,
      0.32139700651168823f,
      0.139958456158638f,
      -0.10619817674160004f,
      0.04528557509183884f,
      0.045598603785037994f,

      -0.007152545265853405f,
      0.109454445540905f,
      -0.1582530289888382f,
      -0.2646341919898987f,
      0.0920850858092308f,
      0.0701494812965393f,
      -0.19062495231628418f,
      -0.24360455572605133f,
      -0.09368397295475006f,
      0.07878211885690689f,
      0.2973634898662567f,
      0.11210034042596817f};

  std::vector<float> bias_data = {
      -0.0540979839861393f,
      -0.06444740295410156f,
      0.03112877532839775f,
      -0.08288222551345825f,
      0.07840359210968018f,
      0.039143580943346024f,
      -0.45591455698013306f,
      -0.11876055598258972f,
      0.3670335114002228f,
      0.028461361303925514f,
      -0.08913630992174149f,
      0.28048714995384216f};

  // No mask_index
  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      0.28109729f, 0.069518551f, 0.0038009658f, 0.29213354f, 0.3692801f, 0.029495837f, -0.084964074f, 0.28169215f};

  std::vector<float> past_data = {};

  std::vector<float> present_data = {
      0.053175069391727448f, 0.12795503437519073f, 0.11125634610652924f, -0.0510881207883358f, -0.55345797538757324f, -0.3045809268951416f, -0.36920222640037537f, 0.060108467936515808f, 0.28109729290008545f, 0.069518551230430603f, 0.45718482136726379f, -0.010400654748082161f, 0.0038009658455848694f, 0.29213353991508484f, -0.17697516083717346f, 0.27086889743804932f};

  bool is_unidirectional = true;
  bool use_past_state = true;
  int past_sequence_length = 0;

  if (!past_present_share_buffer) {
    RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                     batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional,
                     use_past_state, past_sequence_length, &past_data, &present_data);
  } else {
    RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                     batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional,
                     use_past_state, past_sequence_length, &past_data, &present_data,
                     kMaskIndexEnd, 0, sequence_length, true, false, true, {}, {}, 0,
                     true);
  }
}

#ifndef ENABLE_TRAINING  // TRT fused attention is enabled only on non-training builds
static void GetWeightAndBiasForHiddenSize64(std::vector<float>& weight_data, std::vector<float>& bias_data) {
  weight_data = {
      -0.004707f, -0.006775f, 0.0009236f, 0.003067f, -0.00806f, 0.00779f, 0.0004425f, 0.00846f, 0.00048f,
      0.00999f, 0.00115f, 0.00226f, -0.00705f, 0.004467f, 0.001455f, -0.006073f, 0.00465f, -0.00861f,
      -0.002779f, 0.00883f, -0.002996f, 0.008354f, -0.003141f, -0.007374f, 0.001634f, -0.009544f, 0.00198f,
      0.005894f, 0.001434f, 0.001589f, 0.00921f, -0.00507f, 0.00448f, 0.0002687f, -0.003147f, 0.001627f,
      -0.005608f, 0.006516f, 0.00935f, -0.004715f, 0.00833f, -0.00563f, 0.00281f, -0.005875f, 0.000629f,
      0.00993f, -0.002695f, 0.004486f, -0.00528f, -0.003807f, 0.00521f, 0.00010276f, 0.003307f, 0.000701f,
      0.0001151f, 0.00649f, 0.00934f, -0.001063f, 0.002327f, -0.0002892f, 0.003317f, -0.003506f, 0.004875f,
      0.0006566f, 0.000953f, -0.005898f, 0.00326f, 0.00877f, 0.00923f, -0.00622f, -0.006588f, 0.007748f,
      -0.001789f, 0.00002104f, 0.002937f, 0.00816f, 0.005833f, -0.006634f, 0.006985f, 0.00951f, 0.002947f,
      0.001871f, -0.009445f, 0.0004554f, -0.006294f, -0.00649f, 0.00917f, -0.004158f, -0.00462f, -0.001531f,
      -0.00658f, -0.00364f, -0.00462f, 0.003723f, 0.009636f, 0.003305f, -0.00984f, 0.006126f, -0.0010395f,
      -0.00852f, 0.006287f, -0.002949f, -0.004f, -0.002415f, 0.0009527f, 0.001624f, 0.00364f, 0.007088f,
      -0.00717f, -0.009224f, -0.00997f, 0.001726f, -0.00877f, -0.000602f, 0.0089f, 0.009026f, -0.009514f,
      0.00852f, 0.0003986f, -0.006855f, -0.00583f, 0.003622f, -0.00526f, 0.001879f, -0.007053f, 0.00006664f,
      0.00972f, -0.000457f, -0.00759f, -0.007107f, 0.002337f, -0.004204f, -0.005676f, 0.00985f, 0.00978f,
      -0.004486f, 0.005093f, -0.009285f, 0.004093f, -0.00682f, 0.00963f, -0.006954f, -0.003674f, -0.003822f,
      0.00202f, -0.004635f, -0.0009174f, -0.001202f, 0.00639f, -0.004356f, -0.00741f, -0.00586f, -0.00319f,
      -0.002506f, 0.005047f, 0.007156f, -0.00765f, 0.00702f, 0.007477f, 0.000626f, -0.001587f, -0.005455f,
      0.005814f, -0.002127f, 0.00834f, 0.001279f, 0.007996f, -0.005787f, -0.006924f, -0.004063f, -0.00435f,
      -0.00427f, 0.0002115f, 0.00981f, -0.00138f, -0.007965f, -0.004536f, -0.003431f, 0.00416f, 0.005894f,
      0.006054f, 0.00907f, 0.00388f, -0.006763f, 0.001692f, -0.00797f, -0.00691f, 0.00798f, 0.00867f,
      -0.00788f, 0.002062f, -0.003761f, 0.009834f, -0.002445f, -0.00613f, 0.0096f, -0.005466f, -0.0008426f,
      0.0002431f, -0.009995f, 0.003736f, -0.0071f, -0.003593f, 0.006386f, 0.005997f, -0.003328f, 0.007515f,
      -0.008675f, 0.00547f, -0.00388f, 0.00473f, 0.00362f, -0.00469f, 0.006958f, -0.001264f, -0.003887f,
      -0.004276f, -0.000396f, 0.00453f, -0.00465f, -0.007343f, -0.005787f, -0.00927f, -0.006058f, -0.004566f,
      -0.009056f, -0.00891f, 0.007633f, 0.001098f, -0.003368f, -0.007214f, -0.00905f, -0.00898f, -0.008736f,
      -0.00948f, 0.003162f, 0.004402f, -0.006245f, -0.00515f, -0.00378f, -0.003248f, -0.00304f, 0.001834f,
      -0.002672f, 0.005234f, -0.007706f, 0.0084f, 0.00832f, -0.00904f, -0.00596f, 0.009926f, -0.00869f,
      0.001513f, 0.00728f, 0.001057f, 0.001452f, 0.00785f, 0.001203f, -0.004528f, 0.006573f, 0.003656f,
      0.005966f, -0.006985f, 0.002844f, 0.00883f, 0.0004826f, 0.003279f, 0.006916f, 0.00263f, -0.002415f,
      -0.001928f, -0.0004041f, -0.004593f, -0.00204f, 0.007965f, -0.008224f, -0.00591f, -0.002144f, 0.000688f,
      0.001676f, -0.00949f, -0.003304f, -0.007637f, 0.00973f, -0.008224f, -0.001211f, -0.003345f, 0.002115f,
      -0.00615f, -0.004955f, -0.00803f, 0.00807f, -0.0006227f, 0.00845f, -0.006916f, 0.004353f, -0.000934f,
      0.005604f, -0.00825f, -0.004402f, -0.00441f, 0.00257f, -0.008415f, 0.006542f, 0.001357f, -0.004974f,
      -0.00993f, 0.0001058f, 0.002855f, -0.0081f, 0.001513f, -0.00191f, 0.0004003f, 0.003874f, -0.0015545f,
      -0.00736f, 0.006718f, 0.005135f, 0.003859f, -0.0054f, 0.00993f, 0.000952f, 0.00228f, 0.001163f,
      0.00918f, 0.00582f, 0.00308f, 0.008415f, 0.00889f, 0.00011665f, -0.007362f, -0.009926f, -0.00784f,
      0.005817f, -0.002918f, 0.005043f, -0.003029f, 0.0085f, -0.007362f, -0.00857f, 0.006832f, -0.00055f,
      0.008835f, -0.00522f, -0.002085f, 0.00353f, -0.007706f, 0.006283f, 0.004414f, -0.002405f, -0.003002f,
      -0.00946f, -0.001164f, -0.004177f, 0.00834f, -0.001576f, 0.00855f, 0.004025f, 0.000285f, -0.004486f,
      -0.00703f, -0.003061f, 0.003452f, 0.001276f, 0.008446f, -0.001302f, 0.004333f, -0.00898f, -0.002445f,
      -0.006523f, 0.0004334f, -0.003206f, -0.00349f, -0.005497f, -0.007786f, 0.007397f, 0.00925f, 0.002077f,
      0.004074f, 0.006626f, -0.001693f, -0.0005975f, -0.005074f, 0.00324f, 0.00925f, -0.009735f, -0.007133f,
      -0.0064f, -0.00455f, -0.003153f, 0.0056f, -0.006073f, -0.00274f, -0.00587f, -0.005066f, 0.003595f,
      -0.00932f, -0.005f, 0.00569f, 0.008415f, 0.006866f, 0.003952f, -0.009285f, -0.008064f, 0.00824f,
      0.0000188f, -0.001233f, 0.005726f, -0.0007806f, -0.008385f, -0.001798f, -0.008095f, 0.00986f, 0.006924f,
      0.00712f, -0.00964f, -0.00797f, 0.00943f, -0.007416f, 0.007904f, 0.006893f, 0.00799f, -0.007164f,
      0.007214f, 0.00931f, 0.000645f, -0.0058f, 0.009254f, -0.002079f, 0.000969f, 0.009636f, -0.002365f,
      -0.002348f, 0.007053f, -0.002796f, -0.007652f, -0.001554f, 0.00402f, -0.002838f, -0.006958f, 0.000331f,
      0.006435f, -0.004036f, 0.007595f, 0.00812f, 0.00637f, 0.007732f, -0.006916f, 0.003952f, -0.008064f,
      -0.00928f, 0.00468f, -0.000512f, -0.006287f, 0.00607f, -0.001904f, -0.00458f, 0.003412f, 0.000382f,
      -0.00822f, -0.00486f, 0.0008364f, 0.0004992f, 0.003582f, 0.0088f, 0.002453f, -0.00856f, 0.00886f,
      0.0077f, 0.0004592f, -0.001417f, -0.005142f, 0.004696f, -0.003576f, 0.004807f, -0.00851f, -0.006245f,
      -0.003649f, -0.0001528f, 0.004017f, -0.006123f, -0.004158f, -0.00445f, 0.004864f, -0.0005493f, 0.00399f,
      -0.007244f, 0.003246f, 0.00407f, 0.00929f, -0.006706f, 0.0084f, -0.003496f, 0.00843f, 0.00514f,
      0.002714f, -0.0001633f, -0.00866f, 0.004837f, -0.003016f, 0.00593f, -0.00849f, 0.001287f, -0.007706f,
      0.001479f, -0.002241f, 0.00843f, -0.001236f, -0.007572f, -0.004448f, -0.001927f, 0.001139f, 0.004982f,
      -0.00673f, -0.000568f, 0.009346f, 0.000487f, 0.001392f, -0.009605f, 0.00944f, 0.002022f, 0.00617f,
      0.00472f, 0.009575f, -0.006416f, 0.004265f, 0.002005f, 0.000578f, 0.002592f, 0.002707f, -0.005333f,
      -0.00928f, -0.00935f, -0.00833f, -0.00205f, -0.005795f, -0.001061f, -0.003605f, 0.003078f, 0.00592f,
      0.0006485f, -0.00504f, 0.002682f, 0.00826f, -0.003983f, -0.00493f, 0.00406f, -0.00838f, 0.0032f,
      0.0009565f, 0.00471f, 0.00504f, 0.004612f, -0.002768f, 0.00791f, -0.002892f, 0.00471f, 0.00588f,
      0.005978f, -0.005203f, -0.009995f, 0.009346f, -0.00802f, 0.003807f, 0.001364f, -0.00736f, 0.009285f,
      -0.001995f, 0.002632f, -0.00904f, 0.007042f, -0.00326f, 0.006516f, 0.00492f, 0.00734f, -0.00867f,
      -0.002512f, -0.003729f, 0.0027f, -0.002659f, -0.009514f, -0.005634f, -0.001473f, -0.00545f, 0.003551f,
      0.001995f, -0.003704f, 0.006386f, 0.003313f, -0.002823f, 0.00105f, 0.00993f, 0.00951f, -0.007275f,
      -0.002213f, -0.003418f, 0.00599f, 0.00948f, 0.007572f, -0.00944f, -0.00924f, 0.00011665f, 0.0069f,
      -0.00544f, 0.007515f, -0.006832f, -0.007774f, 0.00853f, -0.0007486f, -0.00643f, -0.0001878f, -0.00849f,
      -0.007603f, 0.0016985f, -0.00986f, 0.003975f, -0.002176f, -0.009796f, 0.004795f, -0.00699f, -0.006725f,
      0.00109f, 0.004498f, -0.00569f, -0.00584f, 0.004047f, -0.001022f, 0.001479f, -0.00751f, -0.002579f,
      -0.004086f, 0.007603f, -0.0000106f, 0.007366f, 0.0029f, -0.003498f, 0.007385f, -0.00759f, -0.005886f,
      0.00476f, -0.0003812f, -0.00008225f, 0.00998f, 0.002716f, -0.00925f, -0.00439f, -0.000902f, -0.00296f,
      -0.007347f, -0.005882f, -0.001428f, -0.002855f, -0.003311f, -0.000793f, -0.00403f, -0.00829f, -0.00999f,
      -0.00838f, 0.008804f, 0.004124f, -0.005882f, 0.001305f, 0.00511f, 0.00799f, -0.00953f, -0.008575f,
      -0.00556f, -0.00858f, 0.00565f, 0.00908f, 0.00591f, 0.0007925f, -0.00912f, -0.005894f, -0.002588f,
      -0.00957f, -0.00682f, 0.002174f, 0.00706f, 0.00528f, 0.0069f, -0.004517f, -0.002382f, 0.005596f,
      0.00645f, 0.00956f, 0.00796f, 0.007706f, 0.004818f, 0.002308f, 0.001367f, -0.004177f, 0.00842f,
      0.007416f, -0.00404f, -0.009094f, 0.00447f, -0.00284f, -0.002499f, -0.0001582f, 0.001681f, 0.004993f,
      -0.0059f, 0.007282f, -0.00809f, 0.00927f, 0.004948f, 0.009766f, -0.00618f, -0.001559f, -0.00461f,
      0.001866f, 0.00827f, -0.00785f, -0.003101f, 0.00977f, -0.00444f, -0.00916f, -0.0008535f, 0.004913f,
      0.005627f, 0.007965f, 0.000532f, -0.00878f, 0.004047f, -0.005302f, 0.00201f, 0.002964f, -0.00895f,
      0.005768f, 0.00388f, 0.007526f, -0.00783f, 0.003794f, 0.005363f, 0.003454f, -0.002235f, -0.003494f,
      -0.001541f, -0.00003624f, -0.0007634f, -0.0014f, -0.003124f, 0.00829f, -0.00298f, -0.00868f, -0.001243f,
      -0.005383f, -0.009964f, 0.004433f, -0.002045f, -0.00753f, 0.002361f, -0.007473f, -0.002419f, -0.000931f,
      0.00585f, 0.007114f, -0.002247f, 0.00472f, -0.003033f, -0.001974f, 0.001622f, -0.007473f, -0.005375f,
      -0.005013f, 0.00436f, 0.00662f, -0.0053f, 0.000606f, -0.00849f, -0.007004f, 0.006794f, -0.0005445f,
      -0.001269f, 0.00391f, 0.006294f, 0.007088f, -0.009026f, -0.001965f, -0.008545f, 0.002115f, 0.003534f,
      -0.00857f, 0.00412f, -0.00722f, -0.006386f, 0.00595f, -0.003778f, -0.00886f, -0.0002267f, 0.00249f,
      -0.002825f, 0.0003204f, 0.0002894f, -0.004147f, -0.003632f, 0.001764f, -0.002983f, 0.006584f, -0.004402f,
      0.006493f, 0.002014f, -0.0061f, 0.00816f, 0.005585f, -0.008125f, 0.006546f, -0.00956f, 0.004185f,
      0.001067f, 0.001277f, 0.007835f, -0.003933f, 0.00979f, -0.003376f, 0.006573f, -0.00501f, 0.0007577f,
      0.00133f, -0.00737f, 0.00885f, -0.00599f, -0.001151f, -0.001389f, -0.00987f, -0.003214f, -0.00649f,
      0.005424f, 0.0004575f, 0.002352f, 0.005722f, -0.001995f, -0.007717f, 0.001034f, -0.006557f, 0.0088f,
      -0.003183f, -0.00663f, 0.00634f, -0.003008f, -0.004925f, 0.00539f, -0.00432f, -0.00651f, 0.009895f,
      0.00532f, -0.0003607f, 0.003397f, 0.006145f, 0.00531f, -0.006275f, 0.00985f, -0.00471f, 0.00817f,
      -0.00927f, 0.007217f, 0.005924f, 0.003187f, 0.001192f, -0.003986f, -0.0000217f, -0.0012245f, -0.003933f,
      -0.00617f, -0.002232f, 0.00444f, 0.002008f, 0.0006056f, -0.002827f, -0.007366f, 0.002996f, -0.006752f,
      -0.004143f, 0.001662f, -0.00793f, 0.002161f, 0.0001992f, 0.00803f, -0.0000725f, 0.001066f, 0.004745f,
      -0.005367f, -0.00641f, 0.00431f, -0.004715f, 0.008575f, -0.007202f, 0.003786f, -0.00247f, 0.006382f,
      -0.006832f, 0.00505f, -0.001084f, 0.009674f, 0.00458f, -0.00473f, -0.00656f, -0.00011283f, 0.004417f,
      -0.001419f, -0.0005164f, 0.0000397f, -0.00395f, 0.00417f, -0.005512f, 0.0088f, 0.00568f, -0.0005984f,
      0.003128f, -0.006283f, -0.0000904f, -0.004738f, 0.00687f, 0.00592f, -0.005768f, -0.00859f, 0.003523f,
      0.001169f, -0.004498f, 0.00541f, 0.002956f, 0.00896f, -0.002571f, 0.0006533f, 0.002089f, -0.00473f,
      -0.002241f, 0.005016f, 0.001295f, 0.005993f, -0.008064f, 0.000595f, -0.007744f, -0.00201f, 0.0075f,
      -0.00942f, 0.0002023f, -0.00979f, -0.002243f, 0.002829f, -0.004322f, 0.009125f, 0.00704f, 0.007282f,
      0.00807f, 0.005447f, 0.00518f, -0.0010195f, -0.004803f, -0.001293f, -0.001305f, 0.00975f, -0.00564f,
      -0.005215f, -0.009445f, 0.00999f, 0.00959f, -0.009224f, -0.0053f, -0.002106f, -0.00839f, 0.001516f,
      0.003109f, 0.004414f, -0.00921f, -0.00868f, 0.00833f, 0.00809f, 0.004654f, 0.00678f, 0.002237f,
      0.007195f, -0.004875f, -0.001252f, 0.0073f, 0.007275f, 0.00825f, -0.005936f, 0.00594f, -0.00381f,
      -0.002117f, 0.009f, -0.003998f, -0.00104f, -0.00421f, 0.00526f, 0.001031f, 0.00902f, 0.006794f,
      -0.00912f, -0.0002892f, 0.002966f, 0.00478f, 0.00581f, 0.007217f, 0.008156f, -0.0000639f, -0.003164f,
      0.00859f, -0.00897f, 0.00409f, 0.0008936f, -0.00991f, -0.008316f, -0.004055f, 0.001252f, -0.00473f,
      -0.002f, -0.003933f, 0.000755f, -0.00992f, 0.003569f, -0.00812f, -0.004215f, -0.00774f, 0.00907f,
      0.00653f, -0.00992f, -0.006252f, -0.00468f, -0.001105f, -0.007717f, 0.005302f, 0.003773f, -0.001262f,
      -0.006207f, -0.005707f, 0.0053f, 0.00415f, 0.002441f, 0.0009265f, -0.006744f, 0.00994f, -0.0004816f,
      -0.002108f, -0.003267f, 0.0000461f, 0.004364f, -0.00596f, -0.008675f, 0.005703f, 0.002748f, 0.00961f,
      0.006767f, -0.0000575f, -0.00845f, -0.003597f, 0.003616f, 0.00423f, 0.009705f, -0.00976f, -0.0085f,
      0.00307f, -0.004032f, -0.00784f, -0.00901f, -0.00873f, 0.00543f, 0.00744f, -0.006588f, -0.004765f,
      -0.007202f, 0.006306f, -0.007484f, 0.007442f, -0.00008386f, 0.006374f, 0.00879f, 0.002039f, -0.003298f,
      0.003407f, 0.004673f, 0.0068f, 0.0001981f, 0.002296f, 0.008194f, -0.00805f, -0.007637f, -0.00903f,
      -0.004025f, 0.001553f, 0.00881f, 0.001311f, -0.005016f, -0.006916f, -0.009926f, -0.00801f, 0.00945f,
      0.0001532f, 0.00234f, -0.002968f, -0.002174f, 0.004585f, -0.00658f, 0.000132f, 0.0004494f, -0.00954f,
      -0.00848f, 0.009964f, -0.0006323f, -0.005016f, 0.001238f, 0.00433f, 0.001477f, 0.00578f, 0.00794f,
      -0.00512f, -0.00207f, -0.00145f, -0.001166f, 0.008644f, -0.00915f, 0.007187f, -0.00415f, 0.006035f,
      -0.004177f, 0.00817f, -0.00432f, 0.001062f, -0.005272f, -0.0004163f, 0.005154f, 0.005688f, -0.002985f,
      -0.004f, -0.003176f, 0.00137f, 0.0002158f, 0.003798f, 0.0002009f, -0.01f, 0.00311f, -0.004234f,
      0.00681f, -0.005657f, -0.00963f, 0.00916f, 0.00847f, -0.002085f, -0.00211f, 0.006813f, -0.00473f,
      0.00873f, 0.0008483f, 0.004253f, 0.00865f, -0.007156f, -0.00996f, 0.005413f, -0.004253f, 0.00847f,
      0.004482f, 0.000647f, -0.006702f, 0.00845f, -0.009254f, -0.0001926f, 0.003868f, -0.00788f, 0.00951f,
      -0.0005136f, -0.007698f, 0.00889f, -0.00953f, 0.007965f, 0.004982f, -0.004345f, 0.00841f, 0.007034f,
      0.006092f, 0.004166f, 0.00682f, -0.004635f, 0.003433f, -0.006527f, -0.0002658f, 0.005455f, 0.001926f,
      -0.003582f, -0.0065f, 0.002348f, -0.001918f, -0.00488f, -0.006416f, -0.000873f, -0.00942f, 0.005177f,
      -0.00194f, 0.006374f, 0.003983f, 0.00963f, 0.00697f, -0.00809f, -0.00791f, -0.003254f, -0.00669f,
      -0.001487f, 0.002129f, -0.000799f, -0.003944f, 0.002693f, 0.00667f, 0.00892f, 0.002377f, 0.001005f,
      -0.00792f, 0.002398f, -0.001093f, 0.0006456f, -0.002361f, 0.00533f, 0.0064f, 0.004524f, -0.0066f,
      0.004406f, 0.007538f, 0.00611f, 0.006294f, 0.0004857f, -0.00859f, 0.00928f, -0.005505f, -0.001135f,
      -0.00712f, -0.00923f, 0.007534f, 0.00258f, 0.00685f, -0.00873f, 0.001684f, -0.001002f, -0.0005627f,
      0.00352f, -0.007324f, 0.00838f, 0.00731f, 0.006733f, -0.003832f, -0.00522f, 0.00299f, 0.000935f,
      -0.005245f, 0.000987f, 0.007515f, 0.00704f, 0.0086f, 0.00133f, 0.0038f, 0.00622f, -0.0085f,
      0.00988f, 0.00625f, 0.00835f, -0.006023f, 0.007084f, -0.002728f, 0.009995f, 0.0008073f, 0.00341f,
      -0.004547f, 0.005917f, -0.00818f, -0.009705f, 0.00907f, -0.008965f, 0.003483f, -0.00556f, -0.001769f,
      0.0068f, 0.007442f, 0.00497f, -0.001922f, 0.002583f, -0.00834f, 0.004417f, 0.005028f, 0.006336f,
      0.00402f, -0.00773f, 0.00672f, 0.00324f, 0.003595f, -0.00852f, 0.00503f, -0.00794f, -0.009766f,
      -0.000813f, -0.006924f, -0.006622f, 0.0008802f, 0.004177f, 0.007427f, -0.001697f, 0.008575f, 0.00414f,
      0.00728f, 0.001138f, 0.000674f, -0.00209f, 0.004883f, -0.003029f, 0.0084f, -0.00798f, -0.003302f,
      0.007866f, 0.0006804f, 0.00306f, 0.006325f, 0.000508f, -0.002022f, 0.00473f, 0.00958f, -0.001912f,
      -0.002256f, 0.001385f, 0.001143f, 0.007668f, -0.002575f, 0.004364f, 0.00919f, -0.00924f, 0.00558f,
      -0.00447f, -0.004196f, -0.00547f, 0.00868f, -0.001469f, -0.00849f, 0.006397f, -0.00529f, 0.002329f,
      0.00847f, -0.009705f, 0.00233f, 0.000902f, 0.006073f, -0.00536f, 0.000875f, 0.002682f, -0.003347f,
      0.00905f, -0.00399f, -0.005783f, -0.00942f, 0.00671f, -0.008095f, -0.004467f, -0.008415f, 0.007996f,
      -0.00848f, -0.00531f, 0.002605f, -0.00632f, -0.007652f, 0.009605f, 0.00929f, 0.007782f, -0.006844f,
      -0.00115f, -0.006626f, -0.007526f, -0.001129f, 0.00943f, 0.004242f, -0.00486f, 0.00963f, -0.006386f,
      -0.004513f, 0.00185f, -0.001695f, 0.00976f, -0.001186f, 0.001484f, 0.00429f, 0.000502f, -0.009285f,
      0.005882f, -0.00674f, 0.00882f, 0.00816f, -0.008705f, -0.003618f, 0.00406f, 0.007607f, -0.001528f,
      -0.006336f, 0.006943f, 0.00753f, -0.004963f, -0.00602f, 0.002424f, -0.009476f, 0.007385f, 0.00988f,
      -0.00359f, -0.005722f, 0.006863f, -0.00398f, -0.005486f, -0.004898f, -0.0000809f, -0.001511f, 0.00307f,
      0.002613f, 0.0004046f, 0.005634f, 0.00449f, 0.008606f, -0.002146f, 0.002882f, -0.007065f, -0.00796f,
      -0.001136f, -0.001323f, 0.004715f, -0.007004f, -0.007565f, -0.002895f, 0.007523f, 0.007027f, 0.001487f,
      -0.003323f, 0.004665f, 0.007706f, 0.009186f, 0.00814f, -0.003918f, -0.002062f, 0.00514f, 0.00858f,
      0.00251f, 0.007576f, -0.008736f, 0.001245f, -0.007298f, -0.006157f, 0.00719f, -0.008446f, -0.00864f,
      0.006535f, -0.00002605f, 0.003567f, 0.002258f, 0.003443f, -0.006207f, 0.00934f, 0.007515f, -0.00916f,
      0.00861f, -0.00939f, 0.008644f, 0.00656f, 0.001708f, 0.007935f, -0.001997f, 0.002934f, 0.001758f,
      0.004932f, 0.005432f, 0.007812f, 0.00046f, -0.00562f, 0.009186f, 0.002731f, -0.00234f, 0.00913f,
      0.006542f, -0.001783f, 0.001575f, 0.003267f, 0.00676f, 0.00647f, -0.002998f, 0.00408f, -0.002005f,
      0.002071f, 0.0001754f, -0.003132f, 0.009705f, -0.003107f, 0.00847f, -0.006504f, -0.0005784f, -0.004715f,
      -0.008415f, -0.005634f, -0.00926f, -0.006958f, 0.004932f, 0.0076f, 0.008896f, 0.006042f, 0.001687f,
      0.000543f, 0.005047f, -0.002184f, 0.003963f, 0.00716f, 0.003468f, -0.003925f, 0.0073f, 0.00385f,
      0.002712f, -0.00893f, -0.00004303f, -0.00814f, 0.00937f, 0.0017395f, 0.00555f, 0.005833f, -0.001491f,
      -0.00863f, 0.00947f, 0.001972f, -0.00984f, 0.004642f, 0.003994f, 0.00923f, -0.00984f, 0.0049f,
      -0.00987f, -0.009834f, -0.0005865f, -0.006485f, -0.0005198f, 0.00919f, 0.0004432f, 0.001068f, 0.009254f,
      -0.00881f, -0.003483f, 0.00565f, -0.007793f, -0.00989f, -0.00908f, 0.00276f, -0.002663f, -0.006893f,
      0.006332f, -0.004177f, 0.006104f, -0.00004715f, -0.003693f, 0.003576f, 0.00255f, -0.00928f, -0.002916f,
      -0.007755f, -0.00729f, -0.0061f, 0.006523f, 0.00254f, 0.0008516f, -0.0003228f, -0.004017f, -0.007374f,
      -0.005207f, 0.009056f, -0.002869f, 0.004906f, 0.007675f, 0.003086f, -0.008026f, -0.00861f, -0.006744f,
      0.0002438f, 0.00375f, 0.003315f, 0.00235f, 0.006836f, -0.005516f, 0.00434f, -0.004208f, 0.002483f,
      0.006413f, 0.00674f, 0.005604f, -0.002977f, -0.00732f, -0.00908f, 0.007484f, 0.004456f, -0.00822f,
      0.007442f, -0.003195f, 0.005753f, 0.007698f, -0.006397f, -0.00785f, -0.009605f, -0.00419f, 0.00676f,
      -0.00833f, -0.00997f, -0.0003414f, 0.00906f, -0.0071f, -0.006092f, -0.00885f, -0.007866f, 0.000824f,
      -0.003231f, -0.0006027f, 0.0074f, 0.00764f, 0.005795f, 0.002886f, 0.00958f, -0.007668f, 0.004158f,
      0.00622f, 0.00119f, 0.00277f, -0.00571f, -0.0006685f, -0.006645f, 0.0004497f, 0.00218f, -0.00405f,
      0.00485f, -0.007504f, -0.001411f, -0.001933f, -0.009964f, 0.002077f, 0.00159f, -0.002796f, 0.005787f,
      0.00335f, 0.001426f, -0.005413f, 0.00994f, 0.001742f, -0.00715f, -0.0099f, 0.007744f, -0.0008388f,
      -0.000603f, -0.002f, 0.005436f, 0.00178f, 0.009796f, -0.001966f, -0.007397f, -0.001909f, 0.00931f,
      0.0003397f, -0.006817f, 0.0069f, 0.002558f, 0.00808f, -0.007313f, -0.00984f, -0.00001967f, 0.002756f,
      0.009995f, -0.00715f, 0.004765f, -0.006096f, 0.004337f, 0.005642f, 0.00763f, 0.007103f, -0.0002332f,
      0.00322f, 0.00284f, 0.003447f, 0.0012f, -0.001126f, -0.002625f, 0.00961f, -0.005272f, 0.0053f,
      -0.002483f, -0.00931f, 0.007687f, -0.002417f, 0.004463f, 0.001136f, -0.005375f, -0.00672f, 0.007084f,
      0.0006213f, -0.00912f, 0.006542f, 0.00606f, 0.003868f, 0.001709f, -0.007484f, 0.004448f, -0.00842f,
      0.00427f, -0.00975f, 0.006847f, -0.0071f, 0.0005484f, 0.00909f, -0.004642f, 0.00564f, -0.001863f,
      -0.006863f, 0.0087f, -0.003702f, -0.001783f, -0.004032f, 0.003088f, -0.002344f, -0.00323f, -0.00966f,
      0.008286f, 0.006916f, -0.001279f, 0.003246f, 0.00921f, 0.007122f, -0.006985f, 0.0002171f, 0.000837f,
      0.001388f, 0.001075f, -0.008095f, 0.007515f, 0.00999f, 0.00423f, -0.0004835f, -0.009026f, 0.007538f,
      0.00877f, -0.002445f, 0.003437f, 0.00485f, -0.008125f, -0.007767f, 0.00934f, -0.0069f, 0.00804f,
      -0.001232f, 0.00959f, -0.007687f, 0.005993f, 0.0006413f, -0.00814f, -0.002447f, -0.001008f, 0.002981f,
      0.001682f, 0.004833f, -0.00382f, -0.0008454f, -0.006485f, 0.00567f, 0.004982f, -0.00484f, 0.00922f,
      -0.004585f, 0.00426f, 0.0004027f, 0.0006104f, -0.0063f, -0.00273f, -0.006138f, 0.005367f, -0.004303f,
      0.001937f, -0.003523f, 0.007137f, -0.005737f, -0.00869f, -0.00481f, -0.00816f, 0.0002303f, -0.0002975f,
      -0.002365f, 0.00207f, -0.004353f, -0.00924f, 0.00395f, -0.00843f, -0.0043f, -0.0004406f, 0.004807f,
      -0.00694f, 0.001308f, -0.000525f, 0.000463f, -0.006813f, 0.00775f, 0.006725f, -0.00984f, -0.0003664f,
      0.009964f, 0.007217f, 0.001767f, -0.004524f, 0.002432f, 0.000869f, -0.005993f, 0.007275f, -0.001423f,
      -0.00711f, -0.001464f, 0.007347f, -0.004776f, 0.00513f, -0.00942f, 0.003813f, -0.00489f, -0.00835f,
      -0.00711f, -0.002565f, 0.004646f, 0.002693f, 0.000531f, -0.001337f, -0.0008225f, 0.0005493f, -0.003017f,
      0.003242f, -0.00651f, 0.00958f, 0.006573f, -0.00635f, 0.008f, -0.004864f, 0.003464f, -0.007538f,
      -0.00917f, -0.002682f, 0.00431f, -0.00604f, 0.002548f, 0.000772f, -0.00769f, -0.002756f, 0.004482f,
      0.001484f, -0.004753f, -0.003052f, 0.0002143f, 0.003023f, 0.002924f, 0.00821f, 0.004673f, 0.003557f,
      0.0092f, -0.00654f, 0.001993f, 0.00835f, -0.008736f, -0.0003886f, -0.00677f, 0.0004423f, -0.00723f,
      -0.002926f, 0.00994f, 0.00784f, 0.001214f, 0.00311f, 0.003584f, 0.00856f, 0.001752f, -0.004345f,
      -0.003647f, 0.00984f, -0.006798f, 0.001661f, 0.0005393f, 0.0004299f, 0.001711f, -0.006824f, 0.003633f,
      0.00506f, -0.002146f, 0.005653f, -0.00959f, 0.0009027f, -0.009674f, 0.002176f, -0.002815f, -0.007362f,
      -0.0002565f, -0.005466f, 0.006443f, 0.00541f, -0.006615f, -0.00668f, 0.0000291f, -0.00249f, -0.00648f,
      0.006466f, 0.005596f, -0.00963f, -0.00289f, -0.007336f, 0.001666f, -0.001227f, 0.008835f, -0.00396f,
      -0.001764f, -0.00962f, -0.00461f, 0.00488f, 0.00606f, -0.00959f, 0.005497f, 0.003384f, -0.002548f,
      0.00479f, 0.001423f, -0.004772f, 0.0001752f, 0.00884f, 0.0069f, 0.00792f, -0.001779f, 0.0007215f,
      -0.007557f, 0.004314f, -0.006527f, -0.00513f, -0.00855f, -0.00873f, -0.00709f, -0.007538f, -0.002918f,
      -0.00867f, 0.000341f, 0.0004723f, 0.007336f, -0.0009327f, -0.005554f, 0.007065f, 0.00586f, -0.003202f,
      -0.001984f, -0.007755f, 0.006268f, 0.003624f, 0.001136f, 0.002611f, -0.007374f, -0.00522f, 0.005642f,
      0.003551f, 0.005558f, 0.00512f, -0.001255f, 0.00445f, 0.006657f, -0.003395f, -0.0000211f, -0.00948f,
      -0.00525f, 0.007614f, 0.007603f, -0.00872f, 0.00983f, -0.0059f, 0.005405f, -0.005775f, 0.001911f,
      -0.006306f, -0.008446f, 0.006702f, 0.001295f, -0.007904f, -0.00613f, -0.00737f, 0.004997f, 0.00699f,
      -0.008514f, 0.001029f, 0.008705f, 0.00543f, -0.0097f, -0.00839f, 0.00201f, 0.00319f, -0.00767f,
      0.003147f, -0.00936f, 0.003647f, 0.007465f, 0.00802f, 0.001254f, 0.00955f, 0.006344f, -0.00754f,
      0.007072f, -0.007305f, -0.002403f, -0.006702f, -0.00827f, 0.007183f, -0.001834f, -0.0057f, -0.007095f,
      0.00332f, -0.0008297f, 0.004333f, 0.0008926f, -0.00629f, 0.007393f, 0.006477f, -0.004684f, 0.002182f,
      -0.004246f, 0.007324f, 0.001202f, 0.00993f, 0.001759f, -0.001665f, 0.0067f, 0.003798f, -0.007454f,
      -0.00821f, 0.001178f, 0.004494f, 0.00384f, 0.003609f, 0.007614f, 0.00976f, -0.00918f, -0.002209f,
      0.002016f, 0.0093f, 0.00638f, -0.007572f, -0.008224f, -0.000771f, -0.00854f, -0.001513f, -0.007267f,
      0.00887f, -0.00107f, -0.007755f, 0.004757f, 0.002693f, 0.00439f, -0.00927f, -0.003588f, 0.001711f,
      -0.002756f, 0.00974f, -0.004158f, 0.00621f, 0.00451f, 0.007935f, -0.002064f, -0.0081f, -0.00682f,
      0.006042f, -0.003317f, -0.003391f, -0.00688f, -0.00743f, 0.003933f, -0.00816f, -0.003164f, -0.001821f,
      -0.001942f, 0.005005f, 0.00597f, 0.00595f, 0.0086f, 0.003202f, -0.00803f, -0.00892f, -0.002626f,
      0.000533f, -0.002506f, 0.00506f, -0.00822f, -0.000431f, 0.000955f, -0.004826f, -0.006626f, 0.001367f,
      0.00684f, -0.00793f, 0.00634f, -0.004353f, -0.00682f, 0.00657f, 0.000718f, -0.002306f, 0.006966f,
      0.006992f, -0.006275f, 0.00843f, 0.004826f, 0.00886f, -0.004f, 0.00901f, 0.005543f, -0.00566f,
      -0.009575f, 0.005444f, 0.00633f, -0.005756f, 0.007687f, 0.001801f, -0.005802f, 0.001708f, -0.004517f,
      0.00808f, 0.00984f, -0.00847f, 0.00959f, -0.002443f, -0.001829f, -0.003305f, -0.00392f, -0.006924f,
      -0.002266f, 0.001481f, 0.001099f, -0.00549f, 0.004787f, 0.00784f, -0.008514f, -0.00288f, -0.00858f,
      0.003025f, 0.002846f, 0.001469f, 0.00927f, 0.006443f, -0.00908f, 0.009445f, -0.009636f, -0.0003245f,
      0.003815f, -0.0001975f, 0.0007005f, -0.00984f, 0.0005784f, 0.0006576f, -0.00885f, 0.001424f, -0.004414f,
      0.006252f, 0.002722f, -0.002953f, 0.001995f, 0.00942f, -0.0000668f, -0.007507f, 0.00201f, 0.00344f,
      0.002167f, -0.001902f, -0.00691f, 0.00427f, -0.006607f, -0.003334f, -0.00143f, -0.00676f, 0.00736f,
      0.005222f, -0.0004745f, -0.0005236f, -0.00818f, 0.004253f, -0.002077f, 0.007355f, -0.00157f, -0.004112f,
      -0.007156f, 0.002766f, -0.001808f, -0.003685f, 0.002918f, 0.005814f, 0.00126f, 0.00001913f, -0.0002122f,
      0.00882f, -0.001593f, 0.005184f, -0.006516f, 0.00848f, 0.00835f, -0.0006256f, 0.00252f, -0.00594f,
      0.005688f, -0.0089f, 0.000832f, 0.00922f, 0.002317f, -0.003725f, -0.005905f, 0.001728f, 0.002249f,
      -0.00986f, 0.008896f, -0.0001637f, 0.006817f, -0.0092f, 0.008484f, -0.00751f, -0.002232f, 0.007233f,
      0.001008f, 0.003746f, -0.005726f, 0.006203f, 0.000586f, 0.00568f, 0.000979f, -0.00249f, -0.004295f,
      -0.005775f, 0.0093f, -0.002306f, 0.002426f, -0.00712f, 0.004265f, 0.00659f, -0.00504f, -0.002317f,
      0.003494f, -0.005882f, 0.00602f, 0.001864f, -0.008255f, 0.00559f, -0.001576f, -0.004242f, -0.005627f,
      0.00521f, 0.003729f, -0.005524f, -0.005688f, 0.00695f, -0.00475f, -0.0001157f, -0.007744f, -0.007935f,
      0.006092f, -0.007626f, 0.002676f, -0.00196f, -0.00932f, -0.001797f, -0.0081f, -0.004524f, 0.002777f,
      -0.0007696f, -0.0008817f, -0.00864f, 0.00834f, -0.001039f, -0.00899f, -0.007412f, -0.002197f, 0.003298f,
      0.00258f, 0.00667f, 0.001576f, -0.002626f, -0.001692f, -0.00107f, -0.001912f, -0.00997f, -0.005493f,
      -0.0098f, -0.001864f, 0.001933f, 0.005116f, -0.00938f, -0.0002704f, 0.0001253f, -0.00465f, -0.00414f,
      0.001244f, 0.002434f, -0.003223f, 0.007835f, 0.004036f, 0.00748f, -0.00903f, 0.004265f, -0.002977f,
      0.00732f, 0.006317f, -0.00563f, -0.008224f, -0.00867f, 0.00962f, -0.005325f, 0.00101f, -0.00856f,
      0.00735f, -0.00862f, -0.003899f, -0.004925f, 0.0069f, 0.004513f, -0.009895f, -0.00239f, 0.00992f,
      -0.00149f, 0.001915f, -0.002604f, 0.0095f, 0.007416f, 0.005016f, -0.002281f, 0.008125f, -0.009476f,
      -0.009056f, 0.003843f, -0.002602f, 0.0089f, 0.003674f, 0.00132f, -0.00627f, 0.009186f, 0.006226f,
      -0.00442f, 0.003323f, -0.00282f, 0.00831f, 0.007153f, -0.00724f, -0.002815f, -0.0001028f, 0.00809f,
      0.00871f, -0.007435f, -0.0018835f, 0.002344f, 0.00975f, -0.00286f, -0.00835f, -0.003582f, -0.000401f,
      0.007904f, -0.00499f, 0.003502f, -0.009605f, 0.00367f, -0.007473f, -0.003994f, -0.002377f, 0.00413f,
      0.000489f, 0.005356f, 0.00786f, 0.00476f, -0.005436f, 0.005947f, -0.000724f, -0.00671f, 0.00569f,
      -0.00826f, 0.00846f, -0.006634f, -0.003334f, -0.00802f, -0.007126f, -0.001247f, 0.00596f, -0.009056f,
      0.0005774f, -0.00648f, 0.006126f, -0.00668f, -0.004116f, -0.0002975f, 0.0002549f, -0.006977f, 0.002117f,
      0.0007377f, -0.00803f, -0.003365f, 0.00819f, -0.002949f, -0.00969f, 0.006794f, -0.007645f, -0.00099f,
      0.006966f, 0.009735f, 0.002426f, 0.005592f, 0.0003273f, -0.003353f, -0.002249f, -0.00514f, -0.002508f,
      -0.008156f, -0.000979f, 0.0002344f, -0.006508f, 0.00781f, 0.001318f, -0.00498f, 0.00858f, -0.003828f,
      -0.00504f, 0.00639f, -0.002424f, 0.002552f, 0.003736f, -0.00797f, 0.00761f, 0.006474f, 0.004166f,
      -0.009026f, 0.00638f, 0.0097f, -0.007202f, -0.008224f, -0.005714f, 0.001017f, 0.004894f, -0.00898f,
      0.00874f, -0.004066f, -0.002527f, 0.000754f, -0.002802f, 0.009315f, -0.00817f, -0.008705f, -0.0006857f,
      0.006992f, 0.000913f, 0.005993f, 0.005013f, 0.009346f, -0.00574f, 0.008575f, 0.004166f, -0.00604f,
      -0.0032f, 0.0014925f, 0.008865f, -0.006435f, -0.004417f, 0.000921f, 0.00928f, -0.001739f, 0.000586f,
      0.007904f, 0.007347f, 0.00331f, -0.0078f, -0.004005f, 0.0074f, -0.005825f, -0.007244f, -0.002626f,
      -0.005917f, 0.006508f, 0.007263f, -0.001506f, -0.003498f, 0.00693f, 0.004097f, 0.00934f, -0.003752f,
      -0.006752f, 0.001534f, 0.003906f, 0.001351f, 0.00367f, 0.0086f, -0.00536f, -0.001699f, 0.001546f,
      -0.00277f, -0.0005455f, -0.002718f, -0.00583f, -0.0009003f, -0.001003f, 0.001612f, -0.003557f, -0.006004f,
      0.001006f, -0.00925f, -0.0008187f, -0.002907f, 0.003675f, 0.00394f, 0.005608f, -0.007133f, 0.001691f,
      0.006428f, 0.003813f, -0.00542f, -0.00583f, 0.002207f, -0.001088f, 0.00714f, 0.006233f, 0.002617f,
      -0.00419f, -0.00916f, 0.004063f, -0.002892f, 0.000514f, 0.00224f, 0.0001853f, 0.0007997f, 0.0005536f,
      -0.00639f, -0.007015f, 0.00309f, 0.006184f, -0.00982f, -0.0002372f, 0.0009604f, 0.00962f, 0.00678f,
      -0.006653f, -0.004955f, -0.0003958f, 0.006428f, 0.004517f, 0.00672f, 0.003792f, -0.006046f, -0.00221f,
      -0.00727f, -0.00748f, -0.004204f, -0.00982f, 0.0007663f, 0.00661f, -0.003647f, -0.006973f, 0.002605f,
      0.0001023f, 0.004536f, -0.00647f, 0.009735f, 0.00945f, -0.00967f, -0.0003023f, -0.0086f, 0.008736f,
      -0.00701f, 0.00258f, -0.002716f, -0.00162f, -0.006996f, 0.007664f, 0.007595f, 0.00403f, 0.00233f,
      -0.00481f, -0.001349f, -0.005196f, -0.009026f, 0.00606f, 0.001146f, 0.001434f, 0.00967f, 0.004448f,
      -0.004837f, -0.007168f, -0.005234f, 0.002514f, 0.005306f, 0.003088f, 0.0018215f, -0.00558f, -0.006596f,
      0.002018f, -0.003408f, -0.001384f, -0.006065f, -0.001212f, -0.002604f, -0.00767f, 0.0001342f, -0.00851f,
      -0.00392f, 0.003862f, 0.00701f, 0.003605f, -0.00965f, -0.00714f, 0.00956f, -0.00888f, -0.001019f,
      0.0024f, -0.00961f, -0.005238f, 0.005333f, 0.00871f, 0.007607f, -0.00756f, -0.004772f, -0.00912f,
      -0.004047f, 0.003483f, 0.003294f, 0.006577f, -0.005505f, 0.00996f, 0.009964f, 0.0004187f, 0.005898f,
      0.00796f, -0.00165f, -0.003225f, -0.001258f, 0.00853f, -0.008865f, 0.00815f, -0.001117f, -0.00685f,
      0.001974f, 0.00915f, 0.00667f, 0.009605f, 0.007107f, 0.007698f, -0.004387f, -0.0003958f, -0.005062f,
      0.002188f, -0.004875f, -0.002922f, -0.0003638f, 0.006268f, 0.00785f, 0.006138f, 0.000505f, -0.0003953f,
      -0.00841f, -0.00958f, -0.007126f, -0.003107f, 0.0078f, 0.003452f, -0.009254f, -0.00117f, -0.00878f,
      -0.00911f, -0.0004418f, 0.00831f, -0.004524f, 0.003872f, 0.0044f, 0.006424f, 0.000634f, -0.004883f,
      -0.002487f, -0.00512f, -0.00692f, -0.00521f, -0.001761f, 0.008575f, -0.006393f, 0.00351f, 0.00914f,
      -0.006035f, -0.002264f, -0.009636f, 0.00918f, -0.00967f, -0.004944f, -0.0004587f, -0.002478f, -0.00814f,
      0.00816f, -0.004776f, 0.00954f, 0.003471f, -0.006172f, 0.003603f, 0.009346f, 0.00455f, 0.00982f,
      0.00476f, 0.0007815f, -0.003096f, -0.0000307f, -0.005608f, 0.009315f, 0.00374f, -0.007366f, -0.001133f,
      -0.00944f, 0.006847f, 0.00631f, 0.005394f, 0.003088f, -0.00644f, -0.0004168f, -0.00923f, -0.003254f,
      -0.005077f, 0.00637f, -0.001415f, -0.003235f, -0.001729f, -0.0082f, 0.006664f, -0.006f, 0.00663f,
      -0.001547f, -0.004116f, -0.00542f, 0.00521f, -0.00286f, -0.00396f, 0.004547f, -0.0001363f, 0.000979f,
      -0.00634f, -0.006767f, -0.0000603f, 0.008316f, 0.00756f, -0.004993f, -0.00645f, -0.002295f, 0.004288f,
      0.00901f, 0.008194f, -0.004192f, -0.002182f, -0.005836f, -0.003983f, -0.007183f, -0.0061f, 0.001098f,
      -0.0009274f, 0.005207f, 0.0002102f, -0.003925f, 0.0056f, -0.00296f, 0.006134f, -0.007744f, 0.006126f,
      -0.005047f, -0.006134f, 0.004818f, -0.005283f, 0.005272f, -0.00779f, -0.003086f, -0.000607f, 0.005486f,
      0.0005345f, -0.007305f, -0.0048f, -0.00876f, -0.00433f, 0.006165f, -0.002474f, -0.00953f, -0.002066f,
      0.002918f, 0.006382f, 0.003317f, 0.00826f, -0.009995f, 0.004143f, 0.00985f, -0.0002116f, -0.002989f,
      -0.007805f, 0.0003633f, -0.00365f, -0.00916f, 0.009834f, 0.003513f, -0.00379f, 0.00736f, -0.00957f,
      0.005726f, -0.00772f, -0.00803f, -0.002052f, -0.005585f, -0.00781f, -0.00599f, 0.00954f, 0.002024f,
      -0.005745f, 0.003054f, -0.009415f, -0.005054f, 0.00424f, 0.003218f, 0.00826f, 0.00817f, -0.00409f,
      0.00518f, 0.00216f, 0.006756f, -0.00411f, -0.003344f, -0.004898f, 0.00001055f, 0.006104f, 0.001057f,
      -0.000702f, 0.00771f, -0.001743f, 0.001407f, -0.005104f, -0.007717f, -0.002026f, -0.006405f, 0.00886f,
      0.0006466f, -0.00951f, -0.00395f, -0.00814f, 0.00936f, 0.001143f, -0.00485f, 0.00584f, -0.002224f,
      0.00834f, -0.0003467f, -0.000945f, 0.007034f, -0.0009427f, -0.009445f, 0.0007753f, -0.006973f, 0.001507f,
      0.004105f, -0.002523f, 0.002872f, -0.001515f, -0.00869f, 0.003103f, 0.000389f, -0.00774f, 0.00441f,
      -0.001002f, 0.00783f, -0.001102f, -0.003883f, -0.007187f, -0.0001678f, -0.00742f, -0.00686f, -0.006702f,
      0.00894f, -0.0003886f, -0.005543f, 0.00988f, 0.00411f, -0.00002635f, 0.00851f, -0.002317f, 0.00873f,
      -0.00532f, -0.000835f, -0.004166f, -0.004036f, -0.003325f, -0.00799f, 0.003025f, 0.001356f, -0.009575f,
      -0.00426f, -0.003431f, 0.00899f, -0.001455f, -0.0007324f, -0.00492f, 0.00989f, -0.0002503f, -0.00814f,
      -0.00535f, -0.0035f, -0.001434f, 0.00635f, -0.005108f, 0.002626f, 0.00983f, 0.00672f, -0.00725f,
      -0.004826f, 0.007275f, -0.006763f, 0.002605f, 0.002369f, -0.000976f, 0.00263f, 0.00465f, -0.009544f,
      -0.0008945f, -0.00175f, -0.00799f, -0.0006666f, -0.00514f, -0.002842f, -0.001805f, 0.000992f, 0.00844f,
      -0.000964f, -0.00636f, 0.001281f, 0.001717f, 0.00569f, 0.005917f, -0.00826f, -0.00859f, 0.004246f,
      0.004078f, -0.005566f, 0.00835f, -0.006893f, -0.005867f, 0.001273f, -0.005856f, 0.004448f, 0.004562f,
      -0.00392f, -0.00855f, 0.0005975f, 0.006817f, -0.005524f, -0.0009527f, -0.00695f, -0.002172f, -0.003683f,
      -0.00546f, 0.007698f, -0.00858f, 0.003372f, 0.001414f, -0.007786f, -0.00482f, 0.0083f, 0.007534f,
      0.00554f, 0.005768f, 0.001982f, -0.004597f, -0.001634f, 0.000563f, 0.00298f, 0.001768f, 0.0004673f,
      0.009285f, 0.00518f, 0.00798f, 0.00557f, -0.002504f, -0.00777f, 0.007904f, 0.00939f, -0.004646f,
      0.00527f, 0.00817f, 0.00526f, 0.007935f, -0.00413f, -0.002628f, -0.008194f, -0.006195f, 0.00884f,
      0.007282f, 0.003819f, -0.00904f, 0.001354f, -0.004368f, -0.0002527f, 0.004684f, -0.002907f, -0.003862f,
      -0.002197f, 0.00858f, -0.00989f, 0.0004277f, 0.008484f, -0.008865f, 0.007275f, 0.00869f, -0.0000226f,
      0.0006456f, 0.0002527f, 0.003267f, 0.007793f, -0.001359f, -0.007423f, -0.004204f, 0.006824f, -0.00801f,
      0.006992f, -0.002182f, 0.00181f, 0.00966f, -0.00888f, -0.006527f, -0.00873f, -0.004623f, -0.006767f,
      -0.006317f, 0.003017f, 0.002218f, 0.00805f, -0.00677f, -0.00974f, -0.0083f, 0.008095f, -0.00424f,
      -0.009636f, 0.002298f, -0.00864f, 0.004044f, 0.000354f, 0.00949f, 0.00635f, 0.009026f, -0.00806f,
      -0.0008893f, 0.002377f, -0.001343f, -0.001965f, -0.00442f, -0.006615f, -0.004166f, 0.00719f, -0.006306f,
      -0.009674f, -0.00787f, 0.00712f, -0.003637f, 0.0008287f, 0.005352f, -0.004227f, -0.00549f, -0.0058f,
      0.00489f, -0.005165f, 0.001942f, 0.00591f, 0.00612f, 0.005306f, -0.00723f, 0.0051f, 0.002329f,
      -0.001097f, 0.002022f, -0.006416f, -0.006577f, 0.003603f, 0.004303f, 0.007652f, 0.00884f, -0.003191f,
      0.002787f, -0.009254f, 0.003475f, -0.002266f, 0.00936f, -0.00793f, -0.00738f, 0.008194f, 0.003998f,
      -0.0049f, 0.008965f, -0.000592f, 0.00711f, 0.00905f, -0.0006223f, -0.00735f, -0.00399f, -0.00808f,
      -0.005367f, 0.00705f, 0.0007415f, 0.00864f, 0.00883f, -0.001155f, 0.00898f, 0.004406f, 0.00967f,
      0.004677f, -0.003113f, -0.0009146f, 0.00756f, 0.005733f, -0.003647f, 0.00446f, 0.00798f, 0.003305f,
      -0.0000515f, -0.003746f, -0.002283f, -0.004913f, 0.003496f, -0.00773f, -0.003622f, 0.004974f, 0.00244f,
      0.001445f, -0.004826f, 0.002394f, 0.003075f, -0.0006714f, 0.002077f, -0.008675f, -0.001683f, 0.006065f,
      -0.005512f, -0.001691f, 0.007507f, -0.00913f, -0.0008674f, -0.005f, 0.001398f, -0.004875f, -0.000567f,
      -0.002668f, 0.001711f, -0.005306f, -0.00883f, -0.001738f, 0.0035f, -0.006702f, -0.006943f, 0.00884f,
      -0.001516f, 0.00991f, 0.003082f, 0.006077f, -0.00437f, -0.000524f, -0.003986f, 0.007393f, 0.00986f,
      -0.0008f, -0.001425f, -0.001999f, -0.002277f, -0.00901f, 0.004986f, 0.002085f, -0.0009236f, 0.001841f,
      -0.003191f, 0.002205f, -0.00781f, 0.00397f, -0.002066f, -0.008835f, -0.004585f, -0.00953f, -0.006496f,
      -0.006996f, 0.007233f, -0.00544f, 0.001037f, 0.0028f, -0.007935f, -0.0055f, -0.007866f, 0.00436f,
      -0.0009565f, 0.001419f, 0.007587f, -0.0004418f, -0.00318f, -0.003857f, 0.007763f, 0.008896f, 0.004925f,
      0.00979f, -0.00928f, -0.001149f, 0.00678f, -0.002733f, -0.002972f, -0.001726f, 0.006706f, -0.001256f,
      -0.00636f, 0.0004964f, 0.0005093f, -0.0008807f, 0.002026f, 0.00215f, -0.007603f, -0.00936f, -0.001715f,
      -0.000935f, 0.0005236f, 0.000975f, 0.00786f, -0.002583f, 0.003407f, -0.002033f, -0.00217f, 0.001398f,
      -0.0001027f, 0.0009203f, 0.0009117f, 0.00741f, -0.003925f, 0.0007577f, -0.006317f, 0.001241f, 0.005623f,
      0.001732f, 0.00374f, 0.00341f, 0.006714f, 0.001987f, -0.0037f, 0.00349f, -0.00431f, -0.00895f,
      -0.009605f, -0.007214f, -0.00393f, -0.002583f, 0.00841f, 0.00782f, -0.005657f, -0.00655f, 0.003542f,
      -0.004143f, 0.003202f, -0.002695f, 0.0002656f, 0.001797f, -0.0065f, 0.00628f, -0.0001239f, -0.002842f,
      0.00119f, -0.00979f, 0.006287f, -0.00646f, 0.00769f, 0.00831f, -0.0055f, 0.0005436f, 0.006554f,
      -0.0001364f, 0.00699f, 0.004364f, -0.00227f, 0.00489f, 0.0026f, 0.0007696f, 0.0004685f, -0.001103f,
      0.001123f, -0.002245f, 0.006527f, -0.00828f, -0.002954f, -0.005226f, -0.005814f, -0.0002468f, -0.00884f,
      0.008606f, -0.00001067f, -0.00417f, -0.003376f, 0.00918f, -0.00776f, 0.002684f, 0.006145f, -0.0006285f,
      -0.004173f, -0.004917f, -0.00678f, 0.00248f, 0.007263f, 0.002188f, -0.000213f, 0.00413f, 0.002676f,
      0.004948f, 0.007614f, -0.001845f, -0.00436f, 0.00591f, 0.004833f, -0.002085f, -0.006096f, 0.007378f,
      0.001922f, 0.006573f, 0.0016985f, 0.001776f, 0.00993f, -0.00829f, -0.0001675f, 0.004753f, 0.00008494f,
      0.00989f, 0.0008593f, 0.00636f, 0.0008297f, -0.00482f, -0.001189f, -0.001576f, -0.001331f, -0.00881f,
      0.00416f, -0.0008516f, -0.002281f, -0.00399f, -0.00603f, 0.0031f, 0.00994f, 0.0009284f, -0.00446f,
      -0.00944f, 0.00272f, -0.001006f, -0.006733f, 0.00815f, 0.004932f, -0.004894f, 0.007156f, 0.0001193f,
      -0.00745f, -0.000041f, -0.004074f, 0.00829f, 0.006042f, 0.006176f, -0.00509f, 0.005375f, -0.00554f,
      -0.001078f, -0.002928f, 0.00813f, 0.004013f, 0.002552f, -0.0086f, 0.000254f, -0.005844f, 0.004093f,
      -0.008224f, 0.006016f, -0.004883f, -0.006504f, 0.0003617f, -0.00008327f, 0.00382f, 0.00786f, -0.00915f,
      -0.004963f, 0.003756f, 0.00689f, 0.00833f, 0.005455f, -0.00871f, -0.00872f, -0.0008054f, 0.001023f,
      -0.003527f, -0.00735f, 0.00691f, 0.0092f, 0.004837f, -0.000847f, 0.0006146f, -0.00829f, 0.007317f,
      -0.002722f, 0.005962f, -0.004005f, 0.002857f, 0.004414f, 0.00437f, -0.003452f, -0.004383f, -0.004654f,
      0.007866f, -0.000736f, -0.001158f, -0.005924f, -0.002207f, 0.00904f, 0.004505f, 0.005688f, 0.003448f,
      -0.00414f, -0.00986f, -0.007446f, 0.00479f, 0.00314f, 0.0084f, 0.005714f, -0.002865f, 0.0008903f,
      -0.00831f, 0.009415f, -0.001098f, 0.0007825f, -0.002136f, -0.009995f, 0.00798f, -0.0002449f, -0.00454f,
      0.00262f, -0.001926f, 0.003874f, -0.001987f, 0.00456f, 0.00994f, -0.00275f, -0.0013485f, 0.00911f,
      -0.0011f, -0.005253f, 0.003504f, 0.004726f, -0.00821f, 0.00008196f, 0.004696f, 0.00473f, 0.006893f,
      -0.002386f, 0.007145f, 0.007584f, -0.00542f, -0.000596f, 0.002354f, 0.001427f, 0.001673f, 0.004646f,
      0.004826f, 0.00847f, -0.005226f, 0.0003307f, 0.00536f, 0.002802f, 0.006264f, -0.000479f, 0.00222f,
      0.00817f, 0.005253f, 0.005257f, 0.001163f, 0.005417f, 0.006603f, 0.00514f, -0.003473f, -0.001948f,
      0.006695f, 0.003492f, 0.000456f, 0.00933f, 0.00283f, 0.006935f, -0.004658f, -0.0008807f, -0.001274f,
      -0.0006485f, -0.00349f, -0.002163f, 0.00811f, 0.001358f, -0.002134f, 0.0005803f, -0.001573f, -0.005478f,
      -0.00496f, 0.00968f, 0.001645f, -0.005756f, -0.0008974f, -0.00608f, -0.00528f, 0.00005585f, -0.005756f,
      -0.004025f, -0.00772f, -0.0008974f, 0.00786f, 0.00396f, -0.008865f, -0.00645f, -0.00903f, 0.00802f,
      -0.001602f, -0.0072f, 0.00736f, 0.002499f, -0.00839f, -0.00925f, 0.005943f, -0.00785f, -0.0081f,
      -0.00802f, -0.005554f, 0.004078f, 0.009476f, -0.00877f, 0.00257f, -0.00439f, 0.006744f, -0.00419f,
      -0.005413f, 0.002476f, -0.002373f, -0.006424f, 0.008736f, 0.006977f, 0.009735f, 0.009514f, 0.0009437f,
      -0.001418f, 0.004066f, 0.004986f, -0.008644f, -0.007427f, -0.00988f, 0.006714f, -0.00118f, 0.00924f,
      0.000984f, 0.001846f, -0.00418f, 0.00341f, 0.0007763f, 0.008545f, 0.007313f, 0.00999f, -0.000682f,
      0.003416f, 0.00465f, -0.000676f, 0.00206f, -0.00654f, -0.002478f, 0.003826f, -0.001733f, -0.003693f,
      -0.001044f, -0.004696f, 0.00688f, 0.00632f, 0.004963f, -0.00365f, -0.00772f, -0.001813f, -0.004898f,
      -0.008385f, 0.002f, -0.007782f, -0.000961f, -0.003376f, 0.005157f, -0.002651f, 0.007935f, 0.003716f,
      0.009f, 0.001195f, -0.00982f, -0.00532f, -0.00828f, -0.000279f, -0.007626f, 0.00879f, 0.006996f,
      0.00942f, 0.002588f, -0.0097f, -0.00011635f, -0.001595f, -0.0006347f, -0.001799f, 0.00126f, 0.005085f,
      0.001865f, 0.003216f, -0.000628f, -0.00474f, -0.004925f, -0.00626f, 0.006287f, -0.005054f, 0.0079f,
      -0.005177f, -0.009796f, -0.00805f, -0.001599f, 0.0085f, -0.008965f, -0.002886f, -0.008606f, -0.008965f,
      0.004757f, -0.009285f, 0.00548f, 0.00816f, -0.001941f, 0.00622f, 0.00755f, 0.00926f, 0.009125f,
      -0.004364f, 0.006214f, -0.007137f, 0.001763f, -0.002035f, 0.004326f, 0.00653f, -0.007072f, -0.003609f,
      -0.00504f, 0.004448f, -0.005928f, -0.007057f, -0.002148f, -0.004593f, -0.004467f, -0.009514f, -0.00854f,
      0.001922f, 0.007572f, -0.005016f, 0.003345f, 0.008575f, -0.00967f, 0.000532f, -0.002897f, -0.005013f,
      -0.009834f, 0.00302f, 0.005688f, 0.005096f, -0.003983f, 0.00851f, 0.001554f, 0.00394f, 0.005688f,
      -0.00537f, 0.00655f, 0.007526f, 0.002298f, 0.006126f, -0.00654f, -0.003433f, -0.00818f, -0.003098f,
      -0.00822f, -0.00898f, -0.007675f, 0.005955f, -0.003288f, 0.006237f, -0.002f, 0.002678f, -0.00639f,
      0.00899f, 0.009766f, 0.009384f, -0.0001253f, 0.007263f, -0.003555f, -0.00988f, -0.00534f, -0.005356f,
      -0.00805f, 0.001697f, 0.002516f, 0.0022f, 0.007397f, -0.002075f, 0.00247f, 0.004593f, -0.00543f,
      0.000358f, -0.005047f, 0.00476f, -0.003937f, -0.002851f, 0.007507f, 0.001389f, 0.003235f, 0.00205f,
      -0.00474f, -0.0059f, 0.001666f, 0.002943f, -0.00954f, -0.00828f, -0.008804f, 0.002356f, 0.00836f,
      0.002785f, 0.00881f, -0.00716f, 0.005608f, 0.007534f, -0.00952f, 0.008965f, 0.0001839f, -0.007412f,
      0.00693f, -0.0000717f, 0.003857f, 0.00021f, 0.002897f, -0.00452f, -0.002552f, -0.005962f, -0.006737f,
      -0.0008616f, 0.008606f, -0.005814f, -0.007397f, -0.006096f, 0.0099f, -0.00955f, 0.001134f, 0.00702f,
      -0.0003154f, 0.00366f, -0.009186f, -0.001096f, 0.00984f, -0.005787f, 0.00369f, -0.001496f, 0.002462f,
      -0.00623f, -0.00426f, -0.004837f, -0.00558f, -0.003311f, -0.0066f, 0.0077f, 0.003609f, 0.004646f,
      0.007996f, -0.00788f, 0.006348f, -0.00986f, 0.00817f, 0.0001633f, 0.0001796f, -0.00899f, -0.001417f,
      0.00972f, -0.00067f, -0.005535f, 0.001376f, 0.004974f, 0.0008225f, 0.008484f, -0.00589f, 0.00828f,
      -0.007206f, -0.00599f, -0.0009503f, 0.000634f, -0.0001874f, -0.00654f, 0.00424f, -0.001244f, 0.002506f,
      -0.009964f, -0.00828f, -0.002129f, -0.003368f, -0.0003746f, -0.006798f, -0.00383f, 0.008514f, 0.00818f,
      -0.005497f, -0.0034f, -0.001681f, -0.004208f, -0.004337f, -0.0000664f, 0.003807f, -0.006073f, -0.003489f,
      -0.00521f, 0.005047f, 0.00367f, 0.005657f, -0.004665f, 0.00671f, -0.003513f, 0.00869f, 0.008095f,
      0.007545f, 0.007214f, 0.002594f, -0.001637f, 0.005642f, 0.00526f, -0.007195f, 0.00413f, -0.006878f,
      0.009224f, 0.008514f, -0.0008245f, -0.004276f, 0.003633f, -0.000534f, -0.00916f, -0.00905f, 0.00827f,
      0.00458f, 0.002428f, -0.002975f, 0.00718f, -0.00888f, 0.004597f, 0.0004854f, -0.003778f, 0.006023f,
      0.001024f, -0.00484f, -0.0048f, 0.001374f, -0.004204f, -0.004368f, 0.005783f, 0.001205f, 0.007774f,
      -0.001196f, -0.007015f, 0.00822f, -0.005875f, 0.003675f, 0.00279f, 0.001947f, -0.00342f, -0.000307f,
      -0.003113f, -0.0017185f, -0.001276f, 0.0031f, -0.003546f, -0.003328f, 0.004078f, 0.00976f, -0.002756f,
      0.00487f, -0.007904f, 0.003613f, 0.007034f, -0.00624f, 0.0007896f, -0.0077f, -0.001974f, 0.007397f,
      0.005966f, -0.00627f, -0.005215f, 0.001178f, -0.00372f, 0.001711f, -0.001743f, 0.00248f, 0.0003877f,
      0.005028f, -0.00789f, -0.0007873f, -0.005753f, 0.00961f, 0.00961f, 0.002813f, 0.002567f, -0.007095f,
      0.003628f, 0.0001531f, 0.0002968f, -0.005493f, -0.0001053f, 0.00964f, 0.004997f, -0.00657f, 0.000724f,
      -0.00563f, -0.009834f, -0.003574f, 0.003572f, -0.006805f, 0.007423f, 0.003103f, -0.005455f, 0.00881f,
      -0.00777f, -0.003508f, 0.0075f, 0.00404f, -0.00747f, 0.003056f, 0.005142f, -0.007156f, -0.00923f,
      0.00401f, 0.007442f, 0.005077f, 0.007393f, 0.004276f, -0.00851f, -0.00263f, -0.006123f, 0.003536f,
      0.005672f, 0.00887f, -0.002031f, -0.00524f, -0.001232f, 0.000433f, 0.005398f, 0.009575f, 0.009705f,
      -0.007267f, -0.00565f, -0.003963f, 0.007477f, -0.00216f, -0.007744f, -0.003347f, -0.00804f, -0.002136f,
      -0.002407f, 0.00826f, -0.006294f, 0.005116f, 0.00007975f, -0.007267f, -0.003428f, -0.005497f, 0.001562f,
      0.003801f, -0.004646f, 0.004234f, 0.00979f, 0.00943f, -0.002726f, 0.0007277f, 0.0007143f, -0.00785f,
      0.00531f, 0.00747f, -0.006287f, 0.0001854f, 0.0005198f, -0.006645f, -0.000202f, -0.0004883f, -0.001946f,
      0.00904f, 0.00122f, 0.005608f, 0.002243f, -0.001732f, -0.00844f, -0.000973f, 0.00898f, 0.00686f,
      0.005028f, 0.005497f, -0.002182f, -0.007122f, 0.00955f, 0.00725f, 0.0000116f, 0.00504f, 0.00864f,
      -0.00827f, -0.00476f, -0.001607f, 0.006145f, 0.00777f, 0.00974f, -0.002163f, 0.00857f, 0.006485f,
      -0.004356f, 0.00010043f, 0.001632f, 0.005432f, 0.00846f, -0.006756f, 0.0005136f, -0.00836f, -0.009544f,
      0.005016f, -0.002354f, -0.004543f, 0.00419f, 0.00798f, -0.001813f, 0.005913f, 0.003494f, -0.002695f,
      -0.009346f, -0.001584f, -0.00886f, -0.007374f, 0.00979f, 0.00961f, 0.0006576f, -0.0018015f, -0.009766f,
      -0.00821f, -0.00924f, 0.0002823f, 0.003115f, -0.00788f, -0.005257f, 0.003233f, -0.00939f, 0.00617f,
      0.003914f, -0.002165f, 0.004215f, 0.00603f, 0.00498f, -0.000754f, 0.0079f, 0.00463f, -0.004574f,
      -0.00494f, 0.0014715f, 0.007866f, 0.005215f, -0.00008845f, 0.00897f, -0.00431f, -0.00416f, 0.001195f,
      -0.007626f, 0.006153f, 0.000168f, -0.001373f, 0.001575f, -0.00368f, -0.00926f, 0.003387f, -0.006237f,
      -0.003305f, -0.004677f, 0.003044f, 0.002283f, -0.00855f, -0.00383f, 0.005135f, -0.003328f, -0.005f,
      -0.006145f, 0.008995f, 0.00933f, -0.0004253f, -0.00697f, -0.00895f, 0.001212f, 0.007114f, 0.005264f,
      0.003008f, -0.0087f, 0.00578f, -0.008354f, 0.009056f, 0.004955f, -0.004787f, 0.001999f, -0.008705f,
      -0.00722f, -0.00211f, 0.00471f, -0.0012245f, -0.0003836f, -0.00119f, -0.005363f, -0.00464f, -0.00628f,
      -0.00855f, -0.000797f, 0.005047f, 0.0006003f, 0.002134f, 0.001738f, 0.006653f, -0.003204f, 0.00568f,
      -0.0003297f, 0.0001493f, -0.00001603f, -0.001742f, -0.0004888f, -0.002066f, -0.003843f, 0.008514f, 0.001038f,
      -0.006084f, 0.002298f, -0.00506f, 0.0028f, -0.00588f, 0.006187f, -0.004707f, 0.00482f, -0.005604f,
      0.0099f, 0.002226f, -0.00418f, -0.00867f, -0.001959f, 0.006733f, 0.00881f, -0.009636f, 0.006523f,
      0.00918f, -0.005287f, -0.00939f, 0.007725f, 0.002266f, -0.00813f, 0.00945f, -0.009735f, 0.00804f,
      -0.00447f, -0.0006757f, -0.002113f, 0.0071f, -0.002256f, -0.001572f, -0.002722f, -0.005325f, 0.005184f,
      0.001163f, 0.00785f, -0.00908f, 0.000957f, -0.004894f, -0.00785f, 0.004192f, 0.005585f, -0.00466f,
      0.00659f, -0.009026f, 0.00393f, -0.00526f, -0.00882f, -0.006893f, -0.008286f, -0.000591f, -0.00449f,
      -0.00882f, 0.004025f, -0.00812f, 0.002306f, 0.003397f, 0.0002433f, -0.00333f, 0.000728f, -0.001259f,
      -0.0006423f, 0.00922f, 0.002346f, -0.00682f, -0.002346f, -0.00688f, 0.0004377f, 0.0007114f, -0.00878f,
      0.00824f, -0.007797f, 0.00000536f, 0.00009096f, 0.00981f, -0.001997f, -0.006676f, -0.006683f, -0.00412f,
      0.00085f, 0.004017f, 0.00645f, -0.00674f, 0.00846f, -0.00847f, -0.00199f, 0.003153f, -0.0002362f,
      -0.0004025f, 0.007996f, 0.002476f, 0.00555f, 0.003628f, -0.00508f, 0.00728f, -0.00266f, 0.003223f,
      -0.007328f, -0.00689f, -0.00229f, -0.001114f, 0.002768f, -0.001708f, 0.003847f, -0.007248f, -0.00689f,
      -0.007065f, -0.00772f, 0.005745f, 0.008804f, 0.006092f, 0.005795f, 0.001585f, 0.005386f, -0.005962f,
      -0.0004244f, 0.008804f, 0.003803f, 0.000961f, 0.00976f, 0.0005674f, 0.00905f, 0.00982f, 0.005295f,
      -0.00009507f, 0.005775f, -0.002659f, -0.001253f, -0.006416f, 0.008194f, 0.00945f, 0.006752f, -0.00935f,
      0.003845f, -0.006237f, 0.00415f, 0.008095f, -0.00645f, -0.009865f, 0.000944f, 0.00811f, 0.00841f,
      0.0002704f, -0.00681f, 0.00514f, -0.005535f, -0.00543f, -0.007355f, 0.006424f, -0.0012665f, -0.007423f,
      0.00501f, 0.0071f, -0.0001485f, -0.004772f, -0.007965f, -0.002703f, 0.00977f, -0.0002038f, 0.00664f,
      0.002275f, 0.004887f, 0.00762f, 0.001178f, 0.001114f, -0.000678f, -0.001807f, -0.004963f, 0.001163f,
      0.00273f, -0.00955f, 0.002756f, 0.0005674f, -0.00551f, -0.00862f, -0.009026f, 0.00948f, -0.00195f,
      -0.001241f, 0.00402f, 0.002943f, 0.0001924f, 0.001133f, -0.004086f, 0.002512f, -0.0058f, 0.00159f,
      -0.00808f, 0.00575f, -0.00857f, -0.00701f, 0.009544f, -0.001974f, 0.002966f, -0.004898f, -0.001783f,
      0.003128f, -0.005596f, -0.00751f, -0.004704f, 0.00719f, -0.00949f, -0.001564f, 0.003157f, 0.005245f,
      -0.00424f, 0.004654f, -0.00425f, -0.008766f, 0.00912f, -0.005386f, 0.00439f, -0.002386f, 0.00576f,
      0.003857f, -0.007004f, 0.0005574f, 0.006065f, -0.0068f, 0.00985f, -0.0003872f, -0.004654f, 0.008675f,
      0.00801f, 0.001015f, 0.0019045f, 0.007225f, 0.0004132f, -0.005173f, 0.001682f, -0.002037f, -0.003492f,
      0.003092f, 0.00231f, 0.007294f, 0.002605f, -0.00941f, -0.004112f, 0.0082f, 0.002506f, -0.00819f,
      -0.0041f, 0.009476f, 0.003584f, -0.00585f, 0.00462f, -0.006348f, 0.00913f, -0.003197f, -0.004265f,
      -0.00945f, -0.001356f, 0.007545f, 0.002289f, 0.001126f, 0.002977f, 0.00948f, -0.00703f, -0.002531f,
      -0.00868f, -0.00619f, -0.0004635f, 0.009254f, -0.0005174f, -0.00736f, -0.006264f, 0.00779f, -0.002342f,
      0.004997f, 0.00269f, 0.00509f, -0.0041f, 0.00506f, 0.002752f, -0.006416f, 0.00794f, 0.003563f,
      0.00551f, 0.006554f, -0.008286f, -0.00296f, 0.008354f, -0.0079f, -0.006348f, 0.001052f, 0.0007205f,
      0.00506f, 0.000453f, -0.00993f, -0.006424f, 0.005787f, -0.001206f, 0.00876f, -0.004513f, -0.002857f,
      -0.00701f, -0.00621f, 0.003498f, 0.00986f, -0.00846f, -0.00128f, 0.006294f, -0.003735f, 0.00843f,
      -0.00841f, -0.007465f, -0.007504f, -0.00734f, 0.00635f, 0.004498f, -0.005688f, -0.003014f, 0.00892f,
      0.00982f, 0.00793f, -0.002365f, 0.003353f, -0.004486f, -0.00651f, -0.00361f, -0.00418f, -0.00786f,
      0.007812f, 0.001912f, -0.008156f, -0.00809f, -0.001939f, -0.003836f, -0.001578f, 0.00331f, -0.008736f,
      -0.006138f, -0.00877f, -0.007595f, 0.002537f, -0.007336f, -0.006477f, -0.007767f, -0.00853f, -0.003601f,
      -0.000952f, 0.007683f, -0.006283f, 0.00796f, 0.006012f, -0.001464f, 0.00718f, -0.0025f, -0.001972f,
      0.004166f, 0.0002615f, 0.00496f, 0.006516f, 0.0016f, -0.008415f, -0.002398f, -0.001027f, 0.0000037f,
      0.00827f, 0.003153f, 0.004826f, 0.00619f, -0.00673f, -0.00834f, -0.001702f, 0.006664f, -0.00465f,
      -0.00909f, 0.003893f, 0.005188f, 0.009415f, 0.00191f, 0.00274f, -0.002968f, -0.003834f, 0.00495f,
      0.005985f, -0.002945f, 0.007317f, -0.00934f, -0.001007f, -0.005333f, -0.008415f, -0.0067f, 0.006084f,
      0.00689f, -0.002855f, -0.009254f, -0.00402f, 0.007694f, -0.007633f, -0.008865f, -0.00846f, 0.007317f,
      -0.00915f, -0.009476f, 0.002455f, -0.001528f, -0.001358f, 0.0016985f, -0.001466f, -0.002584f, -0.006992f,
      -0.00427f, 0.000739f, 0.00258f, 0.0042f, 0.001303f, 0.00963f, 0.002176f, -0.00952f, 0.0005264f,
      -0.005226f, 0.008804f, 0.005707f, -0.00763f, -0.00875f, 0.0002716f, -0.00251f, -0.00646f, -0.00666f,
      0.00936f, 0.005447f, -0.00562f, 0.00967f, 0.001811f, -0.00963f, 0.001052f, 0.00807f, -0.002794f,
      -0.00845f, 0.00685f, -0.003199f, 0.003119f, -0.004333f, 0.0001079f, -0.00884f, -0.002384f, -0.0008464f,
      -0.0053f, 0.0008607f, 0.005f, 0.003218f, -0.001972f, 0.003925f, -0.000635f, -0.003868f, -0.00636f,
      -0.005894f, 0.0005355f, 0.00921f, -0.006687f, 0.00629f, -0.001168f, -0.00646f, 0.005547f, -0.00963f,
      -0.004078f, 0.002125f, 0.008995f, 0.006187f, 0.007397f, -0.00656f, -0.006527f, 0.006042f, -0.001503f,
      -0.00624f, 0.003023f, -0.009995f, -0.002466f, 0.00351f, 0.003439f, -0.003235f, 0.008026f, 0.004158f,
      0.003117f, -0.005856f, 0.00461f, -0.001134f, 0.004257f, 0.00933f, 0.00992f, -0.008156f, 0.004356f,
      0.00917f, -0.007904f, 0.0004003f, -0.00912f, -0.003895f, -0.005566f, -0.00899f, -0.0001261f, -0.00272f,
      -0.00529f, -0.005215f, -0.000558f, -0.006172f, 0.008354f, 0.000414f, -0.004574f, 0.00527f, 0.004333f,
      -0.00728f, -0.00797f, 0.0096f, -0.00344f, -0.00881f, -0.00368f, 0.00844f, -0.00517f, -0.005783f,
      -0.002708f, -0.006958f, 0.00088f, 0.007393f, 0.002115f, 0.00502f, -0.007347f, 0.002518f, -0.007164f,
      0.003891f, -0.006386f, 0.004723f, -0.007137f, 0.00979f, 0.00728f, -0.007385f, -0.003569f, -0.001245f,
      0.007244f, -0.004177f, 0.005627f, -0.001364f, 0.007786f, -0.003647f, 0.00975f, 0.003262f, 0.006668f,
      0.007492f, -0.002676f, 0.00452f, 0.00613f, 0.009895f, -0.0000653f, -0.0002944f, 0.0095f, 0.00829f,
      0.003607f, -0.00763f, 0.001573f, 0.00708f, 0.001338f, 0.00761f, -0.00934f, 0.00425f, 0.004677f,
      0.004356f, -0.00835f, -0.003391f, -0.00722f, 0.00877f, 0.001739f, -0.0078f, -0.003801f, -0.002934f,
      0.00592f, -0.00832f, -0.005596f, 0.00847f, 0.002663f, -0.002655f, -0.00461f, 0.001812f, -0.005447f,
      0.00393f, -0.0001626f, -0.0099f, -0.005177f, -0.000107f, -0.004513f, -0.00942f, -0.004494f, -0.0002584f,
      0.00558f, 0.00919f, 0.00483f, -0.003881f, 0.0000862f, 0.00472f, 0.002277f, 0.00452f, -0.005043f,
      -0.00812f, 0.006695f, -0.001397f, 0.00708f, 0.00666f, 0.009445f, 0.002443f, 0.00672f, 0.00742f,
      0.0047f, -0.0099f, -0.001733f, -0.001216f, 0.002306f, 0.00525f, 0.006687f, 0.007397f, -0.004185f,
      -0.007645f, 0.00497f, 0.002726f, -0.004883f, 0.00545f, 0.001207f, -0.003443f, 0.00855f, -0.008575f,
      -0.00995f, 0.00938f, 0.001395f, -0.005894f, -0.004715f, 0.001335f, 0.007214f, -0.00979f, -0.0009723f,
      -0.00884f, -0.00325f, -0.006447f, -0.0002873f, -0.006546f, -0.00914f, 0.00311f, 0.001508f, -0.008644f,
      0.003849f, -0.00224f, -0.0073f, 0.004158f, -0.007076f, -0.00458f, -0.002794f, 0.00691f, -0.00991f,
      -0.002531f, -0.007236f, 0.00291f, 0.003098f, -0.00666f, 0.00618f, -0.001502f, -0.008026f, 0.0001609f,
      -0.001733f, 0.00476f, 0.007725f, -0.007076f, -0.005398f, -0.001904f, 0.002743f, 0.001987f, 0.002935f,
      0.006363f, 0.007755f, -0.002127f, -0.002626f, 0.003273f, 0.0044f, -0.003975f, -0.00273f, -0.001413f,
      -0.008736f, -0.005775f, 0.00445f, -0.007412f, -0.00647f, 0.0046f, 0.007393f, -0.0003533f, -0.00926f,
      -0.006104f, 0.001658f, 0.00642f, -0.00962f, 0.00724f, -0.00032f, -0.00848f, -0.007442f, 0.001179f,
      -0.004684f, -0.001757f, 0.002796f, 0.00741f, 0.002192f, 0.003952f, 0.002794f, -0.00581f, 0.00923f,
      -0.000795f, -0.008545f, 0.0004318f, 0.007034f, 0.001034f, -0.009224f, 0.0037f, 0.00736f, -0.007587f,
      -0.001963f, 0.00037f, -0.001584f, -0.0001048f, 0.00979f, -0.007168f, 0.003159f, 0.00205f, -0.0082f,
      0.000802f, 0.00919f, 0.005257f, 0.000411f, 0.006824f, 0.00543f, -0.00202f, -0.008705f, -0.0084f,
      -0.0008135f, -0.001487f, -0.00698f, 0.00766f, 0.0003076f, 0.002989f, 0.00785f, 0.004498f, 0.004917f,
      0.001951f, 0.00489f, -0.000938f, 0.00438f, -0.00010777f, 0.00993f, -0.003304f, -0.00859f, 0.00656f,
      -0.009926f, 0.00572f, 0.009445f, 0.004425f, 0.00595f, 0.005547f, -0.00555f, 0.00912f, -0.00391f,
      0.00417f, -0.00732f, -0.00944f, -0.001693f, 0.003319f, 0.007904f, 0.004158f, 0.008026f, -0.004173f,
      0.00174f, 0.00794f, 0.001028f, -0.0004673f, -0.01f, 0.005222f, 0.00968f, 0.00173f, -0.00965f,
      0.00775f, 0.00758f, -0.006916f, -0.006714f, 0.001373f, -0.00906f, 0.005737f, -0.00403f, 0.003036f,
      -0.00832f, -0.001393f, -0.00903f, -0.007996f, -0.001152f, 0.00698f, -0.00907f, 0.00455f, 0.0006533f,
      -0.001487f, -0.0074f, 0.005177f, 0.00607f, 0.006973f, -0.002907f, -0.008446f, 0.004932f, 0.00457f,
      -0.001466f, 0.007805f, 0.002241f, -0.002304f, -0.006294f, -0.00625f, 0.002876f, 0.005146f, 0.000603f,
      0.00309f, -0.00912f, 0.002026f, 0.0096f, -0.000262f, 0.00007397f, 0.001089f, -0.00799f, 0.00948f,
      -0.0007935f, -0.00997f, 0.001588f, 0.009674f, -0.0006795f, 0.00958f, 0.00604f, -0.00975f, -0.001219f,
      -0.005093f, 0.00061f, 0.0002333f, 0.006195f, 0.006245f, 0.00548f, 0.006554f, 0.009155f, 0.003277f,
      -0.0027f, -0.002827f, 0.002981f, 0.00059f, -0.00643f, 0.001903f, 0.006195f, -0.001568f, -0.002792f,
      0.001151f, -0.00969f, 0.0001194f, 0.006084f, -0.00789f, -0.00746f, -0.000923f, -0.002726f, -0.0009117f,
      -0.009155f, 0.0003529f, 0.002682f, 0.00394f, 0.0003521f, -0.006798f, 0.f, 0.006145f, -0.006645f,
      -0.0000278f, 0.005737f, -0.003601f, 0.008156f, 0.006905f, 0.00996f, 0.00752f, 0.00513f, -0.001212f,
      -0.005558f, -0.009796f, -0.009056f, 0.001026f, -0.003756f, 0.002048f, 0.00501f, 0.004303f, 0.00885f,
      -0.002895f, 0.00885f, 0.00881f, 0.008965f, -0.00772f, 0.000675f, -0.00361f, 0.009254f, 0.00947f,
      0.002851f, -0.00443f, -0.0008383f, -0.001255f, 0.007088f, -0.00718f, -0.001156f, 0.00496f, 0.00543f,
      0.009575f, -0.00932f, -0.00289f, -0.00961f, -0.005413f, 0.00887f, 0.008194f, 0.007217f, 0.001349f,
      -0.00616f, -0.00132f, 0.008255f, 0.008354f, -0.001022f, -0.00916f, 0.0012f, 0.00942f, -0.005272f,
      -0.007713f, 0.00924f, -0.002554f, 0.00812f, 0.002947f, 0.006466f, 0.007267f, 0.004383f, 0.006683f,
      -0.004047f, -0.001562f, 0.0000953f, -0.00198f, -0.001826f, -0.005013f, 0.008095f, -0.00878f, -0.006645f,
      0.005096f, -0.0003052f, -0.00815f, -0.00986f, 0.0081f, 0.00661f, -0.00097f, 0.006916f, -0.007244f,
      0.004272f, 0.00444f, -0.00546f, 0.003994f, -0.00191f, 0.001437f, 0.00408f, -0.000537f, -0.007557f,
      0.0009537f, -0.00972f, -0.006805f, -0.007835f, -0.000847f, 0.005665f, -0.0085f, 0.005657f, 0.006027f,
      -0.009285f, 0.00652f, 0.005535f, 0.009224f, 0.0007205f, -0.00692f, -0.00881f, 0.005817f, -0.00506f,
      -0.00877f, -0.00991f, -0.001778f, -0.002598f, -0.00755f, 0.003616f, 0.00898f, 0.002537f, -0.001853f,
      -0.000725f, 0.00202f, 0.001978f, -0.0008383f, 0.004784f, -0.0003638f, -0.00895f, 0.00243f, -0.00395f,
      0.001955f, -0.002853f, -0.005f, 0.00976f, -0.0006366f, -0.002913f, 0.002592f, -0.00857f, -0.0006256f,
      -0.0001568f, 0.003605f, -0.001659f, 0.000935f, -0.005302f, 0.00593f, -0.002056f, 0.003723f, 0.005863f,
      0.0089f, -0.004147f, -0.007706f, 0.0006566f, 0.003222f, -0.00247f, 0.005234f, -0.009605f, -0.00279f,
      0.00031f, -0.008606f, -0.00952f, 0.001459f, 0.008446f, -0.00921f, 0.00895f, -0.00951f, -0.002565f,
      0.00084f, 0.006874f, -0.004066f, 0.004723f, -0.0006924f, 0.00932f, 0.003325f, -0.006763f, 0.004936f,
      -0.003439f, -0.000998f, -0.008224f, -0.00412f, 0.006996f, -0.007057f, -0.00992f, -0.004974f, 0.001361f,
      -0.009865f, 0.00982f, -0.00375f, -0.002928f, 0.00166f, -0.007782f, -0.001827f, 0.000567f, -0.002838f,
      -0.0002354f, -0.00259f, -0.00849f, -0.001284f, 0.002161f, 0.001709f, -0.00802f, 0.00199f, -0.003202f,
      0.002773f, 0.009056f, -0.001113f, -0.006054f, -0.00209f, 0.007355f, 0.008194f, -0.005627f, -0.005226f,
      -0.00478f, 0.001043f, 0.002869f, -0.00657f, -0.003176f, -0.004704f, -0.004574f, -0.00434f, 0.007328f,
      0.00895f, -0.00853f, -0.006207f, -0.00928f, -0.009476f, 0.009125f, 0.004627f, -0.004642f, -0.004658f,
      0.00919f, 0.003496f, 0.002165f, 0.00413f, -0.007694f, 0.003744f, 0.001043f, 0.002182f, -0.00698f,
      -0.003906f, 0.00365f, 0.003763f, -0.0043f, 0.002554f, 0.0094f, 0.00586f, -0.00655f, -0.00171f,
      -0.0009985f, -0.00851f, 0.00584f, 0.004883f, -0.007523f, 0.005016f, 0.003046f, 0.005917f, -0.006622f,
      0.00741f, -0.002499f, 0.0004418f, -0.003113f, 0.0003803f, 0.003252f, -0.00917f, 0.00506f, -0.006687f,
      -0.00916f, 0.000701f, 0.00945f, -0.002863f, 0.00827f, 0.00938f, 0.003405f, -0.00935f, -0.00912f,
      0.00259f, 0.001822f, -0.00674f, 0.0008016f, -0.001132f, 0.00899f, 0.001555f, -0.0007024f, 0.00899f,
      -0.00938f, -0.00109f, -0.00674f, 0.001553f, 0.00696f, 0.009415f, 0.0005765f, -0.0002084f, 0.004097f,
      0.005985f, 0.001656f, 0.005325f, -0.00839f, 0.003904f, 0.00822f, -0.003994f, 0.00635f, -0.000794f,
      -0.00667f, 0.002296f, -0.002838f, -0.00975f, -0.001081f, 0.005127f, 0.001922f, 0.005127f, -0.008156f,
      -0.006653f, 0.00935f, -0.00302f, -0.00052f, -0.005894f, -0.009674f, -0.00613f, 0.009705f, -0.006924f,
      0.004726f, 0.004784f, -0.00146f, 0.001746f, -0.002958f, 0.009636f, 0.005665f, -0.000724f, 0.004875f,
      -0.001856f, -0.002975f, 0.0071f, 0.002045f, 0.00507f, -0.007042f, -0.006958f, 0.002089f, -0.003504f,
      0.0004888f, -0.005943f, 0.007607f, -0.003822f, -0.004692f, 0.0001357f, 0.004456f, -0.00799f, -0.006413f,
      0.002268f, 0.00888f, -0.00872f, -0.004936f, -0.0091f, -0.00353f, -0.0052f, -0.003223f, -0.00825f,
      0.003952f, -0.002771f, 0.006344f, 0.00862f, 0.00904f, -0.00221f, -0.0001844f, -0.00227f, 0.000672f,
      -0.004852f, -0.005795f, -0.002771f, -0.00653f, -0.002579f, 0.006954f, 0.002605f, -0.00804f, 0.00432f,
      0.0000249f, -0.004536f, -0.008514f, 0.00618f, -0.002804f, 0.00895f, -0.009094f, -0.009155f, -0.003836f,
      -0.0008125f, 0.007385f, 0.00554f, -0.0004065f, -0.00517f, -0.006493f, -0.007027f, 0.003748f, -0.00834f,
      -0.006668f, 0.00982f, -0.001279f, -0.0008125f, 0.000629f, 0.003786f, -0.00859f, -0.000755f, 0.0004015f,
      -0.003065f, -0.007042f, -0.00967f, 0.0004108f, 0.00947f, -0.007076f, -0.0006723f, 0.006496f, -0.001414f,
      0.008194f, -0.000413f, 0.008125f, 0.00146f, -0.006462f, 0.002676f, -0.005474f, -0.003166f, 0.006027f,
      0.001129f, 0.001874f, 0.001855f, 0.00766f, -0.006634f, -0.000823f, -0.00303f, 0.005795f, 0.00279f,
      0.002512f, 0.006172f, 0.006474f, 0.000632f, -0.007507f, 0.001753f, -0.002531f, 0.002895f, -0.007034f,
      0.004955f, -0.0096f, 0.007793f, -0.00803f, -0.0095f, 0.006615f, -0.00854f, 0.00214f, 0.00532f,
      -0.00995f, 0.00772f, 0.006977f, -0.00873f, -0.00617f, -0.00808f, -0.00479f, -0.00397f, 0.00456f,
      0.003944f, 0.0001737f, 0.001538f, -0.005756f, 0.009964f, 0.002096f, -0.00984f, 0.001642f, 0.003113f,
      -0.00802f, -0.003527f, -0.00876f, 0.003502f, -0.00562f, 0.003378f, 0.006676f, 0.000644f, 0.002071f,
      -0.00587f, -0.00771f, -0.0009327f, -0.00441f, 0.007095f, 0.005478f, 0.00781f, 0.00952f, 0.006176f,
      0.0003223f, 0.00818f, 0.00678f, -0.004147f, -0.00999f, 0.00903f, -0.00987f, 0.007553f, -0.00438f,
      0.005028f, 0.0003302f, 0.0002394f, -0.005104f, -0.002537f, -0.005333f, 0.004635f, -0.005787f, -0.005177f,
      -0.005615f, -0.00463f, 0.0001181f, -0.00814f, 0.00656f, -0.00132f, 0.003115f, -0.006237f, -0.00123f,
      -0.008804f, -0.002682f, -0.00877f, 0.00749f, -0.00863f, 0.004997f, 0.007736f, -0.00963f, -0.002966f,
      -0.00405f, -0.004005f, 0.006763f, -0.00639f, 0.000797f, 0.002903f, 0.00967f, -0.0009356f, -0.00675f,
      0.00917f, -0.0048f, 0.0088f, 0.007168f, 0.00394f, 0.005524f, 0.0002052f, -0.0004148f, 0.0059f,
      -0.002966f, 0.008f, -0.00955f, -0.008484f, 0.00856f, 0.003498f, -0.005703f, 0.004974f, 0.0089f,
      -0.004208f, -0.005203f, -0.007496f, 0.003206f, -0.007713f, -0.0068f, 0.00437f, 0.008896f, 0.0007954f,
      0.002823f, -0.002413f, -0.004665f, 0.0007997f, -0.005394f, 0.00806f, -0.001563f, -0.001497f, -0.005314f,
      -0.00952f, 0.0093f, 0.005066f, 0.00407f, 0.004482f, -0.00788f, 0.001537f, 0.00806f, -0.005013f,
      -0.003735f, 0.00956f, -0.00946f, 0.002008f, -0.006847f, 0.003038f, 0.003141f, -0.005787f, 0.005665f,
      0.002735f, -0.002401f, 0.003057f, 0.000753f, 0.004444f, 0.00805f, 0.001004f, -0.0065f, -0.001637f,
      0.0065f, 0.004467f, -0.00896f, -0.006573f, -0.007236f, 0.007435f, -0.00392f, -0.001908f, -0.008736f,
      -0.0007854f, 0.000625f, 0.003866f, -0.002039f, -0.002193f, -0.006447f, -0.00793f, -0.002161f, -0.0073f,
      0.00472f, 0.001314f, 0.006416f, -0.009224f, 0.00668f, 0.008865f, 0.009155f, -0.004684f, 0.00807f,
      -0.0008855f, 0.002748f, 0.001529f, -0.004765f, -0.001041f, 0.00859f, 0.005573f, 0.00433f, -0.009155f,
      -0.007614f, 0.00472f, -0.0009365f, 0.00003576f, 0.002872f, -0.003223f, 0.003098f, -0.001782f, 0.001795f,
      0.006645f, 0.002974f, -0.0094f, 0.005337f, 0.00877f, -0.00649f, 0.00959f, -0.008156f, -0.0008917f,
      0.006607f, 0.00905f, -0.001238f, -0.001246f, -0.002775f, -0.002815f, 0.00451f, -0.004486f, 0.003998f,
      0.00956f, -0.00981f, 0.005096f, -0.00876f, -0.002571f, 0.002287f, -0.002996f, -0.008896f, -0.006973f,
      0.003885f, 0.001993f, -0.006523f, 0.0048f, -0.005745f, 0.004883f, 0.005627f, -0.00919f, 0.00978f,
      -0.000961f, 0.00954f, 0.003023f, 0.006172f, -0.00371f, -0.00509f, -0.00392f, -0.00989f, 0.00212f,
      -0.00917f, -0.009865f, 0.00965f, 0.003618f, -0.004303f, 0.00628f, 0.002913f, 0.0086f, -0.00881f,
      0.004963f, -0.006886f, -0.00000197f, -0.008736f, 0.004147f, -0.003227f, -0.001696f, -0.003815f, 0.00957f,
      -0.00994f, -0.006596f, 0.00925f, 0.007454f, -0.001091f, -0.0004747f, 0.009026f, 0.00854f, 0.00133f,
      -0.00263f, 0.00543f, 0.003836f, 0.004856f, -0.006695f, 0.005478f, -0.008415f, 0.003187f, -0.00998f,
      0.009514f, 0.002903f, -0.005165f, -0.0004752f, -0.009f, -0.008965f, 0.005806f, 0.006153f, 0.00893f,
      -0.00877f, -0.006866f, 0.004154f, -0.008125f, 0.007202f, -0.005573f, 0.004f, -0.002998f, 0.002878f,
      0.005672f, 0.00607f, -0.004578f, 0.001471f, -0.002363f, -0.00006247f, 0.0007734f, 0.001287f, -0.0006113f,
      0.003868f, -0.00696f, -0.003672f, 0.00688f, -0.00908f, -0.00665f, 0.003775f, -0.006355f, -0.005634f,
      0.00421f, 0.00937f, -0.004856f, 0.002947f, -0.003933f, -0.0086f, 0.00988f, 0.00546f, -0.0008826f,
      0.00433f, 0.007183f, 0.002195f, -0.005333f, 0.006683f, 0.003277f, 0.001082f, 0.00579f, -0.00623f,
      -0.00966f, -0.002708f, 0.00627f, 0.00581f, -0.0095f, 0.008896f, -0.002478f, -0.00966f, 0.007526f,
      -0.001696f, 0.002949f, 0.001381f, -0.00684f, -0.005974f, 0.00413f, 0.00085f, 0.004032f, 0.004807f,
      0.0004041f, -0.006992f, 0.003105f, -0.0002321f, 0.00867f, 0.00237f, 0.00464f, -0.00887f, -0.005978f,
      -0.005844f, -0.00826f, 0.005035f, 0.00953f, 0.006485f, -0.00415f, -0.00873f, -0.006836f, 0.00572f,
      0.001606f, -0.00828f, -0.001708f, -0.006145f, 0.00914f, -0.00965f, 0.005646f, -0.00857f, 0.006638f,
      0.00327f, 0.00424f, 0.001341f, 0.003788f, -0.000685f, 0.0061f, -0.00782f, 0.003334f, -0.0068f,
      0.001557f, 0.005825f, -0.0058f, -0.000689f, 0.007496f, 0.00708f, -0.006107f, 0.007668f, -0.001199f,
      -0.00948f, 0.00668f, -0.003176f, 0.003733f, -0.001616f, 0.006714f, 0.00789f, 0.001432f, 0.004112f,
      0.00384f, 0.009636f, 0.007053f, -0.00374f, 0.00495f, 0.00959f, 0.004135f, 0.00721f, 0.007225f,
      -0.0008454f, 0.008286f, 0.0000413f, 0.003618f, 0.004047f, 0.00454f, -0.0079f, 0.00869f, 0.00706f,
      -0.007492f, 0.00493f, 0.00689f, -0.0005245f, 0.00604f, 0.00357f, 0.00598f, -0.00959f, -0.003292f,
      0.0008936f, 0.00904f, 0.002445f, 0.00894f, 0.00819f, 0.003876f, 0.002953f, 0.003384f, -0.006687f,
      0.002918f, -0.0056f, -0.0003066f, -0.001384f, 0.007675f, 0.0009513f, -0.007656f, 0.00804f, -0.000968f,
      -0.000649f, 0.00913f, -0.0041f, 0.0002625f, -0.0001359f, -0.008865f, 0.002167f, 0.00687f, -0.00606f,
      0.0003486f, 0.0003984f, -0.004803f, 0.006454f, -0.004997f, 0.00892f, -0.007423f, -0.001277f, -0.007504f,
      0.00762f, 0.003056f, 0.001508f, -0.00391f, 0.00859f, -0.00768f, -0.003675f, 0.002884f, 0.006508f,
      0.000506f, 0.002567f, 0.007607f, -0.003233f, 0.0073f, 0.003862f, -0.003817f, 0.00735f, 0.002506f,
      -0.00823f, -0.006706f, 0.005676f, -0.00931f, -0.004025f, 0.006542f, 0.000566f, 0.00919f, -0.002083f,
      -0.00783f, 0.0013485f, -0.00839f, 0.0089f, -0.0066f, 0.009674f, -0.00821f, 0.0061f, -0.002129f,
      0.00598f, 0.008865f, 0.00513f, -0.00582f, -0.00459f, -0.00962f, -0.00962f, -0.005966f, -0.007187f,
      0.00995f, 0.004295f, 0.004467f, 0.001008f, -0.00809f, 0.00922f, -0.00768f, -0.00994f, -0.005596f,
      0.006706f, 0.00748f, 0.00942f, -0.00396f, 0.001708f, -0.00961f, 0.005653f, 0.00976f, -0.001643f,
      0.003786f, -0.002264f, 0.002747f, -0.0003808f, 0.000354f, 0.001055f, 0.00584f, 0.006306f, 0.005363f,
      -0.006443f, -0.0005603f, 0.00871f, 0.00683f, -0.002083f, -0.00611f, -0.006573f, -0.0027f, 0.004917f,
      0.006207f, 0.004932f, -0.00669f, 0.005665f, 0.002796f, 0.00901f, -0.000798f, 0.001478f, 0.003788f,
      0.000707f, 0.00934f, 0.005985f, -0.00145f, -0.0008683f, 0.00339f, 0.002144f, 0.006596f, 0.00984f,
      0.00258f, 0.0048f, 0.0003848f, -0.002644f, -0.002129f, -0.001171f, -0.002369f, -0.007328f, 0.00841f,
      -0.005325f, 0.00968f, -0.00982f, -0.003754f, -0.0006895f, 0.00784f, 0.003864f, 0.008316f, -0.003483f,
      0.004986f, -0.003044f, -0.005714f, -0.001846f, -0.001568f, 0.0003648f, 0.00724f, 0.006336f, -0.003222f,
      -0.006836f, 0.001214f, -0.003124f, -0.0006356f, -0.001073f, 0.002682f, -0.007538f, -0.001701f, -0.00883f,
      0.00986f, 0.006336f, 0.0011f, -0.00879f, -0.005875f, 0.004025f, 0.00613f, 0.004856f, -0.008896f,
      0.0006967f, 0.0064f, 0.002707f, -0.002317f, -0.002214f, 0.002409f, -0.000346f, -0.006924f, 0.001986f,
      -0.003166f, 0.00836f, -0.00899f, 0.0034f, -0.007755f, 0.00407f, 0.00807f, 0.0076f, 0.003824f,
      0.003876f, -0.00853f, -0.00649f, -0.003506f, 0.001777f, -0.009705f, -0.00516f, -0.0094f, 0.00939f,
      -0.00786f, -0.00911f, -0.000737f, 0.000864f, -0.00851f, 0.00786f, -0.003422f, -0.00832f, -0.0007277f,
      0.005642f, -0.00868f, -0.002851f, 0.0005975f, -0.007347f, -0.001616f, -0.001303f, 0.00717f, -0.00231f,
      -0.008354f, -0.005333f, 0.00864f, 0.006123f, -0.00994f, 0.00313f, -0.00676f, -0.005806f, 0.008446f,
      -0.0007553f, -0.006416f, 0.00223f, -0.00579f, 0.00576f, -0.00892f, 0.002424f, -0.00486f, 0.00636f,
      0.003344f, -0.003195f, 0.001562f, 0.00318f, -0.007202f, -0.001358f, -0.0001854f, 0.002499f, 0.001725f,
      0.000389f, -0.006737f, 0.002745f, 0.000575f, -0.003534f, 0.004284f, 0.0019045f, 0.004898f, -0.004356f,
      0.002254f, -0.00577f, 0.0018215f, -0.008736f, 0.00769f, -0.00885f, -0.00859f, -0.00441f, 0.00583f,
      -0.009285f, -0.00792f, -0.00922f, -0.003815f, -0.00886f, -0.005394f, -0.00663f, -0.008224f, -0.00353f,
      0.002161f, 0.00301f, -0.00542f, -0.0085f, -0.007446f, -0.00846f, -0.00515f, 0.00204f, 0.00543f,
      -0.001219f, -0.007072f, 0.001966f, -0.00894f, 0.0008793f, -0.003418f, 0.00393f, -0.005283f, 0.005756f,
      0.003225f, 0.002123f, 0.002283f, 0.00566f, 0.000477f, 0.00497f, 0.005295f, 0.002136f, 0.00692f,
      0.00872f, 0.00936f, -0.005074f, 0.00645f, -0.001117f, 0.006493f, -0.00574f, 0.001013f, 0.003334f,
      -0.005703f, -0.006992f, -0.004314f, 0.005314f, 0.001457f, -0.00594f, -0.003252f, 0.00844f, 0.002502f,
      0.002604f, 0.00289f, 0.00221f, -0.003344f, -0.006905f, -0.00799f, 0.007378f, -0.00945f, 0.006023f,
      -0.00791f, 0.001273f, 0.003849f, 0.007694f, 0.005424f, 0.00298f, -0.003618f, -0.0001827f, 0.002077f,
      0.001976f, -0.006474f, 0.00079f, 0.00982f, 0.004166f, 0.007027f, 0.008606f, 0.00818f, 0.00697f,
      -0.003006f, 0.0045f, -0.00885f, -0.00515f, 0.00723f, -0.0001746f, -0.00727f, 0.006237f, -0.008385f,
      0.008194f, -0.008316f, -0.002525f, 0.002558f, 0.00639f, 0.003586f, -0.00612f, -0.006756f, -0.008354f,
      0.004883f, -0.00506f, -0.009f, -0.00537f, -0.001243f, -0.005596f, -0.00853f, -0.007545f, 0.00786f,
      0.001839f, -0.002245f, 0.00544f, -0.00196f, 0.004967f, -0.003464f, -0.005108f, 0.003086f, 0.002628f,
      -0.002502f, -0.00665f, -0.006226f, 0.0079f, -0.002287f, 0.0003567f, -0.001279f, 0.004826f, 0.005432f,
      -0.00634f, -0.003204f, 0.0002022f, -0.00198f, -0.0008726f, 0.004055f, 0.00793f, -0.00427f, -0.00533f,
      0.00734f, -0.00799f, -0.0051f, -0.009995f, 0.0051f, 0.00413f, -0.00679f, 0.00262f, 0.001331f,
      0.001461f, -0.00865f, -0.00791f, -0.003975f, 0.002504f, 0.0002255f, 0.002337f, -0.00456f, -0.005974f,
      0.000257f, -0.00545f, 0.00842f, 0.005585f, -0.0003774f, 0.0008087f, -0.001679f, 0.003853f, 0.00991f,
      0.0031f, 0.00523f, -0.00721f, 0.000989f, -0.005642f, -0.001042f, 0.007935f, -0.006195f, 0.001426f,
      0.00414f, 0.00925f, -0.00419f, 0.004852f, -0.00639f, 0.00694f, -0.007706f, -0.00684f, -0.00602f,
      -0.004444f, 0.005016f, -0.00803f, -0.00955f, 0.004097f, -0.003754f, 0.002384f, -0.007515f, 0.003508f,
      -0.00749f, 0.00519f, 0.00228f, 0.007015f, -0.007572f, -0.003864f, -0.00843f, 0.00543f, 0.00911f,
      0.00774f, 0.009125f, -0.003473f, -0.00646f, 0.00856f, 0.004272f, 0.00534f, 0.003859f, -0.0001141f,
      0.001515f, 0.003437f, 0.00737f, 0.003565f, -0.002705f, 0.003675f, 0.003023f, -0.0002156f, -0.00894f,
      0.00103f, -0.001797f, -0.00854f, 0.001505f, -0.00876f, -0.003614f, 0.004887f, -0.005085f, 0.002449f,
      0.00524f, -0.00589f, 0.00784f, 0.001411f, -0.0095f, 0.007797f, -0.003391f, 0.008316f, 0.0094f,
      0.00917f, -0.00658f, -0.00685f, -0.005085f, -0.005375f, 0.008705f, -0.004093f, 0.00764f, -0.006172f,
      -0.00609f, -0.0005703f, -0.00941f, -0.007065f, 0.00942f, 0.00403f, 0.00392f, -0.0000164f, 0.000577f,
      0.001058f, 0.006317f, 0.0008893f, 0.001935f, -0.009865f, -0.00542f, 0.001452f, 0.00916f, -0.00852f,
      -0.00081f, 0.00397f, 0.0069f, 0.003246f, -0.004456f, 0.00777f, -0.004444f, 0.003632f, -0.002512f,
      -0.00284f, 0.009926f, 0.00869f, -0.00636f, -0.006454f, 0.006805f, -0.00232f, -0.00924f, 0.006268f,
      0.00501f, -0.00951f, -0.00518f, 0.006126f, 0.00966f, 0.00881f, -0.009346f, 0.00912f, 0.00341f,
      0.00617f, 0.00984f, -0.00357f, 0.00596f, -0.0081f, -0.0006824f, -0.00711f, 0.004803f, 0.00484f,
      -0.000756f, 0.002865f, -0.00422f, 0.00005835f, 0.00912f, 0.000726f, 0.001402f, 0.00644f, -0.006542f,
      0.006016f, 0.003975f, 0.00556f, 0.0000735f, -0.002203f, 0.003893f, -0.000724f, 0.005882f, -0.006226f,
      -0.006912f, 0.003027f, 0.0004182f, -0.00728f, -0.00726f, -0.00896f, 0.008095f, -0.001346f, 0.00898f,
      0.002956f, -0.003334f, -0.007717f, -0.00876f, 0.00037f, -0.00727f, -0.003258f, 0.009476f, 0.009056f,
      0.00598f, 0.00281f, 0.00586f, -0.00981f, -0.003296f, 0.00769f, -0.000486f, 0.0091f, 0.00634f,
      -0.00542f, 0.00512f, -0.002474f, -0.009514f, 0.00402f, -0.004787f, 0.00274f, -0.001112f, -0.002436f,
      0.00949f, -0.000839f, -0.009155f, 0.002499f, 0.001512f, 0.001406f, -0.00313f, -0.002022f, -0.008896f,
      -0.00528f, -0.009254f, -0.002148f, -0.000707f, -0.0001829f, -0.001159f, 0.00411f, -0.007637f, -0.00364f,
      0.005135f, -0.00928f, -0.0000797f, 0.004642f, -0.00817f, -0.007072f, -0.003914f, 0.00416f, 0.002985f,
      -0.0075f, -0.000736f, 0.008934f, 0.004204f, 0.0004723f, 0.006306f, -0.007675f, -0.007835f, 0.0005293f,
      -0.002478f, -0.006336f, 0.007996f, 0.002539f, 0.001836f, 0.00968f, 0.006844f, 0.001179f, 0.001448f,
      0.006042f, 0.00292f, -0.007122f, -0.001914f, 0.004448f, 0.00822f, 0.00672f, 0.000714f, -0.001145f,
      0.009415f, 0.0015335f, -0.005585f, -0.006104f, -0.0003273f, -0.00987f, 0.001559f, -0.00608f, 0.007664f,
      0.00834f, -0.0002584f, -0.004097f, 0.00745f, 0.005417f, -0.002129f, 0.001597f, 0.00749f, -0.001676f,
      0.006344f, 0.006905f, 0.004364f, -0.00739f, -0.001457f, 0.00806f, -0.008f, -0.004284f, -0.00717f,
      0.00547f, 0.004463f, 0.00529f, -0.00843f, 0.008064f, 0.00556f, 0.0005236f, 0.00918f, -0.004986f,
      0.00578f, -0.001013f, -0.003479f, -0.004425f, -0.0076f, -0.004093f, 0.003084f, -0.00531f, -0.00902f,
      -0.002844f, 0.004982f, -0.00986f, 0.003986f, 0.002125f, 0.004036f, -0.006798f, 0.000773f, 0.000544f,
      -0.0001241f, 0.009155f, 0.002682f, -0.00997f, -0.00826f, 0.003769f, 0.001383f, -0.005318f, 0.004673f,
      -0.005314f, 0.00691f, 0.00212f, -0.00656f, -0.006226f, -0.008705f, 0.00459f, -0.003798f, 0.00869f,
      -0.002985f, -0.000604f, 0.00826f, -0.00541f, -0.00502f, 0.000809f, -0.00969f, -0.006626f, 0.005123f,
      -0.0005465f, -0.00858f, 0.005554f, -0.002083f, 0.007343f, -0.001588f, -0.001642f, 0.0007577f, 0.00318f,
      -0.00391f, 0.00404f, 0.00886f, -0.006374f, -0.00958f, -0.005077f, -0.00218f, 0.00745f, 0.00944f,
      0.007233f, 0.003042f, -0.003296f, 0.006786f, -0.006706f, 0.007114f, 0.00566f, 0.005325f, 0.007637f,
      -0.00661f, 0.0008025f, -0.002693f, 0.005634f, 0.001557f, -0.007133f, -0.00483f, -0.00654f, 0.006313f,
      -0.00926f, -0.00372f, -0.00583f, -0.004025f, 0.00761f, 0.00955f, 0.002691f, -0.00915f, -0.006084f,
      -0.008835f, 0.003885f, 0.009514f, -0.00841f, 0.003637f, -0.00765f, -0.005978f, 0.001959f, -0.005295f,
      -0.001565f, -0.003551f, -0.000824f, 0.005848f, -0.00010514f, 0.00828f, -0.003895f, -0.003197f, 0.00797f,
      0.00998f, 0.004635f, 0.006504f, 0.007023f, -0.00675f, 0.001584f, 0.004642f, 0.007458f, -0.002005f,
      0.0000653f, 0.00715f, 0.00402f, 0.00782f, -0.00331f, 0.00676f, 0.000039f, 0.00644f, -0.0007744f,
      0.005688f, 0.00511f, -0.005135f, 0.000995f, 0.006756f, -0.002304f, 0.003553f, -0.00938f, -0.000616f,
      -0.00897f, -0.00685f, -0.00838f, 0.003983f, -0.004807f, 0.002314f, 0.00847f, 0.00846f, -0.007507f,
      0.002136f, 0.005905f, -0.00899f, 0.0081f, 0.008f, 0.00889f, -0.00907f, -0.00489f, 0.00938f,
      -0.009254f, 0.00627f, 0.0052f, -0.002031f, -0.0006337f, -0.001191f, 0.001453f, -0.003918f, 0.001798f,
      -0.00491f, -0.002062f, -0.00889f, 0.00309f, 0.007526f, 0.0007014f, -0.001351f, -0.003838f, 0.00458f,
      0.004005f, -0.00923f, 0.00581f, -0.002983f, -0.00901f, 0.007095f, 0.00844f, -0.00989f, 0.001532f,
      -0.00867f, 0.001821f, -0.005646f, 0.00698f, -0.001757f, -0.00102f, -0.00511f, -0.007774f, 0.002588f,
      -0.006096f, 0.005196f, -0.002117f, -0.0003762f, 0.00738f, 0.001219f, 0.00447f, 0.00867f, -0.00494f,
      0.007313f, -0.008095f, 0.000967f, 0.004776f, 0.00296f, 0.001068f, 0.00818f, 0.00749f, -0.00939f,
      -0.00738f, -0.006214f, -0.00685f, 0.00569f, 0.00716f, 0.004375f, -0.00512f, -0.006252f, -0.004684f,
      -0.002974f, -0.007965f, 0.0025f, -0.00943f, 0.00539f, 0.0003204f, 0.0005164f, -0.006573f, 0.00646f,
      0.00502f, 0.007965f, -0.002003f, -0.00609f, -0.009285f, -0.005028f, -0.00985f, 0.001395f, 0.00415f,
      0.003494f, 0.00957f, 0.009834f, -0.005905f, 0.002436f, 0.001002f, -0.002335f, -0.00981f, 0.006714f,
      0.005135f, -0.003138f, -0.00786f, 0.005497f, 0.003677f, 0.00479f, -0.00453f, 0.00845f, 0.007454f,
      0.000992f, -0.00647f, 0.001218f, -0.004295f, 0.00004745f, 0.005558f, -0.002914f, 0.00861f, -0.008064f,
      0.003328f, -0.003998f, -0.007595f, 0.00487f, 0.0008106f, 0.005287f, -0.003735f, 0.003054f, 0.006645f,
      -0.002422f, 0.00974f, -0.001171f, 0.006264f, 0.00908f, 0.002903f, 0.00446f, 0.002419f, 0.00806f,
      -0.002483f, 0.0089f, 0.0004303f, -0.001789f, -0.00638f, -0.005802f, -0.00953f, -0.00526f, 0.006203f,
      -0.001033f, -0.00721f, 0.00391f, 0.00923f, 0.006676f, 0.00495f, -0.002512f, -0.000916f, 0.005054f,
      -0.007652f, 0.004738f, 0.00826f, -0.00989f, -0.00202f, -0.00824f, -0.004333f, 0.002779f, -0.00531f,
      0.00181f, -0.00475f, 0.005234f, -0.00558f, 0.002342f, -0.001395f, -0.005856f, 0.004074f, -0.00638f,
      -0.003561f, 0.00819f, 0.006454f, -0.00402f, -0.008766f, -0.006668f, -0.00244f, -0.00392f, -0.007248f,
      -0.00666f, 0.001226f, -0.0071f, 0.00746f, 0.00396f, -0.00057f, 0.0001602f, 0.006924f, -0.0004158f,
      -0.000988f, -0.008385f, 0.004936f, -0.001231f, 0.00533f, 0.00905f, 0.0015335f, 0.003677f, 0.00751f,
      -0.00807f, -0.0051f, 0.00774f, -0.000592f, 0.003368f, -0.001825f, -0.003403f, 0.008194f, -0.0004606f,
      0.00312f, -0.004196f, 0.008026f, 0.004883f, -0.003073f, -0.006607f, 0.00847f, -0.007446f, -0.00982f,
      -0.002375f, 0.009186f, 0.00991f, 0.005642f, -0.00632f, -0.005085f, 0.0084f, 0.002087f, 0.004f,
      0.002495f, 0.004326f, 0.00969f, -0.003504f, 0.008514f, 0.000959f, 0.003632f, -0.001369f, 0.005737f,
      0.002361f, -0.00802f, -0.006603f, 0.007866f, -0.008675f, 0.009384f, 0.001016f, 0.006927f, -0.005165f,
      0.001802f, -0.002798f, 0.008415f, 0.00439f, 0.003819f, 0.002295f, 0.006844f, -0.006813f, 0.0003488f,
      0.000659f, 0.00963f, -0.00946f, 0.002861f, -0.00614f, 0.002499f, -0.00706f, 0.003216f, -0.003124f,
      -0.004585f, 0.001135f, -0.00212f, 0.007435f, -0.003775f, -0.0001405f, -0.000892f, 0.006218f, -0.005333f,
      0.007397f, 0.003202f, 0.009026f, 0.003717f, 0.00787f, 0.005188f, 0.0002823f, -0.0052f, 0.00797f,
      -0.0009027f, -0.006462f, 0.00908f, -0.001527f, 0.005005f, 0.005547f, 0.00665f, -0.002155f, -0.00641f,
      0.00467f, -0.002872f, 0.000676f, 0.0009217f, 0.00424f, -0.000898f, 0.00932f, 0.004444f, -0.009834f,
      0.00908f, -0.0000113f, -0.00378f, 0.00792f, -0.00931f, -0.002563f, 0.003622f, 0.00972f, -0.0066f,
      -0.002348f, -0.00787f, 0.004368f, -0.00385f, 0.0099f, 0.00617f, -0.001304f, 0.008575f, -0.00803f,
      -0.008354f, 0.00794f, -0.00924f, 0.0069f, -0.00811f, 0.000215f, -0.00519f, -0.001069f, 0.000882f,
      -0.007378f, 0.006447f, -0.003225f, -0.00484f, -0.00356f, -0.0004394f, -0.002144f, -0.001932f, 0.0007205f,
      -0.00976f, 0.008514f, -0.006294f, 0.00618f, -0.001758f, -0.00713f, -0.00912f, 0.004726f, 0.00334f,
      0.00847f, -0.0001967f, 0.005165f, -0.004944f, -0.00915f, 0.0062f, -0.00553f, 0.0084f, -0.0054f,
      0.002823f, 0.00272f, -0.00271f, -0.009514f, 0.00629f, -0.006054f, 0.008865f, -0.00813f, -0.0076f,
      0.00857f, -0.003681f, -0.00738f, -0.00872f, -0.001488f, 0.00926f, -0.001791f, 0.00471f, -0.00482f,
      0.007812f, -0.004654f, -0.006138f, 0.003813f, 0.005768f, -0.00375f, -0.00992f, -0.000584f, 0.00783f,
      -0.004147f, 0.001611f, 0.001342f, -0.006832f, -0.00138f, 0.005325f, -0.0000265f, 0.009445f, 0.00872f,
      0.001329f, -0.0026f, 0.002577f, 0.0072f, 0.00547f, 0.006428f, -0.004864f, 0.00876f, -0.00906f,
      0.007317f, -0.007233f, -0.00774f, 0.003387f, -0.002037f, 0.00125f, 0.00655f, -0.003298f, 0.008514f,
      -0.003757f, 0.007935f, -0.003181f, 0.00629f, 0.00838f, 0.0009594f, 0.006897f, -0.008835f, 0.00446f,
      -0.0082f, -0.006042f, 0.00761f, -0.00883f, 0.002434f, 0.001002f, 0.00712f, -0.005688f, 0.003359f,
      -0.00606f, 0.002512f, 0.00576f, 0.006126f, 0.0009394f, -0.00787f, -0.00485f, 0.005936f, 0.002037f,
      -0.0024f, -0.00618f, -0.00157f, 0.00702f, -0.007637f, 0.0077f, -0.00784f, -0.0062f, -0.00975f,
      -0.00849f, 0.00843f, 0.003843f, -0.006443f, 0.004993f, -0.0001615f, 0.00902f, 0.00811f, 0.005333f,
      0.002123f, 0.001081f, 0.0086f, -0.003103f, 0.005783f, 0.004936f, -0.00898f, 0.001179f, 0.0007f,
      0.003462f, -0.00855f, 0.00254f, -0.0000039f, -0.00468f, 0.0012455f, 0.003431f, 0.007538f, 0.0082f,
      0.00843f, -0.001547f, 0.006157f, 0.001941f, -0.0013895f, -0.003096f, -0.003883f, -0.006382f, -0.00475f,
      0.008766f, -0.003225f, 0.0008793f, -0.002806f, -0.00432f, 0.003944f, 0.008286f, 0.003141f, -0.00975f,
      -0.00439f, -0.007645f, 0.0093f, 0.005238f, -0.002018f, -0.006023f, -0.001462f, 0.00286f, 0.00525f,
      0.005463f, -0.0005217f, -0.0003283f, -0.003103f, -0.007656f, -0.003311f, -0.0002983f, 0.005165f, 0.007187f,
      0.00674f, -0.002645f, 0.00882f, 0.009995f, -0.003174f, -0.002956f, -0.00978f, 0.00841f, 0.005043f,
      0.00798f, 0.00003827f, -0.004494f, -0.00883f, -0.0003128f, -0.0015955f, 0.00958f, 0.001948f, -0.007664f,
      -0.002064f, 0.002949f, 0.008736f, 0.00684f, 0.00804f, 0.004642f, -0.000742f, 0.001874f, -0.004864f,
      0.0003529f, -0.001284f, 0.00896f, -0.006954f, -0.003616f, 0.0078f, 0.00815f, -0.00876f, -0.002783f,
      -0.00649f, 0.00976f, 0.009125f, 0.0019f, -0.0004215f, 0.00461f, 0.001037f, 0.009384f, 0.003422f,
      0.001194f, 0.00923f, 0.00554f, -0.00855f, -0.001592f, -0.002981f, 0.006016f, 0.002314f, -0.00483f,
      0.002476f, -0.00894f, -0.000574f, 0.0096f, -0.0002362f, -0.002018f, 0.00283f, 0.00251f, -0.0001559f,
      -0.00557f, 0.00661f, -0.002537f, 0.005524f, 0.00961f, -0.002073f, 0.00454f, -0.006428f, 0.001997f,
      -0.00446f, -0.0007524f, 0.002176f, -0.00209f, -0.00874f, 0.001289f, 0.00523f, 0.001575f, -0.008736f,
      0.007057f, -0.0069f, -0.00512f, -0.005383f, 0.0001678f, 0.001076f, 0.007683f, -0.006207f, -0.006233f,
      -0.00585f, -0.004894f, 0.00773f, 0.00627f, -0.0008707f, -0.00574f, -0.002068f, -0.0003157f, -0.00921f,
      -0.006275f, 0.007275f, -0.0004473f, 0.002474f, -0.009186f, 0.001432f, 0.003687f, -0.004425f, -0.002018f,
      0.00922f, -0.00788f, 0.000894f, -0.001047f, -0.001193f, 0.009094f, -0.0039f, 0.00977f, 0.00951f,
      0.00976f, 0.002201f, 0.006214f, -0.002117f, 0.006203f, 0.00278f, -0.006725f, -0.006157f, 0.003757f,
      -0.001729f, 0.005405f, -0.00904f, -0.000435f, -0.002148f, -0.00849f, 0.00923f, -0.008194f, -0.001804f,
      -0.00392f, 0.0002866f, -0.007317f, 0.005623f, -0.002657f, -0.005657f, 0.006363f, 0.00205f, 0.005215f,
      0.00376f, 0.001134f, -0.003138f, 0.00569f, 0.008446f, -0.003283f, 0.004047f, -0.00322f, -0.001756f,
      -0.006763f, 0.001577f, -0.007225f, 0.006092f, 0.004112f, -0.006554f, -0.00428f, 0.004684f, -0.000417f,
      0.00418f, -0.000349f, -0.00676f, -0.004097f, -0.00899f, 0.004936f, 0.00864f, -0.006325f, -0.004665f,
      -0.00834f, 0.00238f, 0.006153f, -0.00914f, 0.004246f, -0.00963f, 0.003986f, 0.00887f, 0.00852f,
      0.0002384f, 0.007866f, -0.002577f, 0.0007524f, -0.004887f, -0.0003715f, 0.00564f, 0.008606f, -0.009705f,
      -0.009796f, -0.001706f, -0.00965f, 0.00824f, 0.0009446f, -0.00514f, 0.00492f, 0.002787f, 0.00643f,
      -0.0002482f, 0.003603f, 0.004097f, 0.00916f, -0.005463f, -0.003786f, 0.00269f, -0.00688f, 0.002872f,
      0.0079f, 0.002403f, -0.000562f, 0.00747f, -0.00349f, 0.004925f, -0.009f, -0.003199f, -0.0008674f,
      0.004513f, 0.001112f, 0.00242f, -0.003345f, -0.00588f, -0.001415f, 0.001788f, -0.00345f, -0.007744f,
      0.005596f, -0.00871f, -0.001603f, -0.0001678f, -0.00862f, 0.00929f, -0.005604f, 0.00986f, 0.005383f,
      0.00959f, 0.00005203f, -0.002613f, 0.000881f, 0.00828f, -0.00738f, 0.001506f, 0.000615f, -0.001396f,
      0.005566f, -0.00815f, -0.00447f, 0.002577f, -0.00938f, -0.0007024f, 0.000968f, 0.00785f, 0.001473f,
      -0.004387f, 0.008286f, -0.003094f, 0.008125f, -0.004494f, -0.00425f, 0.004585f, -0.00964f, 0.002777f,
      -0.00888f, 0.005466f, 0.00231f, -0.001025f, -0.009186f, 0.004265f, 0.002234f, -0.002064f, 0.006973f,
      -0.007336f, 0.001036f, -0.00965f, -0.003597f, 0.000792f, -0.006615f, 0.00904f, 0.00902f, -0.004856f,
      -0.00782f, -0.0004456f, 0.004826f, -0.001932f, 0.003588f, -0.001571f, -0.003286f, -0.00523f, -0.002085f,
      0.004658f, 0.00324f, -0.00974f, 0.007122f, -0.00806f, -0.003452f, -0.00996f, 0.0004315f, -0.004436f,
      0.00442f, 0.0003521f, -0.0000391f, 0.00986f, 0.007553f, 0.00816f, 0.004242f, -0.00706f, 0.00857f,
      -0.009705f, -0.00789f, 0.006126f, 0.00494f, 0.001126f, -0.003017f, -0.0005965f, -0.00928f, 0.001935f,
      -0.00866f, -0.002542f, 0.003275f, 0.0001297f, -0.00935f, 0.005028f, 0.004097f, -0.006817f, 0.00791f,
      0.0001851f, -0.002525f, 0.00906f, 0.000608f, 0.0004106f, -0.00859f, -0.005623f, -0.00567f, 0.00434f,
      0.004124f, 0.000519f, 0.00947f, -0.002487f, -0.00738f, 0.009346f, -0.004936f, 0.007263f, -0.00096f,
      0.00493f, -0.00823f, 0.003119f, -0.0003824f, 0.0007586f, 0.006584f, 0.00392f, -0.008125f, 0.006313f,
      0.007812f, -0.005913f, 0.005547f, -0.0001316f, -0.007523f, 0.00768f, 0.00142f, 0.00912f, -0.003622f,
      0.00852f, 0.005966f, -0.004467f, -0.00919f, -0.00866f, -0.00875f, -0.0000665f, 0.000144f, 0.00649f,
      0.003706f, -0.001643f, -0.003508f, -0.005817f, -0.0059f, 0.008896f, 0.0088f, -0.005962f, -0.003698f,
      -0.003626f, 0.001465f, 0.003386f, 0.002172f, 0.00159f, 0.003794f, 0.00751f, 0.001184f, -0.0008216f,
      -0.006474f, 0.00761f, -0.006603f, 0.005993f, 0.003044f, 0.00322f, -0.00928f, -0.00667f, -0.00599f,
      0.00869f, 0.001393f, -0.006184f, -0.002693f, 0.003727f, -0.003624f, 0.002987f, -0.002718f, -0.001762f,
      -0.007366f, -0.00294f, -0.004993f, -0.00977f, 0.00814f, -0.001241f, 0.001603f, -0.00352f, -0.004997f,
      -0.005177f, -0.002817f, 0.002464f, 0.00763f, 0.00547f, -0.007217f, -0.00507f, 0.000908f, -0.000513f,
      0.001423f, -0.0006895f, 0.001677f, 0.001864f, -0.00401f, -0.003475f, 0.00604f, -0.003687f, -0.008606f,
      -0.00724f, -0.0061f, 0.002502f, -0.00612f, -0.003128f, 0.000557f, 0.001442f, -0.007397f, -0.0088f,
      -0.0009484f, 0.007244f, -0.008804f, -0.00847f, -0.00967f, 0.00989f, 0.00872f, -0.005753f, 0.003027f,
      0.0014105f, 0.007397f, -0.005905f, 0.007214f, 0.005665f, 0.001882f, -0.002838f, -0.003008f, -0.00795f,
      -0.000239f, 0.0064f, 0.005333f, 0.005028f, 0.006863f, -0.004f, -0.00592f, -0.001575f, 0.005398f,
      0.009575f, -0.003317f, 0.00983f, -0.0006003f, 0.005287f, 0.009094f, -0.00502f, -0.00495f, -0.00962f,
      0.000787f, 0.005604f, -0.006504f, 0.002504f, -0.004066f, -0.009766f, -0.0074f, -0.00766f, 0.009705f,
      0.00814f, -0.005157f, -0.001017f, -0.008316f, -0.00004405f, -0.00802f, -0.004677f, -0.004894f, -0.00705f,
      0.00784f, 0.00448f, -0.007553f, -0.0028f, -0.006226f, 0.0000136f, -0.004192f, -0.00755f, 0.00278f,
      0.00343f, -0.0006332f, -0.00343f, -0.004555f, -0.0093f, 0.00261f, 0.00926f, -0.005093f, 0.00627f,
      -0.00848f, -0.00984f, -0.001426f, -0.00226f, -0.002077f, -0.001703f, 0.009636f, 0.007664f, -0.003628f,
      0.002018f, -0.006012f, -0.00473f, 0.003834f, 0.00939f, -0.00827f, -0.00812f, -0.00792f, 0.00924f,
      0.00776f, 0.001537f, -0.00287f, -0.002401f, -0.00831f, -0.00903f, 0.00591f, 0.003252f, -0.006348f,
      0.001455f, 0.00674f, -0.002382f, 0.0003512f, -0.0017185f, 0.00684f, 0.00665f, 0.00782f, -0.00969f,
      0.00418f, 0.00442f, 0.00979f, 0.006382f, 0.004642f, 0.00398f, 0.007797f, 0.005234f, -0.005566f,
      -0.00903f, 0.003168f, -0.005596f, 0.00006646f, 0.00995f, -0.002335f, -0.00548f, 0.005383f, -0.004562f,
      0.00811f, -0.005035f, 0.0008745f, -0.0086f, -0.00786f, -0.00566f, -0.0096f, -0.000744f, 0.00511f,
      -0.003363f, 0.002739f, 0.002033f, 0.005455f, -0.001077f, 0.003887f, 0.00735f, 0.00757f, 0.008965f,
      -0.002888f, 0.002462f, 0.000919f, 0.0008416f, -0.003096f, 0.00875f, -0.002434f, 0.00318f, -0.002779f,
      0.00725f, 0.005062f, 0.00073f, 0.00845f, 0.003576f, 0.002874f, -0.00836f, -0.00859f, 0.00916f,
      -0.00745f, 0.00869f, 0.001855f, 0.005814f, -0.002064f, 0.0066f, -0.009346f, 0.004307f, -0.00966f,
      0.00877f, -0.002394f, -0.00977f, 0.002356f, -0.008255f, 0.001052f, 0.00495f, -0.00963f, 0.00886f,
      -0.00476f, -0.00917f, -0.000619f, -0.00593f, 0.005497f, 0.003141f, 0.002428f, 0.003363f, 0.001099f,
      0.00731f, -0.005577f, 0.00666f, -0.00328f, 0.004677f, 0.00761f, -0.00864f, -0.00873f, -0.00282f,
      -0.004177f, 0.00867f, -0.00536f, 0.004387f, -0.007294f, -0.0099f, 0.001112f, -0.001063f, 0.004284f,
      0.000729f, 0.005604f, 0.00434f, 0.00563f, -0.00618f, 0.00464f, 0.004417f, 0.00524f, -0.00052f,
      -0.002462f, -0.000902f, 0.005207f, -0.002256f, 0.000805f, -0.006252f, 0.003262f, 0.007603f, -0.000191f,
      0.003582f, -0.002598f, -0.003662f, -0.005585f, -0.00007087f, -0.00784f, -0.001778f, 0.00996f, -0.00643f,
      0.009796f, -0.002966f, 0.005848f, -0.003027f, -0.007587f, -0.003654f, -0.00882f, -0.001206f, -0.005836f,
      -0.0089f, -0.00608f, -0.003944f, -0.000564f, -0.00329f, 0.000377f, 0.000702f, 0.000859f, 0.002554f,
      0.001499f, 0.005997f, 0.0006666f, -0.00584f, 0.005337f, -0.00734f, 0.006847f, 0.00829f, 0.003925f,
      -0.00837f, -0.005886f, -0.006927f, -0.000641f, -0.0000388f, 0.003124f, 0.007427f, 0.00767f, -0.002771f,
      -0.005985f, 0.002094f, -0.007442f, -0.001377f, 0.003183f, 0.0003796f, 0.0068f, 0.0008273f, -0.002102f,
      0.003433f, -0.00931f, 0.0003903f, -0.00771f, -0.000703f, 0.003122f, 0.00833f, 0.001467f, 0.00769f,
      -0.004578f, -0.007393f, 0.0054f, -0.007797f, -0.003767f, -0.009735f, -0.0007954f, 0.005028f, -0.00809f,
      0.002352f, -0.0002111f, 0.003624f, 0.00502f, 0.001048f, 0.00922f, 0.003426f, 0.002258f, -0.00708f,
      0.00517f, -0.00919f, -0.00881f, -0.00548f, 0.00891f, 0.00919f, 0.00597f, 0.001098f, 0.004875f,
      0.004875f, 0.00846f, 0.00829f, 0.003426f, 0.001049f, 0.00669f, 0.003994f, 0.006195f, -0.004585f,
      -0.001221f, -0.000247f, -0.00613f, -0.00613f, 0.00436f, 0.006775f, -0.001169f, -0.001771f, -0.001071f,
      -0.003635f, -0.004475f, -0.00216f, -0.003502f, 0.002285f, -0.006702f, 0.0074f, 0.004845f, 0.00123f,
      -0.00434f, -0.0082f, 0.0000914f, 0.00325f, -0.00717f, -0.003687f, 0.003479f, 0.005894f, -0.002655f,
      0.00833f, 0.002365f, -0.00927f, 0.006416f, -0.0031f, 0.009834f, 0.006855f, 0.004673f, 0.00857f,
      -0.00627f, 0.00887f, -0.002636f, -0.0066f, -0.003975f, 0.003056f, -0.001572f, -0.005142f, 0.007393f,
      0.00863f, -0.000665f, -0.005146f, 0.008965f, 0.005505f, -0.001827f, -0.001454f, 0.002926f, -0.002275f,
      -0.006184f, 0.00991f, -0.005035f, -0.003462f, 0.00855f, -0.009125f, 0.002832f, 0.005817f, 0.007187f,
      0.005844f, -0.003204f, -0.002201f, -0.0095f, -0.00862f, -0.00896f, 0.00543f, 0.00010115f, 0.00392f,
      0.004917f, -0.002266f, 0.0003471f, 0.006306f, -0.004726f, -0.002298f, 0.00234f, -0.004726f, 0.00924f,
      -0.005363f, -0.0002112f, -0.0099f, 0.005604f, -0.00523f, -0.004627f, -0.001949f, -0.00936f, 0.002743f,
      -0.001635f, 0.001984f, 0.00972f, -0.00359f, 0.003296f, 0.00074f, 0.004654f, 0.00995f, -0.001584f,
      0.003048f, 0.0006003f, -0.003628f, -0.007668f, -0.002537f, -0.006584f, 0.00576f, 0.00864f, -0.00899f,
      -0.009636f, -0.005394f, 0.00433f, 0.00706f, 0.005005f, -0.004707f, 0.004597f, 0.00852f, 0.008835f,
      0.003904f, 0.00457f, 0.004128f, 0.005028f, -0.003986f, 0.005997f, 0.0002208f, 0.00777f, 0.00963f,
      0.005787f, 0.007023f, 0.00553f, 0.00449f, 0.005814f, 0.003082f, 0.0093f, 0.00472f, -0.00985f,
      0.00938f, 0.00558f, 0.007088f, 0.00391f, -0.00918f, 0.008415f, 0.00902f, 0.004173f, -0.002716f,
      -0.009926f, -0.00801f, -0.009705f, -0.0086f, -0.009865f, 0.003788f, -0.0092f, 0.00887f, -0.001495f,
      -0.00314f, -0.003246f, -0.000836f, 0.001646f, 0.00902f, -0.007233f, -0.00376f, -0.0057f, 0.005787f,
      -0.002974f, 0.00872f, 0.0086f, -0.00443f, 0.003622f, 0.004593f, 0.008026f, -0.0003214f, 0.00858f,
      -0.00338f, 0.00772f, 0.00448f, 0.00855f, 0.001066f, -0.004692f, -0.005737f, 0.007565f, -0.0002706f,
      -0.002792f, -0.00949f, 0.000827f, -0.004967f, 0.00864f, 0.00788f, 0.009094f, -0.001957f, -0.002716f,
      0.000686f, -0.00499f, -0.004173f, 0.002407f, 0.00923f, 0.001411f, -0.0005016f, 0.00746f, -0.0087f,
      -0.002703f, -0.003134f, -0.001611f, 0.007404f, -0.00999f, -0.004158f, 0.00556f, 0.0005794f, 0.003775f,
      -0.001105f, -0.00338f, 0.00999f, 0.006966f, 0.005802f, -0.009735f, -0.009834f, -0.00723f, -0.00656f,
      -0.007538f, 0.00995f, 0.00586f, 0.001463f, -0.001861f, -0.007015f, 0.005455f, -0.00492f, -0.005337f,
      -0.00855f, -0.002764f, 0.003605f, 0.00967f, -0.007256f, -0.002594f, 0.00397f, -0.00508f, -0.004555f,
      0.009476f, -0.0006495f, 0.003998f, -0.0087f, 0.007294f, -0.007748f, 0.001855f, -0.0002816f, -0.00983f,
      -0.007416f, 0.004444f, 0.003036f, 0.005066f, 0.001116f, -0.0001506f, -0.003181f, -0.003258f, -0.00816f,
      0.00821f, -0.0007715f, 0.00669f, 0.002674f, 0.004074f, 0.009605f, 0.001936f, -0.0052f, -0.002779f,
      0.003435f, 0.003592f, -0.00787f, 0.002615f, 0.007996f, 0.002047f, 0.002438f, 0.000739f, -0.002443f,
      0.00817f, 0.009995f, 0.00749f, 0.00953f, 0.007427f, -0.003246f, -0.004795f, 0.003834f, 0.0087f,
      -0.00863f, 0.003105f, -0.003313f, -0.006187f, 0.005104f, -0.00093f, 0.004158f, 0.003963f, -0.00579f,
      -0.004044f, 0.004044f, -0.0005593f, -0.00388f, -0.00249f, 0.006115f, 0.00322f, 0.007347f, 0.00813f,
      -0.005142f, -0.0004606f, 0.00646f, 0.002186f, 0.00812f, 0.004818f, 0.0009236f, -0.00864f, 0.00948f,
      -0.003057f, 0.003445f, -0.004444f, 0.001763f, -0.005806f, 0.001699f, 0.00843f, -0.007423f, -0.001351f,
      -0.007317f, -0.001196f, 0.002996f, 0.005066f, 0.003227f, 0.00547f, -0.00923f, 0.0008106f, 0.00789f,
      -0.006508f, -0.0003939f, -0.002443f, 0.007107f, -0.00692f, -0.007645f, -0.00353f, 0.00661f, 0.000988f,
      -0.00769f, -0.003134f, 0.002548f, 0.00495f, 0.0034f, 0.001454f, 0.00344f, -0.00323f, -0.006203f,
      0.001063f, 0.008736f, -0.00737f, 0.00234f, -0.00315f, -0.008865f, -0.003918f, 0.006042f, 0.0003307f,
      -0.001405f, 0.002129f, -0.00682f, 0.000836f, -0.005436f, 0.008385f, -0.002783f, -0.0007734f, -0.007088f,
      -0.005924f, 0.00951f, 0.000002f, -0.00504f, -0.005474f, -0.00897f, 0.00339f, -0.003044f, 0.0019245f,
      0.00596f, 0.00756f, -0.005936f, 0.007416f, -0.005173f, 0.006367f, 0.0015545f, -0.001073f, 0.008095f,
      0.004868f, 0.0000308f, -0.005302f, -0.0003858f, -0.00421f, -0.00386f, 0.00925f, 0.004604f, 0.001006f,
      -0.004482f, 0.00634f, -0.006126f, -0.00878f, 0.0095f, -0.006985f, -0.00575f, -0.001845f, -0.002335f,
      0.00908f, 0.00764f, -0.00405f, 0.003431f, 0.004726f, 0.0002171f, -0.005314f, -0.00693f, 0.00867f,
      0.0007024f, -0.007217f, 0.006042f, -0.0002111f, 0.00475f, -0.00635f, 0.00984f, 0.00829f, -0.0008802f,
      -0.005093f, -0.007996f, -0.003607f, -0.00965f, -0.001188f, -0.002707f, 0.002533f, 0.00328f, -0.004807f,
      -0.002724f, -0.005733f, 0.007996f, -0.003893f, -0.0002323f, -0.00577f, -0.007263f, 0.00416f, -0.007385f,
      -0.004906f, 0.002007f, -0.00773f, -0.0004334f, -0.00542f, -0.0009217f, 0.008545f, 0.0005693f, 0.0094f,
      -0.000956f, -0.002106f, -0.0082f, -0.006363f, 0.00431f, -0.001059f, -0.0054f, 0.002123f, 0.0004594f,
      -0.003489f, -0.005173f, -0.007595f, 0.007782f, -0.0001341f, 0.00977f, -0.00463f, -0.0002378f, -0.002296f,
      0.00667f, 0.00701f, 0.001323f, -0.001699f, 0.00955f, -0.0091f, 0.0089f, 0.00791f, -0.0003197f,
      0.007835f, -0.00828f, 0.00854f, 0.00239f, 0.008385f, 0.001974f, 0.000486f, 0.00991f, 0.006542f,
      0.007866f, -0.004803f, -0.004913f, -0.00513f, -0.0004153f, 0.00995f, -0.00516f, -0.003317f, 0.00682f,
      0.0004165f, -0.00903f, -0.005344f, 0.00786f, 0.003769f, 0.004158f, 0.0002446f, 0.00589f, -0.002949f,
      0.0073f, -0.002398f, -0.004757f, 0.0002432f, -0.00439f, -0.00454f, 0.000453f, 0.00823f, -0.009575f,
      0.00535f, -0.008575f, -0.00893f, 0.004303f, 0.00502f, 0.00617f, -0.004402f, 0.00919f, -0.00865f,
      0.00876f, 0.003645f, 0.0002997f, -0.00925f, -0.007076f, 0.004448f, 0.005196f, -0.003986f, 0.007084f,
      -0.000285f, -0.002855f, -0.000422f, -0.00872f, -0.005013f, 0.00952f, -0.008446f, -0.004044f, -0.00907f,
      0.007072f, -0.00918f, -0.007835f, 0.000878f, -0.006847f, -0.006f, 0.00731f, -0.001876f, -0.002565f,
      -0.003584f, -0.003006f, -0.00723f, -0.003433f, 0.0004973f, -0.00795f, 0.0005007f, 0.00608f, 0.00671f,
      0.0001765f, 0.00439f, -0.003738f, -0.006035f, 0.00010353f, -0.00374f, 0.0008683f, 0.00773f, -0.0004847f,
      -0.000992f, 0.004658f, -0.003555f, -0.0056f, -0.001982f, 0.00812f, 0.003386f, -0.001584f, 0.003508f,
      -0.006138f, -0.00587f, 0.001421f, -0.009094f, -0.00468f, -0.0086f, 0.003637f, 0.00896f, 0.00804f,
      -0.00744f, 0.002382f, -0.0097f, 0.000659f, 0.007782f, 0.002981f, -0.00869f, 0.0000934f, -0.00882f,
      0.002771f, -0.009544f, 0.0035f, 0.004124f, -0.0014f, -0.006294f, -0.007614f, 0.00931f, 0.009674f,
      0.0003185f, -0.004295f, 0.007084f, -0.0035f, -0.00334f, -0.001754f, 0.001216f, -0.004375f, 0.003244f,
      0.0001901f, 0.001547f, 0.007183f, 0.006447f, 0.005108f, 0.00679f, 0.001068f, -0.00587f, 0.005745f,
      -0.00634f, 0.0058f, 0.006985f, -0.000697f, 0.00008917f, 0.007835f, -0.0004838f, 0.004795f, -0.006832f,
      0.002398f, 0.00687f, -0.001582f, 0.00709f, -0.00908f, -0.001573f, 0.009865f, -0.001476f, -0.000526f,
      0.00477f, 0.008026f, -0.00171f, 0.00979f, -0.005592f, 0.0006247f, -0.00774f, 0.00463f, -0.006676f,
      0.004368f, -0.002373f, -0.005127f, -0.0013275f, -0.002306f, -0.0087f, 0.00997f, 0.005493f, 0.003786f,
      -0.004414f, -0.005947f, 0.003181f, -0.0004156f, 0.00909f, -0.00656f, 0.001926f, 0.0003731f, -0.009636f,
      0.003124f, -0.0000686f, -0.001972f, -0.006584f, 0.0009604f, 0.004086f, 0.009865f, 0.001302f, -0.00989f,
      -0.0086f, 0.005177f, 0.006493f, -0.00523f, -0.00443f, 0.001586f, 0.00937f, 0.007458f, 0.001883f,
      0.00774f, 0.0004454f, 0.000493f, 0.0003722f, -0.00486f, 0.006435f, 0.002642f, 0.00432f, -0.00272f,
      -0.007446f, -0.007397f, 0.00361f, 0.003618f, 0.003956f, -0.001175f, 0.00832f, 0.00794f, 0.001658f,
      0.00123f, -0.003918f, 0.001215f, -0.007427f, 0.003708f, 0.00492f, -0.00968f, 0.008896f, -0.006786f,
      -0.005856f, 0.006573f, 0.003876f, -0.003983f, 0.00411f, 0.0076f, -0.0008364f, -0.00496f, 0.008026f,
      -0.00986f, -0.001429f, -0.007236f, -0.002172f, -0.003004f, -0.0017185f, -0.00353f, -0.00817f, -0.004353f,
      -0.003458f, 0.002663f, -0.00599f, 0.002125f, -0.00625f, -0.00913f, -0.009796f, -0.004574f, -0.00978f,
      -0.00398f, -0.006096f, 0.003708f, 0.007214f, 0.00444f, 0.003742f, 0.004547f, 0.006042f, 0.001542f,
      0.002424f, 0.0005617f, 0.006477f, -0.002382f, 0.0009637f, -0.00462f, -0.000934f, 0.0004268f, 0.00975f,
      0.002277f, 0.001031f, -0.007103f, 0.006615f, 0.00199f, 0.009f, 0.00995f, -0.002514f, -0.0016575f,
      -0.00875f, 0.00936f, -0.007133f, 0.007412f, -0.001572f, -0.00862f, -0.00675f, 0.009445f, -0.00819f,
      0.004597f, -0.005493f, 0.004894f, -0.004807f, 0.00346f, -0.00114f, 0.006638f, -0.005882f, 0.0041f,
      -0.002684f, -0.0006037f, -0.00842f, 0.001939f, -0.0008016f, 0.00265f, -0.005383f, 0.00963f, 0.0063f,
      0.006386f, 0.004463f, -0.004173f, -0.006317f, 0.003534f, -0.00781f, -0.001414f, -0.004723f, -0.003096f,
      -0.001367f, 0.00955f, -0.0000178f, -0.007214f, 0.00985f, -0.003782f, 0.005688f, -0.002445f, 0.00185f,
      0.00784f, 0.00203f, 0.0003746f, -0.00935f, 0.00559f, 0.00718f, 0.005905f, 0.002926f, 0.006268f,
      0.0002078f, 0.001244f, 0.00467f, 0.006405f, -0.0005364f, 0.00503f, -0.0004387f, 0.006252f, -0.002594f,
      0.001791f, -0.00807f, -0.001451f, -0.0034f, 0.00958f, 0.003035f, -0.00348f, 0.004818f, 0.008644f,
      -0.0005145f, -0.004673f, 0.008934f, 0.00756f, -0.001786f, -0.005634f, -0.002981f, -0.007107f, 0.001145f,
      0.003677f, 0.004997f, 0.009766f, 0.0005856f, -0.002384f, 0.004177f, -0.00965f, 0.005924f, -0.005596f,
      0.004505f, 0.000578f, 0.00663f, -0.006638f, 0.001535f, 0.002502f, 0.002907f, 0.00447f, 0.002016f,
      0.008865f, 0.00828f, -0.00975f, 0.0002487f, -0.00796f, -0.008286f, -0.002083f, -0.00471f, 0.007187f,
      0.004326f, 0.007206f, 0.004307f, 0.009346f, -0.00758f, -0.007545f, 0.00349f, 0.0018425f, -0.00837f,
      -0.007935f, -0.002258f, 0.003757f, -0.0014f, 0.000081f, 0.00449f, -0.000318f, 0.006485f, -0.001184f,
      -0.001842f, 0.009476f, 0.00818f, -0.00986f, 0.001612f, -0.00779f, 0.006676f, -0.0013075f, 0.00464f,
      -0.002117f, -0.0087f, 0.00965f, 0.001394f, 0.00818f, -0.005493f, 0.004673f, -0.00439f, -0.00557f,
      -0.001841f, -0.00948f, 0.00607f, 0.00551f, -0.002834f, 0.004883f, -0.00712f, 0.006573f, -0.002064f,
      0.0008054f, -0.006508f, 0.004467f, 0.00773f, 0.004787f, 0.00523f, -0.001751f, -0.005657f, 0.000278f,
      -0.001822f, -0.00639f, -0.003477f, -0.006767f, -0.007782f, 0.005375f, -0.00726f, 0.007248f, 0.0008335f,
      -0.001856f, -0.00009865f, -0.006054f, 0.006786f, -0.005665f, -0.007393f, -0.0007014f, -0.007046f, -0.0065f,
      -0.00645f, 0.002195f, 0.004818f, 0.00909f, -0.00862f, 0.007614f, -0.00499f, 0.007423f, -0.001478f,
      -0.005028f, -0.007107f, -0.00488f, 0.00322f, -0.003801f, 0.0018425f, 0.001862f, 0.007713f, -0.008675f,
      0.001135f, 0.00788f, -0.006866f, -0.00776f, 0.001423f, -0.00392f, -0.00908f, 0.00918f, -0.006706f,
      -0.00828f, -0.00358f, -0.00956f, -0.00823f, 0.00656f, -0.00617f, -0.004395f, 0.002705f, -0.001398f,
      0.003265f, 0.007793f, 0.00664f, 0.009285f, 0.00851f, 0.00416f, -0.00923f, -0.006733f, 0.00934f,
      -0.00564f, -0.001064f, 0.001106f, 0.00943f, 0.005024f, 0.00793f, -0.005302f, -0.00376f, -0.0005045f,
      0.005325f, -0.002134f, -0.001494f, -0.00891f, -0.00803f, 0.00958f, -0.0000229f, -0.003668f, 0.00602f,
      -0.003649f, -0.002918f, 0.006573f, 0.005146f, -0.009995f, 0.00864f, -0.008255f, 0.004868f, 0.001078f,
      -0.003546f, 0.00235f, 0.005764f, -0.005116f, 0.009186f, -0.008255f, -0.00216f, -0.008f, -0.009125f,
      -0.002754f, -0.0083f, -0.002539f, -0.0007524f, -0.00843f, 0.003647f, -0.00156f, 0.00498f, -0.007904f,
      -0.00502f, 0.00919f, 0.003862f, 0.00599f, 0.001332f, -0.00788f, 0.007374f, 0.001653f, -0.00406f,
      -0.008545f, -0.00444f, -0.00971f, -0.002436f, -0.009834f, -0.005573f, -0.002323f, -0.007126f, 0.004803f,
      -0.00913f, 0.002483f, -0.004704f, -0.0014515f, -0.001035f, -0.008934f, -0.001855f, -0.0071f, 0.00979f,
      -0.008255f, 0.001663f, -0.001383f, 0.000364f, -0.003595f, -0.002163f, 0.002136f, 0.004894f, 0.006966f,
      0.00925f, 0.006557f, -0.0089f, -0.0007167f, 0.002699f, 0.003483f, 0.003017f, 0.004223f, 0.006042f,
      -0.002342f, -0.004868f, 0.003157f, 0.006165f, 0.001519f, -0.00874f, -0.004856f, -0.004116f, 0.002634f,
      -0.001233f, -0.008736f, 0.003529f, -0.001974f, 0.00121f, -0.0006013f, -0.002737f, -0.00596f, 0.007027f,
      -0.00496f, -0.002726f, -0.00787f, 0.001581f, 0.00381f, -0.004932f, 0.007027f, -0.003616f, -0.000989f,
      0.003532f, 0.002346f, 0.0000479f, 0.002907f, -0.004353f, 0.005424f, 0.003124f, 0.00985f, 0.003f,
      -0.007805f, 0.001684f, -0.001324f, 0.0005107f, 0.00483f, -0.00992f, 0.000786f, -0.003649f, -0.0006337f,
      -0.001443f, 0.00782f, 0.008194f, -0.00819f, -0.00844f, -0.004906f, -0.006355f, 0.002932f, 0.004242f,
      0.000638f, -0.00259f, 0.00585f, -0.00864f, 0.00378f, -0.00279f, -0.00319f, -0.001805f, -0.002768f,
      -0.0007725f, -0.004875f, 0.003784f, 0.00947f, -0.008736f, 0.003262f, -0.00325f, -0.003826f, 0.007904f,
      0.00002706f, 0.006187f, -0.001488f, -0.001711f, -0.003317f, 0.007446f, -0.00699f, -0.005573f, 0.00164f,
      0.00938f, 0.0002334f, 0.003819f, -0.001427f, 0.00992f, -0.003433f, -0.0006833f, -0.00492f, 0.005493f,
      0.003014f, -0.006187f, -0.002325f, 0.00741f, -0.009056f, 0.005604f, -0.003874f, 0.00869f, 0.0001504f,
      0.005356f, 0.001178f, 0.00786f, 0.003292f, 0.00947f, -0.002808f, -0.00424f, -0.00999f, 0.004818f,
      0.00372f, -0.003748f, 0.001496f, 0.009796f, 0.0000038f, 0.00379f, 0.0003746f, -0.004147f, 0.007195f,
      -0.0095f, 0.001072f, 0.002129f, 0.00889f, 0.003273f, 0.006958f, -0.004894f, 0.0006795f, 0.00892f,
      -0.004356f, 0.00594f, -0.002378f, 0.00969f, -0.0081f, 0.0003927f, 0.00789f, 0.00343f, 0.00479f,
      -0.0005517f, -0.00652f, 0.000332f, 0.00876f, -0.001309f, -0.002495f, -0.00831f, 0.007786f, -0.00512f,
      -0.003832f, -0.0006423f, -0.003162f, 0.00807f, -0.006298f, -0.003601f, 0.002438f, 0.0017395f, 0.002686f,
      -0.001712f, 0.00424f, 0.00632f, -0.00935f, 0.000598f, 0.005714f, -0.00921f, -0.002935f, 0.008064f,
      -0.001802f, -0.002634f, -0.006786f, 0.00976f, 0.00867f, 0.004066f, 0.002306f, 0.001495f, -0.0003717f,
      -0.00597f, 0.00958f, -0.00881f, 0.00856f, -0.00538f, -0.008575f, -0.003626f, 0.006702f, 0.00932f,
      0.001552f, 0.0006847f, 0.00159f, 0.002314f, 0.008606f, 0.005955f, 0.00862f, 0.0003278f, 0.003115f,
      -0.006863f, -0.0051f, -0.00824f, 0.00592f, -0.005653f, 0.00871f, -0.008286f, 0.0005655f, -0.005154f,
      -0.008766f, 0.008896f, -0.009674f, 0.003782f, -0.000774f, 0.00323f, -0.00935f, 0.007694f, -0.003578f,
      -0.00912f, 0.007362f, -0.00561f, 0.00817f, -0.00852f, -0.00006425f, -0.003166f, 0.0004108f, 0.006325f,
      -0.00928f, -0.008026f, -0.003891f, -0.005924f, -0.004284f, 0.00515f, -0.00749f, 0.002983f, 0.003885f,
      0.006535f, -0.001574f, 0.005695f, -0.009155f, -0.006996f, -0.0012665f, 0.002983f, -0.00932f, -0.00575f,
      -0.008545f, -0.0005817f, 0.002466f, -0.003382f, 0.007477f, 0.00166f, 0.004562f, -0.001331f, -0.0095f,
      -0.00291f, 0.002815f, -0.009796f, -0.00496f, 0.005592f, -0.00365f, -0.00609f, 0.0008597f, 0.00516f,
      0.003986f, 0.002157f, 0.00934f, -0.003363f, 0.000835f, 0.003725f, 0.002106f, -0.005993f, 0.00795f,
      0.003122f, -0.003313f, -0.005383f, 0.0004141f, 0.006466f, 0.003517f, -0.00809f, 0.005714f, -0.007294f,
      -0.001924f, -0.002457f, -0.001897f, -0.001449f, 0.00543f, 0.000466f, 0.008125f, -0.002316f, 0.003128f,
      -0.008255f, -0.001908f, 0.00911f, 0.00793f, -0.001612f, -0.00899f, -0.004013f, -0.002962f, 0.001639f,
      -0.006916f, -0.009056f, -0.005795f, -0.001411f, -0.00745f, 0.003126f, 0.000916f, -0.0007496f, 0.003273f,
      0.005184f, 0.004128f, 0.003195f, -0.004635f, 0.004826f, 0.00745f, 0.006348f, -0.008865f, -0.00217f,
      0.006275f, -0.00971f, 0.005478f, -0.003456f, 0.0065f, 0.00943f, -0.005703f, 0.002666f, -0.005745f,
      -0.006134f, 0.003513f, 0.00683f, -0.004803f, -0.003841f, -0.006435f, -0.007122f, 0.001902f, 0.005844f,
      0.007313f, 0.004723f, 0.001233f, -0.00402f, 0.001288f, 0.002878f, 0.004196f, -0.002884f, -0.007454f,
      0.000933f, -0.003576f, -0.005608f, -0.00908f, 0.00426f, 0.001788f, -0.004856f, -0.008965f, -0.00546f,
      -0.004684f, -0.002708f, -0.006145f, 0.002111f, -0.000599f, -0.007187f, -0.002018f, -0.001014f, -0.006676f,
      -0.00335f, -0.00528f, -0.009224f, -0.009285f, -0.00063f, -0.0045f, -0.005157f, 0.008865f, 0.008835f,
      -0.00672f, 0.002237f, 0.002687f, 0.005703f, 0.00585f, 0.007175f, -0.007496f, 0.0002145f, 0.00924f,
      -0.00611f, -0.003202f, -0.0057f, -0.001237f, 0.006752f, 0.001596f, -0.001424f, 0.007492f, 0.00459f,
      -0.00668f, -0.001726f, 0.00209f, 0.001924f, 0.0008316f, 0.0004334f, 0.001638f, 0.005665f, 0.000911f,
      -0.00552f, 0.00619f, -0.00979f, 0.00549f, 0.004967f, 0.00818f, -0.006157f, -0.00816f, 0.001334f,
      0.0002472f, 0.00653f, 0.005257f, 0.0000934f, -0.00261f, 0.00755f, 0.000494f, 0.001341f, 0.00236f,
      -0.00876f, 0.005054f, -0.00503f, 0.007465f, -0.005676f, 0.003174f, -0.006325f, -0.005238f, -0.005608f,
      0.0002413f, -0.003477f, -0.00379f, -0.002457f, 0.002943f, -0.006855f, 0.001733f, 0.006504f, -0.004406f,
      -0.00929f, -0.00009567f, 0.000722f, 0.001004f, -0.00633f, 0.001915f, -0.001345f, -0.002802f, -0.00858f,
      -0.001694f, -0.000937f, 0.004486f, -0.00567f, 0.000247f, 0.007782f, -0.0036f, -0.003588f, 0.00717f,
      -0.00928f, 0.00838f, -0.0063f, 0.00916f, 0.005352f, 0.00736f, 0.00083f, -0.007248f, -0.005722f,
      0.00325f, -0.00503f, 0.001647f, 0.007767f, -0.00539f, 0.0065f, -0.002151f, 0.003359f, 0.0002371f,
      -0.007057f, 0.000602f, 0.00692f, -0.008415f, -0.001443f, 0.006783f, -0.00778f, 0.00946f, -0.002735f,
      -0.006832f, 0.00419f, -0.009315f, 0.00963f, -0.003994f, -0.00833f, 0.00411f, 0.0076f, 0.005817f,
      -0.001542f, -0.003956f, 0.004513f, 0.001667f, -0.002378f, -0.003075f, 0.002481f, -0.001739f, -0.005566f,
      -0.002113f, 0.003263f, -0.00797f, -0.008675f, 0.006916f, 0.002848f, 0.008446f, -0.004627f, -0.002216f,
      -0.0005455f, -0.00882f, 0.00846f, 0.001422f, -0.000527f, -0.00826f, 0.0012245f, 0.006226f, -0.008316f,
      0.002134f, -0.006298f, 0.00672f, -0.008026f, 0.003248f, 0.0046f, 0.001113f, 0.000221f, 0.000791f,
      0.00836f, 0.007805f, 0.006355f, 0.004723f, 0.000991f, -0.00904f, 0.007164f, 0.00896f, 0.00788f,
      0.004128f, -0.003473f, -0.00242f, 0.003466f, 0.003286f, 0.002634f, 0.009865f, 0.006947f, -0.0004823f,
      -0.005455f, 0.003603f, 0.002008f, -0.004536f, 0.006187f, 0.005722f, -0.00010717f, 0.00227f, 0.00967f,
      -0.004883f, -0.0011015f, 0.009285f, 0.002121f, -0.006718f, 0.00782f, 0.00481f, 0.002974f, -0.002855f,
      -0.001182f, -0.000961f, -0.002497f, -0.005707f, -0.00536f, -0.000726f, -0.004868f, -0.000473f, -0.002764f,
      0.0002033f, -0.00961f, -0.00828f, -0.001335f, 0.005314f, 0.007263f, 0.005386f, -0.0006895f, 0.00444f,
      -0.00443f, 0.001597f, 0.00753f, 0.005608f, 0.002354f, 0.00399f, 0.003551f, 0.0035f, 0.00319f,
      0.0017185f, -0.006195f, -0.004467f, 0.006042f, -0.007217f, -0.00907f, 0.004025f, -0.00671f, -0.002226f,
      -0.00557f, 0.000518f, -0.00805f, 0.008865f, -0.007195f, -0.004032f, -0.005047f, 0.007072f, -0.003544f,
      -0.00706f, -0.000232f, -0.00829f, -0.00835f, -0.002449f, 0.002384f, -0.00886f, -0.00177f, -0.00641f,
      0.006733f, -0.001213f, -0.005184f, 0.009995f, 0.006573f, 0.003773f, -0.00962f, 0.003693f, 0.003815f,
      0.004353f, 0.00224f, 0.0003662f, 0.007187f, 0.00817f, -0.002918f, -0.006615f, 0.00834f, 0.002783f,
      -0.000913f, 0.004993f, -0.006687f, -0.008224f, 0.00864f, -0.000776f, -0.003668f, 0.002398f, 0.001138f,
      0.001902f, -0.004894f, 0.00398f, 0.001741f, -0.00922f, 0.002316f, 0.0000156f, 0.00923f, -0.004314f,
      0.00844f, -0.002323f, -0.001928f, 0.006115f, 0.006283f, -0.001401f, -0.006443f, 0.00693f, 0.007225f,
      0.0005593f, -0.00996f, -0.00842f, -0.001854f, 0.001111f, 0.00157f, -0.003658f, -0.0003986f, 0.005455f,
      0.004204f, -0.006065f, 0.00812f, -0.00642f, -0.004932f, -0.00778f, 0.004032f, 0.005814f, 0.00329f,
      -0.007164f, -0.00576f, 0.002708f, -0.005424f, -0.006355f, -0.003983f, -0.006695f, -0.00661f, 0.005814f,
      -0.007137f, -0.00739f, -0.001341f, 0.000845f, 0.000429f, -0.002764f, 0.006496f, 0.00785f, -0.00622f,
      0.003235f, 0.00425f, -0.00612f, 0.00803f, 0.007404f, -0.001365f, 0.002625f, 0.001886f, 0.003359f,
      -0.00518f, -0.002394f, 0.00475f, 0.003391f, 0.00693f, -0.002079f, -0.000818f, -0.002357f, -0.005272f,
      -0.002317f, -0.000729f, 0.004074f, 0.005486f, 0.006023f, -0.006363f, 0.00527f, -0.003586f, -0.00925f,
      0.003809f, 0.00087f, 0.007133f, -0.001788f, 0.002201f, 0.00955f, 0.003735f, 0.007324f, -0.00614f,
      -0.007187f, -0.006783f, -0.006145f, -0.004665f, 0.007175f, 0.00984f, 0.00314f, 0.008064f, 0.007336f,
      -0.00337f, -0.00559f, 0.004944f, -0.007744f, -0.00197f, -0.006714f, -0.002281f, -0.002087f, 0.0009074f,
      -0.00753f, 0.004993f, 0.00319f, -0.002535f, -0.001945f, 0.0008793f, -0.003357f, 0.004246f, -0.00838f,
      0.007698f, 0.001307f, 0.001717f, 0.00824f, -0.001335f, -0.0002145f, 0.00561f, -0.007168f, -0.001333f,
      -0.00551f, -0.003637f, -0.007786f, 0.001738f, 0.007748f, 0.001321f, -0.001924f, 0.006046f, -0.009125f,
      0.009674f, 0.006313f, 0.002666f, 0.002287f, -0.00956f, -0.004803f, -0.008675f, 0.003038f, -0.00514f,
      0.00935f, 0.006756f, 0.004425f, 0.002203f, 0.00642f, 0.004555f, 0.00657f, 0.00157f, 0.00652f,
      -0.000512f, 0.003416f, 0.00883f, -0.003372f, -0.001136f, -0.00302f, 0.007435f, -0.00564f, 0.001519f,
      -0.007687f, -0.00783f, -0.008736f, 0.003899f, -0.00231f, 0.006927f, 0.00558f, -0.007786f, 0.008156f,
      0.004417f, -0.004173f, 0.008865f, 0.004707f, 0.002438f, -0.008896f, 0.00009686f, -0.00338f, 0.002985f,
      0.0000722f, 0.004047f, 0.00991f, 0.00222f, 0.00381f, -0.003147f, 0.0081f, 0.00392f, 0.001678f,
      -0.00647f, 0.00942f, -0.002876f, -0.001987f, -0.00758f, -0.003983f, -0.00814f, 0.00255f, -0.001071f,
      0.006855f, -0.00676f, -0.00801f, 0.00399f, 0.002998f, 0.003906f, -0.002068f, 0.005444f, -0.003128f,
      0.001452f, -0.000623f, 0.007122f, -0.003498f, -0.000979f, -0.003366f, -0.001828f, 0.004135f, 0.006786f,
      -0.003593f, -0.00814f, -0.00749f, -0.004894f, 0.009445f, -0.00828f, -0.005108f, -0.005836f, -0.002945f,
      -0.008125f, -0.001417f, -0.003443f, 0.00201f, 0.001321f, 0.00578f, 0.00224f, -0.00895f, -0.001515f,
      -0.008194f, 0.00883f, -0.000655f, -0.00831f, 0.005695f, 0.00663f, 0.00704f, -0.00393f, 0.003603f,
      -0.005608f, 0.00107f, -0.00902f, -0.0001382f, 0.006287f, 0.006393f, 0.0005302f, 0.00898f, 0.00172f,
      0.0033f, -0.001728f, -0.004436f, 0.006794f, 0.001925f, -0.00698f, 0.002726f, -0.00372f, 0.003744f,
      0.007004f, 0.002556f, -0.00895f, -0.005096f, 0.003044f, -0.002342f, -0.00802f, 0.0067f, 0.006172f,
      0.0005546f, 0.009f, 0.006405f, 0.003557f, -0.006527f, 0.002508f, -0.002115f, -0.00497f, 0.004852f,
      0.002605f, 0.009155f, -0.00941f, 0.000894f, -0.00825f, 0.005333f, 0.006023f, -0.001292f, 0.009445f,
      -0.007217f, 0.003368f, -0.007156f, -0.006386f, -0.00293f, 0.00218f, -0.00803f, 0.00927f, 0.008965f,
      0.001402f, 0.00525f, -0.00784f, 0.00418f, -0.00978f, -0.003138f, 0.002974f, 0.001657f, -0.009834f,
      0.001901f, -0.00948f, 0.005455f, -0.001604f, 0.00559f, 0.006447f, 0.0008035f, -0.002773f, 0.006332f,
      -0.00896f, 0.00488f, 0.004177f, -0.00319f, 0.00708f, 0.0003064f, -0.0007687f, -0.003065f, 0.005558f,
      -0.003864f, 0.003887f, -0.00855f, 0.006237f, 0.008415f, -0.002693f, -0.002817f, -0.00904f, 0.003407f,
      0.000946f, -0.00738f, -0.00562f, -0.0009713f, -0.003506f, -0.00766f, 0.00953f, -0.004005f, 0.00867f,
      0.0004733f, -0.005787f, 0.0005293f, 0.006996f, 0.001659f, 0.000469f, 0.001537f, 0.002247f, -0.004242f,
      0.00243f, -0.004093f, -0.007355f, -0.001f, 0.006374f, -0.004963f, 0.006035f, 0.005245f, -0.00839f,
      0.002262f, -0.008286f, 0.00845f, 0.00911f, -0.001388f, -0.001848f, -0.0008616f, 0.006363f, 0.002584f,
      -0.002827f, -0.00755f, -0.009834f, 0.002735f, -0.001286f, 0.006f, 0.001821f, -0.001493f, -0.00819f,
      -0.0003796f, 0.008606f, 0.000496f, 0.001856f, -0.00668f, -0.009186f, -0.00736f, 0.0048f, -0.003502f,
      0.001626f, -0.0001339f, -0.006126f, -0.00596f, -0.0001252f, 0.001953f, 0.009575f, -0.001304f, 0.004192f,
      -0.006035f, -0.001251f, 0.007587f, 0.001031f, -0.00928f, 0.00793f, 0.00653f, 0.0007644f, -0.002647f,
      0.003609f, -0.00461f, 0.000423f, -0.000656f, 0.005367f, -0.00425f, 0.004215f, 0.006554f, 0.005634f,
      -0.001172f, 0.00472f, -0.0002402f, 0.003582f, 0.00738f, 0.00301f, 0.005417f, 0.009254f, 0.007145f,
      -0.0094f, 0.000404f, 0.00837f, -0.00894f, 0.004658f, 0.0004907f, -0.001399f, -0.00873f, 0.0008955f,
      -0.001738f, -0.001934f, 0.003742f, 0.002077f, -0.004063f, -0.007736f, -0.001259f, 0.00867f, 0.00488f,
      0.006584f, -0.00822f, -0.00585f, 0.006927f, -0.003298f, -0.004593f, 0.000567f, -0.004543f, -0.007378f,
      0.00718f, -0.00876f, 0.005707f, 0.00701f, 0.001537f, 0.005993f, -0.0044f, 0.00847f, 0.00694f,
      0.00419f, -0.00511f, 0.00535f, 0.000936f, -0.0007434f, 0.001556f, -0.0008616f, -0.0085f, 0.003342f,
      0.00982f, 0.005077f, 0.005566f, -0.003716f, 0.00839f, 0.007786f, -0.00749f, -0.007614f, -0.00774f,
      0.00209f, 0.005894f, -0.007534f, 0.003998f, -0.00518f, -0.00033f, -0.00831f, -0.00556f, 0.004837f,
      -0.001809f, -0.00423f, 0.00916f, -0.006786f, 0.009476f, 0.00841f, -0.000718f, 0.002834f, -0.00947f,
      0.0001942f, -0.007904f, -0.003672f, -0.001356f, -0.004658f, -0.005825f, 0.002747f, -0.00737f, 0.00845f,
      0.005226f, -0.002941f, -0.005226f, -0.00415f, 0.00848f, 0.0007825f, -0.005276f, 0.003502f, -0.005974f,
      0.00866f, -0.0076f, 0.003042f, -0.003267f, -0.00536f, -0.006935f, 0.007515f, 0.008255f, 0.003098f,
      -0.007183f, 0.007355f, -0.00878f, -0.0001291f, -0.0009227f, 0.000577f, 0.00787f, 0.003855f, 0.005337f,
      -0.004837f, 0.005676f, 0.004658f, -0.00798f, 0.006424f, -0.007534f, 0.002682f, -0.003042f, 0.00868f,
      -0.003332f, 0.00318f, 0.00199f, 0.001096f, 0.00871f, 0.005028f, -0.001416f, 0.006233f, -0.007736f,
      0.00808f, -0.001244f, 0.001611f, 0.005127f, 0.00781f, -0.003036f, -0.00453f, -0.00516f, 0.007233f,
      -0.001684f, -0.002474f, 0.002844f, -0.00723f, -0.002401f, 0.0015f, -0.005444f, -0.003035f, -0.00929f,
      0.00947f, 0.00247f, 0.004017f, 0.0008864f, -0.003862f, 0.0062f, -0.00172f, -0.00449f, 0.00796f,
      0.009445f, 0.007687f, -0.007034f, -0.001731f, -0.00585f, -0.005653f, -0.002281f, 0.004925f, -0.006744f,
      -0.002542f, 0.005775f, 0.00861f, 0.003054f, 0.00666f, -0.00694f, -0.00822f, -0.001123f, 0.006557f,
      -0.00476f, 0.006397f, 0.00957f, 0.00888f, -0.003952f, -0.006313f, 0.001164f, 0.001948f, -0.00758f,
      0.007263f, -0.00801f, 0.00924f, 0.009476f, -0.00979f, 0.007748f, -0.00533f, 0.006195f, 0.00659f,
      0.003437f, 0.00546f, -0.00859f, 0.002409f, -0.006824f, -0.006172f, 0.00663f, 0.004215f, 0.00291f,
      0.001303f, -0.007786f, -0.000654f, 0.00965f, -0.002867f, 0.002117f, 0.00484f, -0.002012f, -0.004826f,
      -0.00801f, -0.00259f, 0.002625f, -0.000174f, 0.006844f, -0.005554f, -0.001617f, 0.00741f, -0.00145f,
      -0.001762f, 0.005222f, 0.001931f, 0.006676f, -0.002014f, 0.005676f, -0.001987f, 0.003426f, -0.00088f,
      0.002485f, -0.007698f, -0.00604f, 0.006687f, -0.003902f, -0.00783f, -0.00817f, 0.00841f, 0.006134f,
      -0.00659f, -0.004807f, 0.00649f, -0.00855f, -0.00605f, -0.003489f, 0.00594f, -0.00818f, -0.001544f,
      0.003778f, 0.00706f, -0.0002632f, 0.005882f, 0.003763f, 0.003439f, 0.00872f, 0.004265f, 0.00522f,
      -0.00886f, -0.00803f, -0.0003037f, -0.00807f, -0.006756f, 0.00789f, -0.00428f, -0.000516f, 0.005196f,
      -0.00981f, 0.00926f, -0.007507f, -0.00952f, -0.00259f, -0.003004f, 0.00828f, -0.000515f, -0.00759f,
      -0.002186f, -0.00375f, -0.00902f, 0.002289f, -0.002497f, 0.00996f, 0.004932f, -0.00803f, -0.00785f,
      0.00993f, -0.007694f, 0.000255f, -0.0002395f, -0.005318f, 0.005173f, 0.00518f, -0.007427f, 0.00505f,
      0.008545f, -0.00238f, -0.002556f, 0.00932f, 0.009094f, -0.002436f, -0.00971f, 0.000679f, 0.00931f,
      -0.00531f, 0.003595f, 0.0065f, -0.001422f, 0.002657f, 0.00864f, 0.001987f, -0.001189f, -0.0007544f,
      0.0002537f, -0.003994f, -0.00898f, -0.00314f, -0.00829f, 0.006683f, -0.006706f, -0.005634f, 0.00001407f,
      0.006878f, 0.004093f, 0.001739f, -0.003754f, 0.006306f, -0.001363f, -0.00145f, -0.00985f, -0.003508f,
      -0.007454f, 0.00352f, -0.004467f, -0.00601f, -0.007763f, -0.00894f, 0.00583f, -0.00698f, 0.0099f,
      -0.006313f, 0.00404f, -0.002666f, -0.00373f, 0.004604f, -0.00813f, -0.006283f, 0.004066f, -0.00592f,
      -0.0003827f, -0.002565f, 0.006275f, 0.008705f, -0.007404f, 0.00793f, -0.0009556f, 0.001682f, 0.00866f,
      0.00774f, 0.00332f, 0.0008507f, -0.005215f, -0.00757f, -0.001497f, 0.005787f, 0.001453f, -0.001265f,
      -0.00909f, 0.006832f, 0.00836f, 0.002867f, 0.002851f, 0.002344f, 0.001552f, -0.0006785f, -0.00941f,
      -0.007114f, -0.003008f, 0.002539f, 0.0002484f, -0.00774f, 0.000987f, 0.00991f, 0.00611f, 0.0009437f,
      -0.001054f, 0.000739f, 0.00809f, -0.003117f, -0.007812f, -0.001368f, -0.009674f, -0.001733f, 0.006268f,
      0.003513f, 0.00852f, -0.007652f, 0.004547f, -0.0001137f, 0.003424f, 0.000804f, -0.003584f, -0.00599f,
      -0.005333f, -0.00303f, 0.004303f, 0.009f, -0.0006638f, -0.0008726f, 0.007774f, -0.0000234f, -0.0002577f,
      0.005783f, -0.008316f, -0.00841f, -0.003605f, 0.001991f, 0.006767f, 0.00508f, 0.00787f, 0.003464f,
      0.00908f, 0.007133f, 0.007504f, -0.00896f, 0.000183f, -0.00929f, -0.0009255f, -0.0034f, -0.00848f,
      0.002066f, 0.0002947f, 0.005394f, 0.002613f, 0.00701f, -0.00833f, -0.001219f, 0.004704f, 0.00446f,
      -0.00775f, 0.00476f, -0.007195f, -0.00163f, -0.003307f, -0.007484f, -0.00889f, -0.00846f, 0.008156f,
      -0.002731f, 0.005733f, 0.0099f, -0.00276f, -0.00869f, -0.00962f, -0.00841f, -0.004955f, 0.004997f,
      0.008896f, 0.00907f, -0.000695f, 0.00972f, 0.00685f, 0.004505f, -0.00726f, -0.003025f, -0.002087f,
      0.00797f, 0.006016f, -0.006485f, -0.00491f, 0.001922f, -0.00934f, 0.006355f, -0.0004008f, -0.005714f,
      0.002274f, -0.005512f, 0.005424f, -0.0003483f, 0.001698f, 0.0006733f, 0.00815f, -0.005264f, 0.002876f,
      -0.0000476f, -0.003105f, -0.001815f, -0.00997f, 0.0004442f, -0.00557f, -0.007656f, -0.003036f, 0.002333f,
      -0.001329f, 0.003675f, -0.00706f, -0.00807f, 0.001302f, -0.00788f, 0.003828f, -0.00995f, -0.006676f,
      -0.001514f, -0.005756f, -0.001301f, 0.002438f, 0.007313f, 0.00913f, 0.003407f, -0.002222f, 0.00981f,
      0.0012245f, 0.009155f, 0.008194f, -0.004368f, -0.006615f, -0.0008593f, -0.00582f, 0.003933f, 0.005173f,
      -0.001201f, 0.002068f, -0.00915f, 0.00797f, -0.002686f, -0.00958f, 0.005775f, 0.002453f, -0.003305f,
      0.00697f, 0.0001255f, 0.00218f, 0.009926f, -0.007473f, 0.007965f, 0.0066f, -0.003874f, 0.00658f,
      -0.007618f, 0.000942f, 0.002375f, -0.007053f, -0.003815f, 0.00569f, -0.001039f, 0.004536f, 0.003641f,
      0.004314f, -0.003353f, 0.00857f, -0.0006385f, -0.000856f, -0.007175f, 0.007557f, -0.00978f, 0.002863f,
      -0.005424f, 0.005215f, -0.000666f, -0.006275f, 0.005527f, 0.00827f, -0.006187f, -0.005993f, 0.000444f,
      -0.0001373f, 0.00458f, 0.009315f, -0.005093f, -0.00154f, 0.002647f, 0.00586f, 0.007473f, -0.00275f,
      0.00046f, 0.008965f, -0.0002766f, 0.00485f, -0.00974f, 0.001143f, -0.00859f, -0.00027f, 0.007748f,
      -0.00341f, -0.006992f, -0.006664f, 0.0005536f, 0.00828f, -0.003752f, 0.000553f, 0.008575f, 0.004868f,
      -0.0004208f, -0.001359f, 0.002785f, 0.00247f, 0.0002398f, 0.00441f, -0.007866f, -0.00444f, 0.000598f,
      0.00985f, 0.0041f, 0.001188f, -0.00271f, -0.003817f, -0.0008373f, -0.004078f, 0.00927f, -0.002739f,
      -0.004578f, 0.004482f, 0.000669f, -0.003761f, -0.00921f, -0.003477f, -0.00516f, -0.00893f, 0.0007854f,
      0.00305f, 0.004894f, 0.00165f, -0.009834f, -0.00859f, 0.000812f, -0.007256f, -0.00276f, -0.003006f,
      0.001255f, -0.002705f, 0.005894f, 0.00904f, 0.004845f, 0.00814f, -0.003206f, 0.007042f, -0.003756f,
      -0.003365f, -0.00868f, 0.00358f, -0.009514f, 0.00952f, -0.005753f, 0.00848f, 0.003448f, 0.006912f,
      -0.001069f, -0.0006742f, 0.00974f, -0.001088f, -0.0004857f, 0.00841f, 0.006027f, -0.00606f, -0.001904f,
      -0.006058f, -0.004673f, 0.007572f, -0.009674f, -0.008896f, -0.002888f, -0.00806f, 0.00633f, -0.000787f,
      -0.002151f, 0.002234f, -0.00991f, 0.00663f, -0.00541f, -0.006706f, -0.00598f, -0.00592f, 0.0001597f,
      0.001887f, -0.00104f, 0.00994f, 0.0083f, -0.009415f, -0.00954f, 0.0003498f, -0.009254f, 0.002195f,
      0.003555f, -0.007557f, 0.006336f, -0.00789f, -0.006927f, 0.005497f, -0.003809f, -0.002302f, -0.00952f,
      -0.0007987f, -0.001707f, 0.00007784f, -0.006718f, -0.005337f, 0.008934f, 0.006355f, 0.006626f, 0.00514f,
      0.006844f, -0.005447f, -0.001604f, -0.0008254f, -0.004185f, -0.006702f, -0.001056f, -0.00847f, -0.005917f,
      -0.002684f, -0.00482f, -0.009514f, 0.004032f, 0.003906f, 0.0048f, -0.004612f, 0.000876f, -0.00497f,
      0.008415f, -0.00986f, -0.00565f, -0.000717f, -0.003967f, -0.006863f, 0.00825f, -0.003292f, -0.00966f,
      0.00263f, 0.001377f, -0.0084f, 0.004414f, -0.0054f, 0.00609f, -0.009026f, -0.000778f, -0.008385f,
      0.008286f, -0.00352f, 0.00549f, 0.00738f, -0.007515f, -0.002409f, -0.00558f, -0.003153f, -0.005985f,
      -0.00919f, 0.00001955f, 0.004105f, -0.0009418f, 0.001782f, 0.0007043f, -0.00539f, -0.004562f, -0.003515f,
      -0.00916f, -0.00623f, 0.0002017f, -0.003117f, 0.00392f, 0.00738f, 0.001152f, -0.00806f, -0.005108f,
      0.00985f, -0.001203f, 0.00719f, 0.001182f, -0.0002191f, -0.00661f, -0.003593f, -0.001818f, 0.00765f,
      0.004604f, -0.005318f, -0.0009274f, 0.002466f, -0.0003357f, 0.00783f, -0.006584f, -0.00664f, 0.003544f,
      -0.002964f, -0.00983f, 0.001785f, -0.000708f, -0.00793f, 0.00785f, 0.006046f, 0.007812f, 0.0096f,
      0.00849f, -0.001343f, 0.00623f, -0.007465f, 0.001237f, -0.00393f, -0.0007534f, -0.004776f, -0.002806f,
      0.00451f, -0.004726f, 0.00364f, 0.002312f, -0.00561f, -0.00462f, -0.001799f, -0.0005593f, 0.00191f,
      -0.002151f, -0.0076f, 0.001353f, 0.001949f, -0.004097f, 0.005615f, 0.002104f, 0.00746f, -0.00824f,
      -0.006596f, 0.009285f, -0.008026f, 0.00331f, -0.008736f, -0.00988f, -0.002468f, 0.003393f, -0.007675f,
      -0.00852f, 0.0067f, 0.00552f, 0.00002897f, 0.0002024f, -0.004135f, 0.003683f, -0.001939f, -0.002998f,
      -0.006897f, -0.00462f, 0.00989f, 0.001207f, 0.001254f, -0.0008793f, -0.004036f, -0.00255f, 0.00871f,
      0.00695f, 0.00251f, 0.005455f, -0.00592f, -0.001793f, -0.0005703f, -0.00213f, 0.004787f, -0.0025f,
      -0.00712f, -0.003109f, -0.0074f, 0.003607f, -0.003696f, -0.001566f, 0.007812f, -0.004433f, 0.001471f,
      0.004066f, -0.001959f, -0.001853f, -0.00985f, 0.006023f, 0.006184f, -0.00586f, -0.002455f, 0.007687f,
      -0.003036f, -0.001865f, 0.0052f, -0.005646f, 0.002298f, -0.0049f, -0.001856f, -0.003754f, -0.003891f,
      0.00979f, 0.008415f, -0.00886f, 0.009926f, 0.001531f, -0.001119f, -0.004818f, 0.007763f, -0.004997f,
      0.009415f, 0.002409f, 0.00149f, 0.003786f, -0.001091f, -0.00852f, 0.00888f, 0.0092f, 0.004227f,
      0.004055f, -0.001675f, -0.004677f, 0.003109f, 0.006733f, 0.00538f, 0.0086f, 0.002913f, -0.00939f,
      -0.006355f, 0.00495f, -0.007866f, 0.00885f, 0.005394f, -0.00323f, 0.00578f, -0.00476f, 0.006634f,
      -0.00769f, 0.001916f, -0.001957f, 0.00988f, 0.004417f, -0.00677f, 0.007565f, 0.00842f, -0.00919f,
      -0.0055f, 0.003214f, 0.00413f, -0.00813f, 0.002834f, 0.005272f, -0.00954f, 0.006275f, -0.00836f,
      0.00561f, 0.00951f, 0.004837f, 0.00753f, 0.000762f, -0.002527f, -0.003277f, -0.00522f, 0.003021f,
      0.00706f, -0.008f, -0.00916f, -0.002863f, 0.002209f, -0.00828f, 0.00499f, -0.001951f, -0.002157f,
      0.004375f, 0.006233f, -0.007336f, -0.0002134f, 0.004395f, -0.004135f, -0.00865f, 0.001095f, 0.003302f,
      -0.00732f, 0.002275f, 0.00976f, 0.002602f, -0.003263f, 0.00766f, 0.003126f, 0.001476f, -0.001589f,
      0.00351f, 0.007305f, 0.00553f, 0.007236f, -0.005352f, -0.006542f, -0.002747f, -0.002932f, -0.002441f,
      -0.008575f, -0.00934f, -0.00197f, -0.004387f, 0.001285f, 0.003265f, 0.001039f, 0.004814f, -0.001674f,
      -0.00887f, 0.003067f, -0.007866f, 0.00903f, 0.003162f, -0.004402f, 0.00029f, 0.00928f, -0.002539f,
      -0.003176f, 0.002398f, 0.004284f, 0.001891f, -0.000756f, 0.00846f, 0.00686f, 0.001065f, -0.008934f,
      -0.00705f, 0.002884f, -0.006603f, -0.004486f, 0.00396f, -0.009766f, -0.003494f, 0.004738f, 0.00899f,
      0.006016f, 0.007515f, 0.003511f, -0.00786f, 0.00949f, -0.00682f, 0.004265f, 0.00728f, 0.0047f,
      0.00902f, -0.00474f, -0.0005236f, 0.005547f, -0.002396f, -0.006386f, -0.007904f, 0.00722f, 0.005135f,
      0.000564f, -0.003956f, -0.00997f, -0.00982f, 0.001334f, 0.001509f, -0.002422f, -0.001891f, 0.002316f,
      0.00309f, -0.006355f, 0.007336f, -0.00487f, 0.00010824f, -0.0008583f, 0.002853f, 0.003754f, -0.006348f,
      0.00793f, 0.00723f, -0.00981f, -0.003706f, 0.00317f, -0.008446f, -0.002966f, -0.0009055f, 0.002184f,
      0.003096f, 0.003244f, 0.009674f, 0.002132f, 0.0016165f, -0.006443f, -0.00423f, -0.00905f, 0.001218f,
      0.004185f, 0.00935f, -0.00193f, 0.00179f, 0.004192f, -0.006424f, 0.002945f, 0.0005383f, 0.004173f,
      -0.001795f, 0.00803f, 0.006462f, -0.00502f, -0.003693f, 0.001283f, -0.001253f, 0.00715f, -0.002525f,
      0.00824f, -0.008995f, -0.00549f, 0.004345f, 0.002205f, 0.00827f, -0.004692f, -0.000714f, 0.00686f,
      0.003473f, 0.009636f, -0.001164f, -0.002003f, 0.00674f, -0.008224f, -0.00462f, 0.00948f, 0.002377f,
      0.00781f, 0.002586f, 0.00744f, -0.001399f, 0.003376f, 0.005226f, -0.003313f, 0.007713f, -0.004364f,
      0.0005984f, -0.004997f, 0.00611f, -0.00772f, 0.006653f, -0.002066f, 0.00196f, 0.004326f, 0.00797f,
      -0.002724f, -0.005474f, 0.007782f, 0.00728f, 0.007442f, -0.002098f, 0.005306f, -0.007206f, -0.001974f,
      0.0000934f, -0.003695f, -0.007633f, 0.006306f, 0.006794f, -0.002983f, -0.00424f, 0.0018215f, 0.000337f,
      -0.00849f, -0.00768f, 0.00659f, 0.002615f, -0.008514f, 0.00282f, 0.003607f, 0.009544f, 0.00924f,
      0.00949f, -0.006145f, -0.003231f, -0.001794f, 0.006004f, -0.0005646f, 0.005558f, 0.00455f, -0.005344f,
      0.003881f, -0.00979f, -0.00946f, -0.0007844f, 0.00922f, 0.001785f, 0.00854f, -0.0094f, -0.005318f,
      0.006126f, -0.0023f, -0.00576f, -0.00449f, -0.00931f, 0.006935f, -0.007477f, 0.001311f, 0.00797f,
      0.003727f, -0.000941f, -0.00816f, -0.00646f, -0.004032f, -0.002666f, 0.009735f, -0.007072f, -0.007362f,
      0.003067f, 0.007732f, 0.00457f, 0.001084f, -0.0085f, 0.00392f, 0.0006833f, -0.001245f, -0.00907f,
      -0.00574f, -0.006786f, 0.005386f, -0.001034f, 0.00993f, 0.00913f, -0.001817f, 0.00613f, 0.002943f,
      -0.00825f, -0.008804f, -0.00333f, -0.00754f, 0.00971f, -0.0002515f, 0.004715f, 0.006126f, 0.004963f,
      0.000591f, -0.00912f, -0.002254f, 0.0006866f, -0.00998f, 0.001433f, 0.00787f, -0.00933f, -0.004326f,
      0.00771f, 0.002146f, -0.006893f, -0.003952f, 0.001425f, -0.006123f, 0.00807f, -0.00702f, -0.006565f,
      0.001073f, 0.001927f, -0.004864f, 0.000273f, -0.008224f, 0.00826f, -0.001634f, -0.006905f, -0.00831f,
      -0.00594f, -0.002901f, -0.001668f, -0.00987f, 0.006264f, -0.00452f, -0.00924f, 0.0096f, 0.001883f,
      0.005104f, 0.003798f, -0.00859f, 0.002163f, 0.000841f, 0.0001701f, -0.00549f, 0.008896f, -0.00641f,
      -0.0086f, 0.0094f, -0.000762f, 0.000456f, 0.002989f, -0.002628f, -0.00817f, -0.000566f, 0.005928f,
      -0.002151f, -0.004353f, -0.00403f, -0.0009055f, 0.00814f, -0.005325f, 0.001588f, -0.00841f, 0.001743f,
      -0.00651f, -0.002144f, 0.007225f, -0.00623f, -0.002226f, -0.004345f, 0.007904f, -0.007748f, 0.001748f,
      -0.003706f, -0.00867f, 0.00432f, -0.00954f, 0.0089f, -0.00607f, 0.00603f, 0.00857f, 0.003477f,
      -0.0007524f, 0.000207f, -0.00069f, 0.00925f, -0.003777f, -0.0002985f, -0.001528f, 0.005077f, 0.007435f,
      0.005886f, -0.001046f, 0.00491f, -0.00346f, -0.00944f, 0.0085f, 0.00011885f, -0.007687f, 0.005142f,
      -0.005444f, 0.005745f, 0.00565f, -0.005436f, 0.002954f, 0.0009327f, -0.001357f, -0.006035f, -0.0038f,
      -0.00277f, 0.001201f, -0.006207f, 0.00892f, -0.00958f, 0.002432f, 0.009636f, -0.006413f, -0.000683f,
      0.000565f, 0.00664f, 0.006424f, 0.004097f, 0.00754f, -0.0082f, 0.002491f, 0.00003463f, -0.001084f,
      0.009895f, -0.001157f, -0.0044f, -0.003542f, -0.005615f, 0.00814f, -0.002285f, 0.009605f, 0.008865f,
      0.00906f, 0.0059f, -0.00735f, 0.0007353f, -0.00103f, -0.004868f, 0.007378f, 0.0074f, -0.001978f,
      -0.00555f, -0.004807f, 0.006527f, -0.00968f, -0.001172f, -0.00988f, 0.00564f, 0.00213f, 0.004536f,
      -0.001937f, 0.007717f, 0.00901f, -0.000779f, 0.003677f, -0.00831f, -0.005554f, -0.005386f, -0.00959f,
      -0.00885f, 0.007416f, -0.00618f, 0.001828f, -0.0004594f, -0.0006585f, -0.009636f, 0.007168f, -0.00868f,
      -0.00848f, -0.003803f, -0.00875f, 0.002884f, 0.0002168f, 0.005486f, 0.00989f, -0.00828f, 0.00000566f,
      -0.00811f, -0.003649f, 0.003096f, 0.00365f, -0.002344f, -0.00879f, 0.006554f, -0.0003917f, 0.00814f,
      -0.001268f, 0.00318f, 0.003078f, -0.002525f, -0.00848f, -0.0004594f, 0.003298f, 0.003225f, 0.002396f,
      -0.00686f, -0.00503f, 0.007534f, 0.009636f, -0.00483f, -0.00788f, 0.004208f, 0.0003386f, -0.001907f,
      0.0008726f, 0.004757f, -0.00989f, -0.007004f, 0.0063f, -0.006622f, -0.00978f, 0.00899f, 0.002703f,
      0.00864f, -0.009964f, 0.00617f, 0.005688f, 0.00846f, 0.00576f, 0.00788f, 0.0002687f, 0.00853f,
      -0.0002925f, -0.003065f, -0.0000076f, 0.007706f, 0.002523f, -0.00212f, -0.00532f, 0.007347f, 0.001383f,
      -0.004616f, -0.008514f, -0.00672f, -0.00883f, 0.00195f, -0.003576f, -0.006306f, 0.005207f, -0.002554f,
      -0.001393f, -0.005966f, 0.005707f, -0.001915f, -0.002625f, 0.007797f, 0.00756f, -0.003504f, -0.004597f,
      -0.002932f, -0.006004f, -0.00928f, 0.006176f, 0.004486f, -0.00594f, -0.009476f, 0.006813f, -0.00312f,
      -0.0014715f, 0.003428f, 0.00991f, -0.004757f, -0.0006704f, 0.001299f, 0.002937f, 0.005505f, 0.00843f,
      -0.004585f, -0.00931f, 0.001348f, -0.008545f, 0.001818f, -0.002092f, -0.00689f, -0.009026f, 0.00949f,
      0.00166f, 0.000547f, -0.000135f, -0.000778f, -0.001905f, 0.002375f, 0.00974f, -0.004833f, 0.0094f,
      0.004898f, -0.00005084f, -0.001083f, -0.00499f, -0.00918f, -0.004326f, 0.001663f, 0.00681f, -0.003672f,
      0.00694f, -0.00438f, -0.007336f, 0.0089f, 0.00451f, -0.00564f, 0.00986f, 0.006157f, -0.00539f,
      -0.00551f, 0.00947f, 0.00881f, 0.005436f, -0.008354f, -0.005894f, 0.002949f, 0.0009093f, -0.002594f,
      -0.002369f, 0.00507f, -0.0088f, 0.0051f, -0.0004027f, 0.001238f, 0.00854f, 0.008804f, 0.0005126f,
      0.00786f, -0.001762f, -0.002861f, 0.001445f, -0.006268f, -0.002352f, -0.00737f, -0.006973f, 0.005512f,
      0.005188f, 0.00951f, -0.006603f, 0.002338f, -0.001549f, 0.000984f, 0.00819f, 0.002796f, -0.003716f,
      -0.00731f, -0.004124f, -0.00725f, -0.002102f, 0.00493f, 0.00313f, -0.002922f, 0.0076f, 0.00537f,
      -0.00929f, 0.00819f, 0.00932f, 0.00975f, 0.00345f, 0.001942f, 0.001167f, -0.003649f, -0.00787f,
      0.00857f, 0.00359f, 0.0015545f, -0.001327f, -0.00813f, 0.006893f, -0.00185f, -0.00689f, 0.00396f,
      0.003069f, -0.002464f, -0.003843f, 0.004967f, -0.00865f, -0.00503f, 0.003744f, 0.0003045f, 0.006298f,
      0.0011835f, 0.004654f, -0.00736f, -0.00171f, -0.00807f, -0.00462f, 0.00526f, 0.00905f, -0.006798f,
      -0.0001366f, 0.00969f, -0.005116f, 0.007614f, -0.007317f, -0.0052f, 0.0007396f, 0.00735f, -0.00347f,
      -0.002716f, 0.005177f, 0.003021f, -0.0026f, 0.00685f, -0.003214f, 0.001522f, -0.000601f, 0.00642f,
      0.002537f, 0.009705f, 0.0004787f, 0.00933f, 0.005848f, -0.00789f, -0.005962f, -0.003063f, 0.00734f,
      0.008644f, -0.00652f, 0.00389f, 0.00219f, -0.005104f, 0.004536f, 0.006638f, -0.00424f, -0.000966f,
      -0.00242f, -0.003347f, 0.000761f, -0.006855f, -0.00816f, -0.00339f, 0.003853f, 0.00752f, 0.000502f,
      0.00394f, 0.00875f, -0.001621f, -0.00972f, -0.000609f, -0.00796f, -0.003817f, 0.004166f, 0.003754f,
      -0.007385f, -0.001137f, -0.004467f, -0.001389f, 0.0093f, 0.003342f, -0.005795f, -0.00792f, 0.0082f,
      0.00557f, -0.00656f, 0.003494f, 0.002573f, 0.0014925f, -0.003141f, 0.002457f, 0.00789f, 0.0071f,
      -0.004307f, 0.001407f, 0.000862f, -0.007122f, -0.005196f, -0.00306f, -0.00808f, -0.004246f, 0.00772f,
      0.006165f, 0.002718f, -0.00569f, -0.000952f, -0.005917f, 0.003725f, -0.0008345f, -0.00265f, -0.0063f,
      0.001651f, -0.00962f, 0.006016f, 0.005035f, -0.004337f, 0.00552f, 0.00373f, -0.0005794f, 0.00202f,
      -0.006985f, -0.00747f, -0.001536f, -0.007122f, -0.00937f, -0.00641f, -0.00871f, -0.00182f, 0.0000921f,
      0.007484f, -0.00974f, 0.00521f, 0.001293f, 0.0006785f, -0.00888f, 0.005943f, -0.00055f, -0.00676f,
      -0.0000759f, 0.00414f, 0.007065f, 0.0000026f, -0.003262f, -0.001492f, 0.00802f, 0.003487f, -0.00977f,
      -0.006863f, -0.004192f, -0.007458f, -0.001814f, -0.004482f, 0.008835f, -0.004826f, 0.00872f, 0.004635f,
      0.007317f, -0.00498f, -0.003536f, -0.004375f, 0.005074f, -0.002346f, 0.00384f, 0.00853f, -0.00416f,
      -0.007164f, 0.0006695f, 0.0008926f, -0.001899f, 0.005783f, 0.00535f, 0.00557f, -0.00402f, 0.00006354f,
      -0.001951f, -0.002588f, -0.005276f, -0.001826f, -0.006058f, 0.001427f, -0.009735f, 0.009224f, -0.00006384f,
      -0.002344f, -0.00004303f, 0.00946f, -0.00841f, -0.00199f, -0.00494f, -0.00841f, -0.008835f, 0.00596f,
      -0.006348f, 0.007545f, 0.001068f, 0.00624f, -0.005306f, 0.001778f, -0.0009108f, -0.0048f, -0.000988f,
      -0.0005326f, -0.005173f, 0.003748f, 0.001759f, -0.003914f, -0.006252f, 0.004486f, 0.00882f, 0.006035f,
      -0.002064f, -0.003456f, -0.006615f, -0.004963f, 0.003847f, -0.00342f, 0.006115f, -0.005974f, 0.002302f,
      -0.00856f, 0.006847f, -0.006416f, -0.00226f, 0.005363f, 0.008224f, -0.0003793f, -0.009224f, -0.002298f,
      -0.005264f, -0.000623f, -0.00803f, -0.007706f, 0.001601f, 0.007046f, -0.004757f, 0.0044f, 0.0046f,
      -0.003963f, -0.007156f, 0.0004344f, 0.005592f, -0.00053f, 0.001337f, 0.009186f, -0.00897f, -0.005627f,
      -0.001647f, 0.0092f, 0.0016985f, -0.003633f, 0.008064f, 0.004543f, -0.00698f, -0.005695f, 0.00478f,
      -0.001252f, 0.00881f, -0.00876f, -0.00202f, -0.009514f, 0.000278f, -0.005013f, 0.007404f, -0.0005183f,
      -0.001753f, -0.00442f, 0.00199f, -0.008156f, -0.008865f, -0.00308f, -0.00973f, -0.005714f, 0.007996f,
      -0.004395f, 0.00455f, -0.00862f, -0.0004373f, 0.00885f, 0.00984f, -0.00422f, 0.00382f, 0.001032f,
      -0.0003273f, 0.004593f, 0.004982f, 0.00259f, -0.00604f, 0.000337f, 0.009186f, -0.003052f, -0.005085f,
      0.005188f, 0.00417f, 0.004345f, 0.003605f, -0.000079f, -0.009575f, 0.00894f, 0.00992f, 0.008f,
      -0.00476f, 0.00871f, -0.007538f, -0.00739f, -0.0069f, -0.008804f, -0.00526f, -0.001096f, 0.0009003f,
      0.005367f, 0.005283f, 0.005047f, -0.0003638f, -0.001063f, -0.00399f, 0.0081f, 0.004395f, 0.00805f,
      -0.00531f, 0.001779f, 0.003176f, 0.00775f, 0.0071f, 0.00682f, -0.0007925f, -0.00318f, 0.00897f,
      -0.006172f, -0.00376f, -0.002518f, -0.007618f, 0.00728f, 0.007042f, 0.006863f, -0.005936f, 0.004787f,
      0.005726f, -0.0009775f, -0.004757f, -0.0002875f, 0.00844f, 0.005302f, 0.003609f, 0.005863f, 0.005436f,
      0.004433f, -0.002047f, 0.003025f, 0.007694f, -0.007565f, -0.006165f, -0.00202f, -0.004505f, -0.004784f,
      0.00921f, -0.00059f, 0.004604f, 0.002249f, -0.004814f, -0.00519f, -0.00625f, 0.0000181f, 0.00531f,
      0.001533f, 0.006847f, -0.00959f, -0.00846f, -0.00928f, -0.006386f, 0.002766f, -0.005516f, -0.0071f,
      0.006073f, 0.00907f, 0.005585f, -0.00644f, -0.00855f, -0.003466f, -0.009514f, -0.00914f, 0.003702f,
      -0.00503f, -0.00497f, 0.00796f, -0.007763f, 0.007614f, 0.00544f, 0.00933f, 0.008316f, -0.003374f,
      -0.00763f, 0.002035f, 0.002916f, -0.0006156f, -0.003872f, -0.0002236f, -0.00917f, -0.003334f, -0.004528f,
      0.00978f, -0.0005903f, -0.006786f, -0.00913f, -0.009254f, -0.006096f, 0.002638f, 0.003622f, -0.007805f,
      0.00873f, 0.001586f, -0.003641f, 0.001905f, -0.00311f, -0.000627f, 0.005222f, -0.004986f, 0.000169f,
      -0.007088f, -0.00783f, -0.004852f, 0.000881f, 0.004627f, -0.00405f, -0.006405f, 0.003586f, 0.002258f,
      -0.00988f, 0.000979f, -0.002949f, 0.00912f, 0.00885f, -0.002743f, 0.00833f, 0.003326f, -0.0003536f,
      -0.003792f, -0.00941f, 0.000213f, -0.002922f, -0.001483f, -0.003443f, -0.00307f, -0.005894f, 0.003468f,
      0.001887f, -0.006832f, -0.00828f, -0.006172f, -0.00746f, 0.002558f, 0.00998f, 0.001123f, -0.00611f,
      -0.005863f, -0.0007744f, 0.003525f, -0.00573f, 0.0009665f, -0.002241f, -0.0007176f, -0.00918f, -0.00794f,
      0.00216f, -0.0049f, 0.002016f, 0.006763f, 0.00445f, 0.004715f, 0.001216f, 0.002068f, -0.001449f,
      0.00249f, 0.00953f, -0.0007606f, -0.00256f, 0.0006046f, -0.004406f, -0.009415f, 0.003393f, -0.004787f,
      0.002743f, 0.00841f, 0.00972f, -0.00194f, 0.004185f, 0.00585f, 0.007504f, -0.00622f, 0.001107f,
      -0.0044f, 0.00576f, 0.00772f, 0.00818f, 0.00536f, 0.002644f, -0.00465f, -0.0087f, -0.00816f,
      0.004547f, 0.001851f, -0.005634f, 0.003641f, 0.007618f, -0.00985f, 0.009766f, -0.00459f, -0.002457f,
      0.00393f, -0.008224f, -0.003952f, -0.00813f, 0.007393f, 0.005188f, 0.007126f, 0.00639f, 0.001274f,
      0.002176f, -0.00894f, 0.002445f, -0.001414f, -0.00952f, 0.004444f, -0.001607f, -0.001501f, 0.00857f,
      -0.005585f, -0.000724f, 0.003077f, 0.007797f, 0.007473f, 0.003546f, -0.00948f, -0.003933f, 0.004017f,
      -0.003176f, 0.001448f, 0.002731f, 0.003504f, 0.00831f, 0.007763f, 0.002405f, -0.006264f, 0.00536f,
      -0.0083f, 0.001413f, -0.0003624f, -0.001836f, 0.006027f, 0.005173f, -0.003073f, -0.008354f, 0.00164f,
      -0.001941f, -0.002981f, 0.008156f, -0.004414f, -0.005413f, 0.002527f, -0.0004022f, 0.00625f, 0.008575f,
      0.00637f, 0.00765f, 0.0003421f, 0.00798f, -0.005287f, 0.00808f, -0.00646f, 0.000603f, 0.00955f,
      0.00889f, -0.002356f, -0.005306f, 0.002333f, 0.009514f, -0.003855f, 0.0054f, 0.005417f, 0.000675f,
      -0.004402f, 0.00933f, -0.005234f, -0.00958f, 0.0089f, 0.009254f, -0.00757f, 0.0098f, -0.001879f,
      0.00789f, 0.002071f, 0.000677f, -0.007763f, -0.001941f, 0.001637f, -0.003653f, 0.00528f, 0.007465f,
      -0.00557f, -0.006004f, -0.009476f, 0.000802f, 0.002075f, -0.007168f, 0.00398f, -0.006268f, 0.006287f,
      -0.009575f, -0.001453f, 0.0092f, -0.00995f, -0.002644f, 0.005024f, 0.00966f, -0.006878f, 0.00995f,
      -0.001319f, -0.002237f, 0.002209f, 0.00861f, -0.00883f, -0.003874f, -0.002903f, 0.00992f, -0.0016365f,
      -0.00633f, 0.00823f, -0.00771f, -0.003204f, -0.00563f, 0.00563f, 0.00805f, -0.004936f, 0.003477f,
      0.00741f, 0.0043f, 0.006905f};

  bias_data = {
      -0.003569f, -0.00789f, 0.002047f, -0.002829f, -0.000592f, -0.003313f, 0.00805f, -0.007397f, -0.006844f,
      0.00809f, -0.003479f, -0.0017395f, 0.007904f, -0.009056f, 0.005806f, 0.008896f, 0.004585f, -0.002356f,
      -0.003815f, -0.00673f, 0.005787f, -0.001892f, 0.003233f, 0.005566f, -0.007626f, 0.00835f, 0.009415f,
      -0.005707f, -0.0002623f, -0.007496f, -0.003569f, -0.00568f, -0.000693f, 0.00857f, 0.006607f, 0.005245f,
      -0.0006056f, 0.008896f, 0.0000753f, -0.0001878f, -0.00957f, -0.003975f, 0.003006f, -0.006794f, -0.007935f,
      0.004246f, 0.004948f, 0.008896f, -0.0046f, -0.002516f, -0.000887f, -0.004555f, 0.002409f, 0.00364f,
      -0.002491f, 0.004204f, 0.00010544f, 0.000783f, 0.00895f, 0.005367f, -0.004097f, -0.00592f, 0.009834f,
      0.001047f, 0.00677f, -0.004974f, -0.003212f, 0.00771f, -0.002256f, -0.001008f, -0.008484f, -0.002802f,
      0.00462f, 0.001329f, 0.004562f, 0.006687f, 0.002615f, 0.001449f, -0.0006714f, -0.001256f, 0.0003803f,
      -0.005238f, -0.004112f, 0.001925f, -0.002827f, -0.00861f, -0.004723f, -0.002748f, -0.006134f, -0.00342f,
      -0.007168f, 0.006626f, 0.001948f, -0.003838f, 0.006878f, -0.001717f, -0.003347f, -0.006287f, 0.00455f,
      -0.00136f, 0.004364f, 0.006573f, -0.007545f, -0.004845f, 0.00883f, 0.00572f, 0.00675f, -0.003206f,
      -0.00842f, 0.006428f, 0.00394f, 0.000642f, -0.002016f, 0.004486f, 0.009964f, -0.00918f, -0.0084f,
      0.001972f, 0.002031f, -0.00976f, -0.004494f, 0.006958f, -0.00262f, 0.00874f, 0.009865f, 0.0075f,
      -0.00271f, -0.006386f, 0.002562f, 0.006397f, 0.00699f, -0.001731f, 0.005432f, 0.00271f, -0.006447f,
      -0.00892f, 0.002897f, -0.0004315f, 0.001859f, -0.003462f, 0.007122f, -0.005135f, 0.005363f, 0.0051f,
      0.00806f, 0.00721f, 0.00799f, 0.00945f, -0.006943f, 0.006393f, 0.00935f, -0.0003269f, -0.004536f,
      -0.006752f, 0.0095f, 0.00628f, -0.00418f, 0.001624f, -0.005577f, -0.008606f, 0.005486f, 0.002077f,
      0.007378f, 0.004734f, 0.0035f, 0.00991f, -0.001775f, 0.00247f, -0.00613f, 0.007202f, -0.00596f,
      0.003876f, -0.00789f, 0.004505f, 0.004795f, -0.002575f, -0.002932f, -0.003098f, -0.005463f, -0.00912f,
      -0.00729f, 0.004486f, 0.006138f, 0.006924f, -0.00722f, 0.00841f, -0.001812f, -0.00959f, -0.000497f,
      -0.00513f, -0.006042f, 0.007645f};
}

TEST(AttentionTest, Causal_EmptyPastState) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 64;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.00838f, 0.007523f, -0.00872f, 0.002882f, -0.003567f, 0.000859f, -0.002821f, 0.000563f, 0.007675f, -0.002758f,
      0.000947f, 0.001149f, -0.001534f, 0.0006075f, 0.002853f, 0.004517f, 0.00825f, 0.003687f, -0.002161f, 0.001167f,
      0.005913f, 0.00394f, -0.002136f, 0.00946f, 0.000461f, -0.003593f, -0.002377f, -0.001609f, -0.006363f, 0.0013485f,
      -0.006706f, -0.005188f, 0.002165f, 0.006073f, 0.007717f, -0.007675f, 0.000827f, 0.004253f, 0.00697f, -0.0035f,
      -0.00301f, 0.006565f, -0.0002068f, -0.004593f, 0.00198f, 0.00107f, -0.003082f, 0.002243f, 0.00983f, 0.00608f,
      0.001682f, 0.001701f, -0.006935f, 0.004765f, -0.002333f, 0.003805f, -0.00905f, 0.00599f, 0.00998f, -0.001602f,
      0.00744f, -0.008514f, 0.005424f, -0.002413f, 0.00862f, 0.00459f, -0.002516f, 0.00283f, -0.00272f, -0.005207f,
      -0.00738f, -0.005386f, -0.00951f, 0.008415f, 0.002865f, -0.00726f, 0.00494f, 0.002226f, 0.0000424f, -0.007507f,
      0.002193f, -0.004482f, 0.002386f, 0.005997f, -0.001786f, 0.009f, 0.006435f, -0.0067f, -0.001984f, 0.001514f,
      -0.004917f, 0.003468f, -0.0013685f, -0.007122f, 0.00788f, 0.000825f, 0.00621f, -0.00437f, 0.005653f, 0.009674f,
      0.003576f, 0.00956f, 0.0064f, 0.00283f, -0.00797f, 0.00867f, 0.004536f, -0.00985f, 0.004856f, -0.006878f,
      0.006012f, -0.0042f, -0.00328f, -0.00885f, -0.0079f, 0.004917f, -0.00594f, 0.003452f, -0.006355f, -0.003536f,
      0.0022f, 0.003494f, -0.008865f, 0.00461f, -0.00485f, 0.00889f, -0.002272f, 0.00596f};

  std::vector<float> weight_data;
  std::vector<float> bias_data;
  GetWeightAndBiasForHiddenSize64(weight_data, bias_data);

  // No mask_index
  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      0.0027942657f, 0.0067901611f, 0.0070953369f, -0.0020713806f, 0.0055351257f, 0.0030479431f, -0.0060462952f,
      -0.0087127686f, 0.0030956268f, -0.00036644936f, 0.0014686584f, -0.0038146973f, 0.0072097778f, -0.0052490234f,
      0.0056114197f, 0.0050926208f, 0.0080947876f, 0.0074501038f, 0.0079498291f, 0.0098876953f, -0.0066146851f,
      0.0064735413f, 0.0093307495f, -0.00051593781f, -0.0047683716f, -0.0069198608f, 0.0094604492f, 0.0066146851f,
      -0.0040054321f, 0.0017976761f, -0.0058059692f, -0.0087051392f, 0.0054740906f, 0.0022010803f, 0.0075340271f,
      0.0047035217f, 0.00340271f, 0.0096969604f, -0.0016756058f, 0.0020771027f, -0.0063018799f, 0.0073280334f,
      -0.0056381226f, 0.004032135f, -0.0082473755f, 0.0045280457f, 0.0045814514f, -0.0026607513f, -0.0031585693f,
      -0.003660202f, -0.0053253174f, -0.0089187622f, -0.0073509216f, 0.0048408508f, 0.0058364868f, 0.0069313049f,
      -0.0071868896f, 0.008392334f, -0.0018663406f, -0.0092163086f, -0.00048780441f, -0.0054283142f, -0.0061683655f,
      0.0078048706f, 0.0025291443f, 0.0065917969f, 0.0072250366f, -0.0018520355f, 0.005531311f, 0.003118515f,
      -0.0061264038f, -0.0090484619f, 0.003276825f, -0.00047063828f, 0.0015802383f, -0.0037345886f, 0.0069732666f,
      -0.0054092407f, 0.0052947998f, 0.004940033f, 0.0085220337f, 0.007194519f, 0.0078659058f, 0.0095214844f,
      -0.0065574646f, 0.0064315796f, 0.0093383789f, -0.00058555603f, -0.0046386719f, -0.0067710876f, 0.0096130371f,
      0.0064315796f, -0.0040740967f, 0.0017337799f, -0.0057067871f, -0.008682251f, 0.0054855347f, 0.0019645691f,
      0.0075149536f, 0.0047187805f, 0.0036354065f, 0.0096282959f, -0.0019168854f, 0.0021934509f, -0.0063018799f,
      0.0072937012f, -0.006187439f, 0.0039825439f, -0.0081253052f, 0.0046577454f, 0.0045700073f, -0.0028266907f,
      -0.0028438568f, -0.0035438538f, -0.0053100586f, -0.0090332031f, -0.0071105957f, 0.004699707f, 0.0058021545f,
      0.0071411133f, -0.0071678162f, 0.0085449219f, -0.0018749237f, -0.0095825195f, -0.00049686432f, -0.0053634644f,
      -0.0057945251f, 0.0078277588f};

  std::vector<float> past_data = {};

  std::vector<float> present_data = {
      0.0070152283f, -0.0049858093f, -0.0029277802f, 0.0078277588f, -0.001991272f, -0.0010290146f, -0.0084457397f,
      -0.0028400421f, 0.0048294067f, 0.0012731552f, 0.0047149658f, 0.0069084167f, 0.0027809143f, 0.0014457703f,
      -0.0010128021f, -0.0011024475f, 8.4400177e-05f, -0.0049972534f, -0.0040206909f, 0.002073288f, -0.0034713745f,
      -0.0087203979f, -0.0047302246f, -0.0023326874f, -0.0063209534f, -0.0031681061f, -0.006942749f, 0.0064888f,
      0.0014505386f, -0.0037765503f, 0.0067138672f, -0.0018196106f,
      0.0064506531f, -0.0049514771f, -0.0036487579f, 0.0081558228f, -0.0024414062f, -0.0014820099f, -0.0086212158f,
      -0.0025672913f, 0.0047111511f, 0.0011997223f, 0.0042953491f, 0.0067138672f, 0.0028495789f, 0.0015869141f,
      -0.00037360191f, -0.0012044907f, 0.00029373169f, -0.005065918f, -0.0038700104f, 0.0014038086f, -0.0030422211f,
      -0.0084838867f, -0.004863739f, -0.0028686523f, -0.0063362122f, -0.0034809113f, -0.0075874329f, 0.0066947937f,
      0.0019130707f, -0.0036792755f, 0.0070266724f, -0.0016460419f,

      -0.003238678f, -0.0066452026f, 0.0043983459f, -0.0016002655f, 0.0045623779f, 0.0065002441f, -0.0072174072f,
      -0.0050315857f, 0.0087356567f, 0.0061645508f, 0.0069580078f, -0.003320694f, -0.0087814331f, 0.0062255859f,
      0.0035037994f, 0.00064849854f, -0.0018444061f, 0.0043945312f, 0.01008606f, -0.0089874268f, -0.0087585449f,
      0.0020160675f, 0.00207901f, -0.0097732544f, -0.0042991638f, 0.0070266724f, -0.0028743744f, 0.0087051392f,
      0.0099868774f, 0.0076217651f, -0.0027103424f, -0.006439209f,
      -0.0033836365f, -0.0063171387f, 0.0043144226f, -0.001707077f, 0.0044555664f, 0.0069885254f, -0.0072593689f,
      -0.0050468445f, 0.008895874f, 0.0050582886f, 0.0064926147f, -0.0030384064f, -0.0083618164f, 0.0065307617f,
      0.0038928986f, 0.0005645752f, -0.0024528503f, 0.0043983459f, 0.0099029541f, -0.0088043213f, -0.0081558228f,
      0.0021705627f, 0.0018062592f, -0.0094985962f, -0.0045890808f, 0.0068702698f, -0.002532959f, 0.0081863403f,
      0.009765625f, 0.0077362061f, -0.0026664734f, -0.0060920715f,

      0.0027942657f, 0.0067901611f, 0.0070953369f, -0.0020713806f, 0.0055351257f, 0.0030479431f, -0.0060462952f,
      -0.0087127686f, 0.0030956268f, -0.00036644936f, 0.0014686584f, -0.0038146973f, 0.0072097778f, -0.0052490234f,
      0.0056114197f, 0.0050926208f, 0.0080947876f, 0.0074501038f, 0.0079498291f, 0.0098876953f, -0.0066146851f,
      0.0064735413f, 0.0093307495f, -0.00051593781f, -0.0047683716f, -0.0069198608f, 0.0094604492f, 0.0066146851f,
      -0.0040054321f, 0.0017976761f, -0.0058059692f, -0.0087051392f,
      0.0022659302f, 0.0063896179f, 0.0073509216f, -0.0016336441f, 0.0055236816f, 0.0031890869f, -0.0062026978f,
      -0.0093917847f, 0.0034580231f, -0.00057506561f, 0.0016918182f, -0.0036563873f, 0.0067405701f, -0.005569458f,
      0.0049743652f, 0.0047874451f, 0.0089492798f, 0.0069389343f, 0.0077819824f, 0.0091552734f, -0.0065002441f,
      0.0063934326f, 0.0093460083f, -0.00065517426f, -0.0045127869f, -0.0066223145f, 0.009765625f, 0.0062484741f,
      -0.0041465759f, 0.0016708374f, -0.0056037903f, -0.0086669922f,

      0.0054740906f, 0.0022010803f, 0.0075340271f, 0.0047035217f, 0.00340271f, 0.0096969604f, -0.0016756058f,
      0.0020771027f, -0.0063018799f, 0.0073280334f, -0.0056381226f, 0.004032135f, -0.0082473755f, 0.0045280457f,
      0.0045814514f, -0.0026607513f, -0.0031585693f, -0.003660202f, -0.0053253174f, -0.0089187622f, -0.0073509216f,
      0.0048408508f, 0.0058364868f, 0.0069313049f, -0.0071868896f, 0.008392334f, -0.0018663406f, -0.0092163086f,
      -0.00048780441f, -0.0054283142f, -0.0061683655f, 0.0078048706f,
      0.0054931641f, 0.0017261505f, 0.0074958801f, 0.0047340393f, 0.003868103f, 0.0095596313f, -0.0021572113f,
      0.0023078918f, -0.0063018799f, 0.0072631836f, -0.0067367554f, 0.0039329529f, -0.0080032349f, 0.0047874451f,
      0.0045623779f, -0.0029945374f, -0.0025291443f, -0.0034275055f, -0.0052986145f, -0.0091400146f, -0.0068702698f,
      0.0045623779f, 0.0057678223f, 0.0073547363f, -0.0071487427f, 0.0087051392f, -0.0018835068f, -0.0099411011f,
      -0.00050640106f, -0.0052947998f, -0.0054206848f, 0.0078430176f};

  bool is_unidirectional = true;
  bool use_past_state = true;
  int past_sequence_length = 0;
  bool use_float16 = true;

  // Unfused kernel
  {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kEnableFlashAttention, "0"},
            {onnxruntime::contrib::attention::kDisableFusedAttention, "1"}}};
    RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                     batch_size, sequence_length, hidden_size, number_of_heads, use_float16, is_unidirectional,
                     use_past_state, past_sequence_length, &past_data, &present_data);
  }

  // Fused kernel
  {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kEnableFlashAttention, "0"},
            {onnxruntime::contrib::attention::kDisableFusedAttention, "0"}}};
    RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                     batch_size, sequence_length, hidden_size, number_of_heads, use_float16, is_unidirectional,
                     use_past_state, past_sequence_length, &past_data, &present_data);
  }

  // Fused kernel (fall back to regular fmha since head_size <=64 and sequence_length <= 128)
  {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{
            {onnxruntime::contrib::attention::kEnableFlashAttention, "1"},
            {onnxruntime::contrib::attention::kDisableFusedAttention, "0"}}};
    RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                     batch_size, sequence_length, hidden_size, number_of_heads, use_float16, is_unidirectional,
                     use_past_state, past_sequence_length, &past_data, &present_data);
  }
}
#endif

TEST(AttentionTest, AttentionEmptyPastState) {
  RawAttentionEmptyPastState(false);
}

TEST(AttentionTest, AttentionEmptyPastState_SharedPastPresent) {
  RawAttentionEmptyPastState(true);
}

void RawAttentionPastStateBatch1(bool past_present_share_buffer) {
  int batch_size = 1;
  int sequence_length = 1;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      -0.019333266f, -0.21813886f, 0.16212955f, -0.015626367f};

  std::vector<float> weight_data = {
      -0.4738484025001526f,
      -0.2613658607006073f,
      -0.0978037416934967f,
      -0.34988933801651f,
      0.2243240624666214f,
      -0.0429205559194088f,
      0.418695330619812f,
      0.17441125214099884f,
      -0.18825532495975494f,
      0.18357256054878235f,
      -0.5806483626365662f,
      -0.02251487597823143f,

      0.08742205798625946f,
      0.14734269678592682f,
      0.2387014478445053f,
      0.2884027063846588f,
      0.6490834355354309f,
      0.16965825855731964f,
      -0.06346885114908218f,
      0.4073973298072815f,
      -0.03070945478975773f,
      0.4110257923603058f,
      0.07896808534860611f,
      0.16783113777637482f,

      0.0038893644232302904f,
      0.06946629285812378f,
      0.36680519580841064f,
      -0.07261059433221817f,
      -0.14960581064224243f,
      0.020944256335496902f,
      -0.09378612786531448f,
      -0.1336742341518402f,
      0.06061394885182381f,
      0.2205914407968521f,
      -0.03519909828901291f,
      -0.18405692279338837f,

      0.22149960696697235f,
      -0.1884360909461975f,
      -0.014074507169425488f,
      0.4252440333366394f,
      0.24987126886844635f,
      -0.31396418809890747f,
      0.14036843180656433f,
      0.2854192554950714f,
      0.09709841012954712f,
      0.09935075044631958f,
      -0.012154420837759972f,
      0.2575816512107849f};

  std::vector<float> bias_data = {
      0.4803391396999359f,
      -0.5254325866699219f,
      -0.42926454544067383f,
      -0.2059524953365326f,
      -0.12773379683494568f,
      -0.09542735666036606f,
      -0.35286077857017517f,
      -0.07646317780017853f,
      -0.04590314254164696f,
      -0.03752850368618965f,
      -0.013764488510787487f,
      -0.18478283286094666f};

  // No mask_index
  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      0.20141591f, 0.43005896f, 0.35745093f, 0.19957167f};

  std::vector<float> past_data = {
      0.55445826f, 0.10127074f, 0.71770734f, 0.15915526f, 0.13913247f, 0.77447522f, 0.66044068f, 0.27559045f, 0.35731629f, 0.62033528f, 0.24354559f, 0.22859341f,
      0.45075402f, 0.85365993f, 0.097346395f, 0.28859729f, 0.26926181f, 0.65922296f, 0.8177433f, 0.4212271f, 0.34352475f, 0.059609573f, 0.46556228f, 0.7226882f};

  std::vector<float> present_data = {
      0.55445826f, 0.10127074f, 0.71770734f, 0.15915526f, 0.13913247f, 0.77447522f, -0.30182117f, -0.12330482f, 0.66044068f, 0.27559045f, 0.35731629f, 0.62033528f, 0.24354559f, 0.22859341f, -0.36450946f, -0.19483691f,
      0.45075402f, 0.85365993f, 0.097346395f, 0.28859729f, 0.26926181f, 0.65922296f, -0.027254611f, -0.096526355f, 0.8177433f, 0.4212271f, 0.34352475f, 0.059609573f, 0.46556228f, 0.7226882f, -0.025281552f, -0.25482416f};

  bool is_unidirectional = true;
  bool use_past_state = true;
  int past_sequence_length = 3;

  if (!past_present_share_buffer) {
    RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                     batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional,
                     use_past_state, past_sequence_length, &past_data, &present_data);
  } else {
    RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                     batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional,
                     use_past_state, past_sequence_length, &past_data, &present_data,
                     kMaskIndexEnd, 0, past_sequence_length + sequence_length + 4,
                     true, false, true, {}, {}, 0, true);
  }
}

TEST(AttentionTest, AttentionPastStateBatch1) {
  RawAttentionPastStateBatch1(false);
}

TEST(AttentionTest, AttentionPastStateBatch1_SharedPastPresent) {
  RawAttentionPastStateBatch1(true);
}

void RawAttentionPastStateBatch2(bool past_present_share_buffer) {
  int batch_size = 2;
  int sequence_length = 1;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      -0.10902753f, 0.0041178204f, 0.1871525f, -0.20399982f,
      0.027207348f, -0.25321805f, 0.12869114f, 0.023136809f};

  std::vector<float> weight_data = {
      -0.4738484025001526f,
      -0.2613658607006073f,
      -0.0978037416934967f,
      -0.34988933801651f,
      0.2243240624666214f,
      -0.0429205559194088f,
      0.418695330619812f,
      0.17441125214099884f,
      -0.18825532495975494f,
      0.18357256054878235f,
      -0.5806483626365662f,
      -0.02251487597823143f,

      0.08742205798625946f,
      0.14734269678592682f,
      0.2387014478445053f,
      0.2884027063846588f,
      0.6490834355354309f,
      0.16965825855731964f,
      -0.06346885114908218f,
      0.4073973298072815f,
      -0.03070945478975773f,
      0.4110257923603058f,
      0.07896808534860611f,
      0.16783113777637482f,

      0.0038893644232302904f,
      0.06946629285812378f,
      0.36680519580841064f,
      -0.07261059433221817f,
      -0.14960581064224243f,
      0.020944256335496902f,
      -0.09378612786531448f,
      -0.1336742341518402f,
      0.06061394885182381f,
      0.2205914407968521f,
      -0.03519909828901291f,
      -0.18405692279338837f,

      0.22149960696697235f,
      -0.1884360909461975f,
      -0.014074507169425488f,
      0.4252440333366394f,
      0.24987126886844635f,
      -0.31396418809890747f,
      0.14036843180656433f,
      0.2854192554950714f,
      0.09709841012954712f,
      0.09935075044631958f,
      -0.012154420837759972f,
      0.2575816512107849f};

  std::vector<float> bias_data = {
      0.4803391396999359f,
      -0.5254325866699219f,
      -0.42926454544067383f,
      -0.2059524953365326f,
      -0.12773379683494568f,
      -0.09542735666036606f,
      -0.35286077857017517f,
      -0.07646317780017853f,
      -0.04590314254164696f,
      -0.03752850368618965f,
      -0.013764488510787487f,
      -0.18478283286094666f};

  // No mask_index
  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      0.14902574f, 0.62273371f, 0.43022552f, 0.12759127f,
      0.26993567f, 0.23553593f, 0.43190649f, 0.086044826f};

  std::vector<float> past_data = {
      0.42028648f, 0.55855948f, 0.044569403f, 0.76525789f, 0.13962431f, 0.40977913f,
      0.36911047f, 0.83399564f, 0.36905321f, 0.91414654f, 0.17300875f, 0.78793788f,
      0.10279467f, 0.80501258f, 0.089550517f, 0.85371113f, 0.61801594f, 0.91222942f,
      0.88626182f, 0.069776468f, 0.10591964f, 0.84836882f, 0.83520192f, 0.0098680854f,
      0.3113814f, 0.63999802f, 0.28603253f, 0.98899829f, 0.044405211f, 0.95105386f,
      0.81278932f, 0.63969064f, 0.14494057f, 0.11349615f, 0.87086016f, 0.20983537f,
      0.35107401f, 0.90144604f, 0.68950737f, 0.18928574f, 0.18029204f, 0.074517399f,
      0.70763874f, 0.48440042f, 0.58114725f, 0.1048766f, 0.73694098f, 0.17766342f};

  std::vector<float> present_data = {
      0.42028648f, 0.55855948f, 0.044569403f, 0.76525789f, 0.13962431f, 0.40977913f, -0.22849128f, -0.022080801f,
      0.36911047f, 0.83399564f, 0.36905321f, 0.91414654f, 0.17300875f, 0.78793788f, -0.4449589f, -0.17704415f,
      0.10279467f, 0.80501258f, 0.089550517f, 0.85371113f, 0.61801594f, 0.91222942f, -0.2994619f, -0.14412443f,
      0.88626182f, 0.069776468f, 0.10591964f, 0.84836882f, 0.83520192f, 0.0098680854f, -0.33421949f, -0.18547727f,
      0.3113814f, 0.63999802f, 0.28603253f, 0.98899829f, 0.044405211f, 0.95105386f, -0.033968594f, -0.034833729f,
      0.81278932f, 0.63969064f, 0.14494057f, 0.11349615f, 0.87086016f, 0.20983537f, 0.045759238f, -0.26863033f,
      0.35107401f, 0.90144604f, 0.68950737f, 0.18928574f, 0.18029204f, 0.074517399f, -0.033201858f, -0.10592631f,
      0.70763874f, 0.48440042f, 0.58114725f, 0.1048766f, 0.73694098f, 0.17766342f, -0.054369561f, -0.24562015f};

  bool is_unidirectional = true;
  bool use_past_state = true;
  int past_sequence_length = 3;

  if (!past_present_share_buffer) {
    RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                     batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional,
                     use_past_state, past_sequence_length, &past_data, &present_data);
  } else {
    RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                     batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional,
                     use_past_state, past_sequence_length, &past_data, &present_data,
                     kMaskIndexEnd, 0, past_sequence_length + sequence_length,
                     true, false, true, {}, {}, 0, true);
  }
}

TEST(AttentionTest, AttentionPastStateBatch2) {
  RawAttentionPastStateBatch2(false);
}

TEST(AttentionTest, AttentionPastStateBatch2_SharedPastPresent) {
  RawAttentionPastStateBatch2(true);
}

void RawAttentionPastStateBatch2WithPadding(bool past_present_share_buffer) {
  int batch_size = 2;
  int sequence_length = 1;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      -0.10902753f, 0.0041178204f, 0.1871525f, -0.20399982f,
      0.027207348f, -0.25321805f, 0.12869114f, 0.023136809f};

  std::vector<float> weight_data = {
      -0.4738484025001526f,
      -0.2613658607006073f,
      -0.0978037416934967f,
      -0.34988933801651f,
      0.2243240624666214f,
      -0.0429205559194088f,
      0.418695330619812f,
      0.17441125214099884f,
      -0.18825532495975494f,
      0.18357256054878235f,
      -0.5806483626365662f,
      -0.02251487597823143f,

      0.08742205798625946f,
      0.14734269678592682f,
      0.2387014478445053f,
      0.2884027063846588f,
      0.6490834355354309f,
      0.16965825855731964f,
      -0.06346885114908218f,
      0.4073973298072815f,
      -0.03070945478975773f,
      0.4110257923603058f,
      0.07896808534860611f,
      0.16783113777637482f,

      0.0038893644232302904f,
      0.06946629285812378f,
      0.36680519580841064f,
      -0.07261059433221817f,
      -0.14960581064224243f,
      0.020944256335496902f,
      -0.09378612786531448f,
      -0.1336742341518402f,
      0.06061394885182381f,
      0.2205914407968521f,
      -0.03519909828901291f,
      -0.18405692279338837f,

      0.22149960696697235f,
      -0.1884360909461975f,
      -0.014074507169425488f,
      0.4252440333366394f,
      0.24987126886844635f,
      -0.31396418809890747f,
      0.14036843180656433f,
      0.2854192554950714f,
      0.09709841012954712f,
      0.09935075044631958f,
      -0.012154420837759972f,
      0.2575816512107849f};

  std::vector<float> bias_data = {
      0.4803391396999359f,
      -0.5254325866699219f,
      -0.42926454544067383f,
      -0.2059524953365326f,
      -0.12773379683494568f,
      -0.09542735666036606f,
      -0.35286077857017517f,
      -0.07646317780017853f,
      -0.04590314254164696f,
      -0.03752850368618965f,
      -0.013764488510787487f,
      -0.18478283286094666f};

  // One sequence has both left padding and right padding
  std::vector<int32_t> mask_index_data = {4, 3, 0, 2};

  std::vector<float> output_data = {
      0.14902574f, 0.62273371f, 0.43022552f, 0.12759127f,
      0.18029204f, 0.07451740f, 0.73694098f, 0.17766341f};

  std::vector<float> past_data = {
      0.42028648f, 0.55855948f, 0.044569403f, 0.76525789f, 0.13962431f, 0.40977913f, 0.36911047f, 0.83399564f, 0.36905321f, 0.91414654f, 0.17300875f, 0.78793788f,
      0.10279467f, 0.80501258f, 0.089550517f, 0.85371113f, 0.61801594f, 0.91222942f, 0.88626182f, 0.069776468f, 0.10591964f, 0.84836882f, 0.83520192f, 0.0098680854f,
      0.3113814f, 0.63999802f, 0.28603253f, 0.98899829f, 0.044405211f, 0.95105386f, 0.81278932f, 0.63969064f, 0.14494057f, 0.11349615f, 0.87086016f, 0.20983537f,
      0.35107401f, 0.90144604f, 0.68950737f, 0.18928574f, 0.18029204f, 0.074517399f, 0.70763874f, 0.48440042f, 0.58114725f, 0.1048766f, 0.73694098f, 0.17766342f};

  std::vector<float> present_data = {
      0.42028648f, 0.55855948f, 0.044569403f, 0.76525789f, 0.13962431f, 0.40977913f, -0.22849128f, -0.022080801f, 0.36911047f, 0.83399564f, 0.36905321f, 0.91414654f, 0.17300875f, 0.78793788f, -0.4449589f, -0.17704415f, 0.10279467f, 0.80501258f, 0.089550517f, 0.85371113f, 0.61801594f, 0.91222942f, -0.2994619f, -0.14412443f, 0.88626182f, 0.069776468f, 0.10591964f, 0.84836882f, 0.83520192f, 0.0098680854f, -0.33421949f, -0.18547727f,
      0.3113814f, 0.63999802f, 0.28603253f, 0.98899829f, 0.044405211f, 0.95105386f, -0.033968594f, -0.034833729f, 0.81278932f, 0.63969064f, 0.14494057f, 0.11349615f, 0.87086016f, 0.20983537f, 0.045759238f, -0.26863033f, 0.35107401f, 0.90144604f, 0.68950737f, 0.18928574f, 0.18029204f, 0.074517399f, -0.033201858f, -0.10592631f, 0.70763874f, 0.48440042f, 0.58114725f, 0.1048766f, 0.73694098f, 0.17766342f, -0.054369561f, -0.24562015f};

  bool is_unidirectional = true;
  bool use_past_state = true;
  int past_sequence_length = 3;

  if (!past_present_share_buffer) {
    RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                     batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional,
                     use_past_state, past_sequence_length, &past_data, &present_data, kMaskIndexEndAndStart);
  } else {
    RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                     batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional,
                     use_past_state, past_sequence_length, &past_data, &present_data, kMaskIndexEndAndStart,
                     0, past_sequence_length + sequence_length + 4,
                     true, false, true, {}, {}, 0, true);
  }
}

TEST(AttentionTest, AttentionPastStateBatch2WithPadding) {
  RawAttentionPastStateBatch2WithPadding(false);
}

TEST(AttentionTest, AttentionPastStateBatch2WithPadding_SharedPastPresent) {
  RawAttentionPastStateBatch2WithPadding(true);
}

TEST(AttentionTest, AttentionBatch2MaskIndex2) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2, 2, 0, 0};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEndAndStart);
}

TEST(AttentionTest, AttentionRightPaddingMaskIndex2) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask_index < sequence_length
  std::vector<int32_t> mask_index_data = {1, 0};

  std::vector<float> output_data = {
      8.6899995803833008f, -0.13000002503395081f, 4.25f, 5.6499996185302734f,
      8.6899995803833008f, -0.13000002503395081f, 4.2499995231628418f, 5.6499991416931152f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEndAndStart);
}

TEST(AttentionTest, AttentionLeftPaddingMaskIndex2) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask start position > 0.
  std::vector<int32_t> mask_index_data = {2, 1};

  std::vector<float> output_data = {
      8.69f, -0.13f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEndAndStart);
}

TEST(AttentionTest, AttentionBatch2LeftPaddingMaskIndex2) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask start position > 0.
  std::vector<int32_t> mask_index_data = {2, 2, 1, 0};

  std::vector<float> output_data = {
      8.69f, -0.13f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f,
      3.14959716796875f, 0.10843672603368759f, 4.25f, 5.65f,
      3.9696791172027588f, 0.073143675923347473f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEndAndStart);
}

TEST(AttentionTest, Attention3DMask) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test 3D mask BxSxS*
  std::vector<int32_t> mask_index_data = {
      0, 1,
      0, 1,
      1, 1,
      1, 1};

  std::vector<float> output_data = {
      8.69f, -0.13f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f,
      3.14959716796875f, 0.10843672603368759f, 4.25f, 5.65f,
      3.9696791172027588f, 0.073143675923347473f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMask3D);
}

TEST(AttentionTest, AttentionBatch2AttentionMask) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask start position > 0.
  std::vector<int32_t> mask_index_data = {0, 1, 1, 1};

  std::vector<float> output_data = {
      8.69f, -0.13f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f,
      3.14959716796875f, 0.10843672603368759f, 4.25f, 5.65f,
      3.9696791172027588f, 0.073143675923347473f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskRaw);
}

TEST(AttentionTest, AttentionUnidirectional3DMask) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test 3D mask BxSxS*
  std::vector<int32_t> mask_index_data = {
      0, 1,
      0, 1,
      1, 1,
      1, 1};

  std::vector<float> output_data = {
      3.0146f, 0.1142f, 3.9834f, 5.3394f,
      8.69f, -0.13f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f,
      3.96967912f, 0.07314367f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = true;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMask3D);
}

TEST(AttentionTest, AttentionUnidirectionalAttentionMask) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask start position > 0.
  std::vector<int32_t> mask_index_data = {0, 1, 1, 1};

  std::vector<float> output_data = {
      3.0146f, 0.1142f, 3.9834f, 5.3394f,
      8.69f, -0.13f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f,
      3.96967912f, 0.07314367f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = true;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskRaw);
}

TEST(AttentionTest, AttentionMask1DEndNoWord) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test that all attention masks are zero.
  std::vector<int32_t> mask_index_data = {0, 0};

  std::vector<float> output_data = {
      3.96724534f, 0.07324841f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.96724534f, 0.07324841f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEnd);
}

TEST(AttentionTest, AttentionMask1DNoWord) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test that all attention masks are zero.
  std::vector<int32_t> mask_index_data = {0, 0, 2, 2};

  std::vector<float> output_data = {
      3.96724534f, 0.07324841f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.96724534f, 0.07324841f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEndAndStart);
}

TEST(AttentionTest, AttentionMask2DNoWord) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test that all attention masks are zero.
  std::vector<int32_t> mask_index_data = {0, 0, 0, 0};

  std::vector<float> output_data = {
      3.96724534f, 0.07324841f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.96724534f, 0.07324841f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskRaw);
}

TEST(AttentionTest, AttentionMask3DNoWord) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test that all attention masks are zero.
  std::vector<int32_t> mask_index_data = {0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<float> output_data = {
      3.96724534f, 0.07324841f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.96724534f, 0.07324841f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMask3D);
}

TEST(AttentionTest, AttentionDummyMask2D) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {1, 1};

  std::vector<float> output_data = {
      3.9696791172027588f, 0.073143675923347473f, 4.25f, 5.65f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.65f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.65f,
      3.9696791172027588f, 0.073143675923347473f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskDummy);
}

TEST(AttentionTest, Attention4DMask) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test 4D mask Bx1xmax_Sxmax_S
  std::vector<int32_t> mask_index_data = {
      0, 0, 0, 0,
      0, 1, 0, 0,
      0, 1, 1, 0,
      0, 1, 1, 1};

  std::vector<float> output_data = {
      3.97f, 0.073f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  int input_hidden_size = 0;
  int max_sequence_length = 4;
  bool disable_cpu = true;  // 4D mask not support in CPU kernel
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length,
                   past_data, present_data, kMask4D, input_hidden_size, max_sequence_length,
                   disable_cpu);
}

TEST(AttentionTest, AttentionMaskIndexOutOfRange) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test end_position > sequence length, or start_position < 0
  std::vector<int32_t> mask_index_data = {3, 2, 0, -1};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEndAndStart);
}

#if !defined(__wasm__)
// TODO: fix in web assembly
TEST(AttentionTest, AttentionPastState_dynamic) {
  // ORT enables TF32 in GEMM for A100. TF32 will cause precsion loss and fail this test.
  // Do not run this test unless TF32 is disabled explicitly.
  if (HasCudaEnvironment(800) && ParseEnvironmentVariableWithDefault<int>("NVIDIA_TF32_OVERRIDE", 1) != 0) {
    GTEST_SKIP() << "Skipping AttentionPastState_dynamic in A100 since TF32 is enabled";
    return;
  }

  // create rand inputs
  RandomValueGenerator random{};

  std::vector<int64_t> input_dims{2, 5, 768};
  std::vector<float> input_data = random.Gaussian<float>(input_dims, 0.0f, 0.3f);

  std::vector<int64_t> weight_dims{768, 2304};
  std::vector<float> weight_data = random.Gaussian<float>(weight_dims, 0.0f, 0.3f);

  std::vector<int64_t> bias_dims{2304};
  std::vector<float> bias_data = random.Gaussian<float>(bias_dims, 0.0f, 0.3f);

  std::vector<int64_t> past_dims{2, 2, 12, 15, 64};
  std::vector<float> past_data = random.Gaussian<float>(past_dims, 0.0f, 0.3f);

  OpTester test("Attention", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("num_heads", 12);
  test.AddAttribute<int64_t>("unidirectional", 1);
  test.AddInput<float>("input", input_dims, input_data);
  test.AddInput<float>("weight", weight_dims, weight_data);
  test.AddInput<float>("bias", bias_dims, bias_data);
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<float>("past", past_dims, past_data);

  test.AddReferenceOutputs("testdata/attention_past_state.onnx", 0.005f);
  test.Run();
}
#endif  //! defined(__wasm__)

TEST(AttentionTest, AttentionPrunedModel) {
  int batch_size = 2;
  int sequence_length = 2;
  // test input_hidden_size > hidden_size
  int input_hidden_size = 6;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f, 0.0f, 1.0f,
      0.8f, -0.5f, 0.0f, 1.f, 2.0f, 3.0f,
      0.8f, -0.5f, 0.0f, 1.f, 4.0f, 5.0f,
      0.5f, 0.2f, 0.3f, -0.6f, 6.0f, 7.0f};

  std::vector<float> weight_data = {
      0.1f,
      -0.2f,
      0.3f,
      1.0f,
      1.1f,
      0.3f,
      0.5f,
      0.2f,
      0.3f,
      -0.6f,
      1.5f,
      2.0f,
      0.5f,
      0.1f,
      0.4f,
      1.6f,
      1.0f,
      2.0f,
      0.4f,
      0.8f,
      0.9f,
      0.1f,
      -1.3f,
      0.7f,
      0.3f,
      0.2f,
      4.0f,
      2.2f,
      1.6f,
      1.1f,
      0.7f,
      0.2f,
      0.4f,
      1.0f,
      1.2f,
      0.5f,
      0.2f,
      0.1f,
      0.4f,
      1.6f,
      2.4f,
      3.3f,
      2.1f,
      4.2f,
      8.4f,
      0.0f,
      2.1f,
      3.2f,
      0.1f,
      0.2f,
      0.3f,
      0.4f,
      0.5f,
      0.6f,
      0.7f,
      0.8f,
      0.9f,
      1.0f,
      1.1f,
      1.2f,
      1.2f,
      1.1f,
      1.0f,
      0.9f,
      0.8f,
      0.7f,
      0.6f,
      0.5f,
      0.4f,
      0.3f,
      0.2f,
      0.1f,
  };

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_data = {1, 1, 1, 1};

  std::vector<float> output_data = {
      11.689527f, 2.769937f, 7.05f, 8.350000f,
      11.690000f, 2.770000f, 7.05f, 8.350000f,
      14.276558f, 5.374159f, 9.650001f, 10.95f,
      14.289073f, 5.370287f, 9.650001f, 10.95f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskRaw, input_hidden_size);
}

static void RunModelWithRandomInput(
    int batch_size,
    int sequence_length,
    std::vector<int64_t>& mask_index_dims,
    std::vector<int32_t>& mask_index_data,
    std::string& onnx_model,
    bool is_float16) {
  // ORT enables TF32 in GEMM for A100. TF32 will cause precsion loss and fail this test.
  // Do not run this test unless TF32 is disabled explicitly.
  if (HasCudaEnvironment(800) && ParseEnvironmentVariableWithDefault<int>("NVIDIA_TF32_OVERRIDE", 1) != 0) {
    GTEST_SKIP() << "Skipping RunModelWithRandomInput in A100 since TF32 is enabled";
    return;
  }

  RandomValueGenerator random{234};

  constexpr int hidden_size = 768;
  constexpr int num_heads = 12;

  std::vector<int64_t> batch_input_dims{1, sequence_length, hidden_size};
  std::vector<float> batch_input_data = random.Uniform<float>(batch_input_dims, -1.0f, 1.0f);

  std::vector<int64_t> input_dims{batch_size, sequence_length, hidden_size};
  std::vector<float> input_data;
  for (int i = 0; i < batch_size; i++) {
    input_data.insert(input_data.end(), batch_input_data.begin(), batch_input_data.end());
  }

  std::vector<int64_t> weight_dims{hidden_size, 3 * hidden_size};
  std::vector<float> weight_data = random.Uniform<float>(weight_dims, -1.0f, 1.0f);

  std::vector<int64_t> bias_dims{3 * hidden_size};
  std::vector<float> bias_data = random.Uniform<float>(bias_dims, -1.0f, 1.0f);

  float gpu_threshold = is_float16 ? static_cast<float>(sequence_length) / 32.0f : 0.005f;
  constexpr float cpu_threshold = 0.002f;
  bool enable_cuda = HasCudaEnvironment(is_float16 ? 530 : 0);
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get());
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get() && !is_float16);
  if (enable_cuda || enable_rocm) {
    OpTester test("Attention", 1, onnxruntime::kMSDomain);
    test.AddAttribute<int64_t>("num_heads", num_heads);
    if (is_float16) {
      test.AddInput<MLFloat16>("input", input_dims, ToFloat16(input_data));
      test.AddInput<MLFloat16>("weight", weight_dims, ToFloat16(weight_data));
      test.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
    } else {
      test.AddInput<float>("input", input_dims, input_data);
      test.AddInput<float>("weight", weight_dims, weight_data);
      test.AddInput<float>("bias", bias_dims, bias_data);
    }
    test.AddInput<int>("mask_index", mask_index_dims, mask_index_data);
    test.AddReferenceOutputs(onnx_model, gpu_threshold);
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    if (enable_cuda) {
      execution_providers.push_back(DefaultCudaExecutionProvider());
    } else {
      execution_providers.push_back(DefaultRocmExecutionProvider());
    }
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }

  if (enable_cpu) {
    OpTester test("Attention", 1, onnxruntime::kMSDomain);
    test.AddAttribute<int64_t>("num_heads", num_heads);
    test.AddInput<float>("input", input_dims, input_data);
    test.AddInput<float>("weight", weight_dims, weight_data);
    test.AddInput<float>("bias", bias_dims, bias_data);
    test.AddInput<int>("mask_index", mask_index_dims, mask_index_data);
    test.AddReferenceOutputs(onnx_model, cpu_threshold);

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(AttentionTest, Attention_Mask2D_Fp32_B2_S32) {
  constexpr int batch_size = 2;
  constexpr int sequence_length = 32;

  std::vector<int64_t> mask_index_dims{batch_size, sequence_length};
  std::vector<int32_t> mask_index_data;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < sequence_length; j++) {
      mask_index_data.push_back((i == 0 || j < sequence_length / 2) ? 1 : 0);
    }
  }

  std::string onnx_model = "testdata/attention_mask2d_fp32.onnx";
  RunModelWithRandomInput(
      batch_size,
      sequence_length,
      mask_index_dims,
      mask_index_data,
      onnx_model,
      false);
}

TEST(AttentionTest, Attention_Mask1D_Fp32_B2_S64) {
  constexpr int batch_size = 2;
  constexpr int sequence_length = 64;

  std::vector<int64_t> mask_index_dims{batch_size};
  std::vector<int32_t> mask_index_data;
  for (int i = 0; i < batch_size; i++) {
    mask_index_data.push_back(i == 0 ? sequence_length : (sequence_length / 2));
  }

  std::string onnx_model = "testdata/attention_mask1d_fp32.onnx";
  RunModelWithRandomInput(
      batch_size,
      sequence_length,
      mask_index_dims,
      mask_index_data,
      onnx_model,
      false);
}

// This test is disabled since it is flaky.
TEST(AttentionTest, DISABLED_Attention_Mask1D_Fp16_B2_FusedNoPadding) {
  constexpr int batch_size = 2;

  // Sequence lengths used in TRT fused attention fp16 v2 kernels.
  std::vector<int> sequence_lengths{64, 128, 192, 256, 384, 512};

  for (const auto& sequence_length : sequence_lengths) {
    std::vector<int64_t> mask_index_dims{batch_size};
    std::vector<int32_t> mask_index_data;
    for (int i = 0; i < batch_size; i++) {
      mask_index_data.push_back(sequence_length);
    }

    std::string onnx_model = "testdata/attention_mask1d_fp16.onnx";

    RunModelWithRandomInput(
        batch_size,
        sequence_length,
        mask_index_dims,
        mask_index_data,
        onnx_model,
        true);
  }
}

#ifndef ENABLE_TRAINING  // Prepacking is enabled only on non-training builds
TEST(AttentionTest, SharedPrepackedWeights) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test that all attention masks are zero.
  std::vector<int32_t> mask_index_data = {0, 0, 2, 2};

  std::vector<float> output_data = {
      3.96724534f, 0.07324841f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.96724534f, 0.07324841f, 4.25f, 5.65f};

  OpTester tester("Attention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));
  tester.AddAttribute<int64_t>("unidirectional", static_cast<int64_t>(0));

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> weights_dims = {hidden_size, 3 * hidden_size};
  std::vector<int64_t> bias_dims = {3 * hidden_size};

  std::vector<int64_t> mask_index_dims = {2 * batch_size};
  std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size};

  tester.AddInput<float>("input", input_dims, input_data);
  tester.AddInput<float>("weight", weights_dims, weight_data, true);  // Trigger pre-packing
  tester.AddInput<float>("bias", bias_dims, bias_data);
  tester.AddOutput<float>("output", output_dims, output_data);
  tester.AddInput<int32_t>("mask_index", mask_index_dims, mask_index_data);

  OrtValue weight;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape(weights_dims),
                       weight_data.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), weight);

  SessionOptions so;

  // Set up weight as a shared initializer to be shared between sessions
  ASSERT_EQ(so.AddInitializer("weight", &weight), Status::OK());

  // We want all sessions running using this OpTester to be able to share pre-packed weights if applicable
  tester.EnableSharingOfPrePackedWeightsAcrossSessions();

  // Pre-packing is limited just to the CPU EP for now and we will only test the CPU EP
  // and we want to ensure that it is available in this build
  auto cpu_ep = []() -> std::vector<std::unique_ptr<IExecutionProvider>> {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    return execution_providers;
  };

  size_t number_of_pre_packed_weights_counter_session_1 = 0;
  size_t number_of_shared_pre_packed_weights_counter = 0;

  // Session 1
  {
    auto ep_vec = cpu_ep();
    tester.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
               &ep_vec, {}, &number_of_pre_packed_weights_counter_session_1, &number_of_shared_pre_packed_weights_counter);
    // Assert that no pre-packed weights have been shared thus far
    ASSERT_EQ(number_of_shared_pre_packed_weights_counter, static_cast<size_t>(0));
  }

  auto number_of_elements_in_shared_prepacked_buffers_container =
      tester.GetNumPrePackedWeightsShared();
  // Assert that the number of elements in the shared container
  // is the same as the number of weights that have been pre-packed
  ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_elements_in_shared_prepacked_buffers_container);

  // On some platforms/architectures MLAS may choose to not do any pre-packing and the number of elements
  // that have been pre-packed will be zero in which case we do not continue with the testing
  // of "sharing" of pre-packed weights as there are no pre-packed weights to be shared at all.
  if (number_of_pre_packed_weights_counter_session_1 == 0)
    return;

  // Session 2
  {
    size_t number_of_pre_packed_weights_counter_session_2 = 0;
    auto ep_vec = cpu_ep();
    tester.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
               &ep_vec, {}, &number_of_pre_packed_weights_counter_session_2, &number_of_shared_pre_packed_weights_counter);

    // Assert that the same number of weights were pre-packed in both sessions
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_pre_packed_weights_counter_session_2);

    // Assert that the number of pre-packed weights that were shared equals
    // the number of pre-packed weights in the second session
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_2,
              static_cast<size_t>(number_of_shared_pre_packed_weights_counter));
  }
}
#endif

}  // namespace test
}  // namespace onnxruntime
