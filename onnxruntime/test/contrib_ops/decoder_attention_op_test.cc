// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void RunAttentionTest(
    const std::vector<float>& query_data,
    const std::vector<float>& key_data,
    const std::vector<float>& q_weights_data,
    const std::vector<float>& kv_weights_data,
    const std::vector<float>& bias_data,
    const std::vector<float>& output_data,
    int batch_size,
    int sequence_length,
    int kv_sequence_length,
    int input_cache_sen_len,
    int hidden_size,
    int num_heads,
    bool static_kv,
    bool use_past,
    bool has_layer_state,
    bool has_key_padding_mask,
    const std::vector<float>* new_key_cache = nullptr,
    const std::vector<float>* new_value_cache = nullptr,
    const std::vector<float>* key_cache = nullptr,
    const std::vector<float>* value_cache = nullptr,
    const std::initializer_list<bool>* key_padding_mask_data = nullptr) {
  bool enable_cuda = HasCudaEnvironment(0);
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get());
  bool enable_cpu = false;

  if (enable_cpu || enable_cuda || enable_rocm) {
    OpTester tester("DecoderAttention", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));
    tester.AddAttribute<float>("mask_filter_value", static_cast<float>(-10000.0f));

    int head_size = hidden_size / num_heads;
    std::vector<int64_t> query_dims = {sequence_length, batch_size, hidden_size};
    std::vector<int64_t> key_dims = {kv_sequence_length, batch_size, hidden_size};
    std::vector<int64_t> q_weights_dims = {hidden_size, hidden_size};
    std::vector<int64_t> kv_weights_dims = {hidden_size, 2 * hidden_size};
    std::vector<int64_t> bias_dims = {3 * hidden_size};
    std::vector<int64_t> input_cache_dims = {batch_size, num_heads, input_cache_sen_len, head_size};

    const std::vector<int64_t> output_dims = {sequence_length, batch_size, hidden_size};

    tester.AddInput<float>("query", query_dims, query_data);
    tester.AddInput<float>("key", key_dims, key_data);
    tester.AddInput<float>("q_weight", q_weights_dims, q_weights_data);
    tester.AddInput<float>("kv_weight", kv_weights_dims, kv_weights_data);
    tester.AddInput<float>("bias", bias_dims, bias_data);

    int src_len = 0;
    if (!has_layer_state || !use_past) {
      if (!static_kv) {
        src_len = sequence_length;
      } else {
        src_len = kv_sequence_length;
      }
    } else {
      if (!static_kv) {
        src_len = input_cache_sen_len + sequence_length;
      } else {
        src_len = input_cache_sen_len;
      }
    }

    if (nullptr == key_padding_mask_data || !has_key_padding_mask) {
      tester.AddOptionalInputEdge<bool>();
    } else {
      std::vector<int64_t> key_padding_mask_dims = {batch_size, src_len};
      tester.AddInput<bool>("key_padding_mask", key_padding_mask_dims, *key_padding_mask_data);
    }

    if (!has_layer_state || !use_past) {
      tester.AddOptionalInputEdge<float>();
      tester.AddOptionalInputEdge<float>();
    } else {
      tester.AddInput<float>("key_cache", input_cache_dims, *key_cache);
      tester.AddInput<float>("value_cache", input_cache_dims, *value_cache);
    }
    tester.AddInput<bool>("static_kv", {1}, {static_kv});
    tester.AddInput<bool>("use_past", {1}, {use_past});
    tester.AddInput<bool>("has_layer_state", {1}, {has_layer_state});
    tester.AddInput<bool>("has_key_padding_mask", {1}, {has_key_padding_mask});

    tester.AddOutput<float>("output", output_dims, output_data);
    if (has_layer_state) {
      std::vector<int64_t> output_cache_dims = {batch_size, num_heads, src_len, head_size};
      tester.AddOutput<float>("new_key_cache", output_cache_dims, *new_key_cache);
      tester.AddOutput<float>("new_value_cache", output_cache_dims, *new_value_cache);
    }
    tester.SetOutputTolerance(0.001f, 0.001f);

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    if (enable_cuda) {
      execution_providers.push_back(DefaultCudaExecutionProvider());
    }
    if (enable_rocm) {
      execution_providers.push_back(DefaultRocmExecutionProvider());
    }
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(DecoderAttentionTest, SelfAttentionNoStateNoCache) {
  int batch_size = 1;
  int sequence_length = 2;
  int kv_sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> kv_weight_data = {
      1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  // self-attn without cache
  RunAttentionTest(input_data, input_data, q_weight_data, kv_weight_data, bias_data, output_data,
                   batch_size, sequence_length, kv_sequence_length, 0, hidden_size, number_of_heads,
                   /*static_kv*/ false, /*use_past*/ false, /*has_layer_state*/ false, /*has_key_padding_mask*/ false);
}

TEST(DecoderAttentionTest, CrossAttentionNoStateNoCache) {
  int batch_size = 1;
  int sequence_length = 2;
  int kv_sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> kv_weight_data = {
      1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  // cross-attn without cache
  RunAttentionTest(input_data, input_data, q_weight_data, kv_weight_data, bias_data, output_data,
                   batch_size, sequence_length, kv_sequence_length, 0, hidden_size, number_of_heads,
                   /*static_kv*/ true, /*use_past*/ false, /*has_layer_state*/ false, /*has_key_padding_mask*/ false);
}

TEST(DecoderAttentionTest, SelfAttentionNoStateOutputCache) {
  int batch_size = 1;
  int sequence_length = 2;
  int kv_sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> kv_weight_data = {
      1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  std::vector<float> new_key_cache = {
      3.2800f, 3.2400f, 0.2900f, -0.4000f, 2.5000f, 5.1600f, -0.5200f, -1.0000f};

  std::vector<float> new_value_cache = {
      8.6900f, -0.1300f, -4.0900f, 0.4200f, 4.2500f, 5.6500f, -0.1100f, 0.5700f};

  // self-attn without cache
  RunAttentionTest(input_data, input_data, q_weight_data, kv_weight_data, bias_data, output_data,
                   batch_size, sequence_length, kv_sequence_length, 0, hidden_size, number_of_heads,
                   /*static_kv*/ false, /*use_past*/ false, /*has_layer_state*/ true, /*has_key_padding_mask*/ false,
                   &new_key_cache, &new_value_cache);
}

TEST(DecoderAttentionTest, CrossAttentionNoStateOutputCache) {
  int batch_size = 1;
  int sequence_length = 2;
  int kv_sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> kv_weight_data = {
      1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  std::vector<float> new_key_cache = {
      3.2800f, 3.2400f, 0.2900f, -0.4000f, 2.5000f, 5.1600f, -0.5200f, -1.0000f};

  std::vector<float> new_value_cache = {
      8.6900f, -0.1300f, -4.0900f, 0.4200f, 4.2500f, 5.6500f, -0.1100f, 0.5700f};

  // self-attn without cache
  RunAttentionTest(input_data, input_data, q_weight_data, kv_weight_data, bias_data, output_data,
                   batch_size, sequence_length, kv_sequence_length, 0, hidden_size, number_of_heads,
                   /*static_kv*/ true, /*use_past*/ false, /*has_layer_state*/ true, /*has_key_padding_mask*/ false,
                   &new_key_cache, &new_value_cache);
}

TEST(DecoderAttentionTest, SelfAttentionWithCache) {
  int batch_size = 1;
  int sequence_length = 2;
  int kv_sequence_length = 2;
  int input_cache_sen_len = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> kv_weight_data = {
      1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<float> output_data = {
      1.502f, 0.05172f, 4.25f, 5.6499996185302734f,
      2.0621f, 0.037995f, 4.2499995231628418f, 5.6499991416931152f};

  std::vector<float> key_cache = {
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  std::vector<float> value_cache = {
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  std::vector<float> new_key_cache = {
      0.0f, 0.0f, 0.0f, 0.0f, 3.2800f, 3.2400f, 0.2900f, -0.4000f,
      0.0f, 0.0f, 0.0f, 0.0f, 2.5000f, 5.1600f, -0.5200f, -1.0000f};

  std::vector<float> new_value_cache = {
      0.0f, 0.0f, 0.0f, 0.0f, 8.6900f, -0.1300f, -4.0900f, 0.4200f,
      0.0f, 0.0f, 0.0f, 0.0f, 4.2500f, 5.6500f, -0.1100f, 0.5700f};

  // self-attn without cache
  RunAttentionTest(input_data, input_data, q_weight_data, kv_weight_data, bias_data, output_data,
                   batch_size, sequence_length, kv_sequence_length, input_cache_sen_len, hidden_size, number_of_heads,
                   /*static_kv*/ false, /*use_past*/ true, /*has_layer_state*/ true, /*has_key_padding_mask*/ false,
                   &new_key_cache, &new_value_cache, &key_cache, &value_cache);
}

TEST(DecoderAttentionTest, CrossAttentionWithCache) {
  int batch_size = 1;
  int sequence_length = 2;
  int kv_sequence_length = 2;
  int input_cache_sen_len = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> kv_weight_data = {
      1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  std::vector<float> key_cache = {
      3.2800f, 3.2400f, 0.2900f, -0.4000f, 2.5000f, 5.1600f, -0.5200f, -1.0000f};

  std::vector<float> value_cache = {
      8.6900f, -0.1300f, -4.0900f, 0.4200f, 4.2500f, 5.6500f, -0.1100f, 0.5700f};

  std::vector<float> new_key_cache = {
      3.2800f, 3.2400f, 0.2900f, -0.4000f, 2.5000f, 5.1600f, -0.5200f, -1.0000f};

  std::vector<float> new_value_cache = {
      8.6900f, -0.1300f, -4.0900f, 0.4200f, 4.2500f, 5.6500f, -0.1100f, 0.5700f};

  // self-attn without cache
  RunAttentionTest(input_data, input_data, q_weight_data, kv_weight_data, bias_data, output_data,
                   batch_size, sequence_length, kv_sequence_length, input_cache_sen_len, hidden_size, number_of_heads,
                   /*static_kv*/ true, /*use_past*/ true, /*has_layer_state*/ true, /*has_key_padding_mask*/ false,
                   &new_key_cache, &new_value_cache, &key_cache, &value_cache);
}

TEST(DecoderAttentionTest, SelfAttentionNoStateNoCachePaddingMask) {
  int batch_size = 1;
  int sequence_length = 2;
  int kv_sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> q_weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      0.3f, 0.2f, 4.0f, 2.2f,
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> kv_weight_data = {
      1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  std::initializer_list<bool> key_padding_mask_data = {false, false};

  // self-attn without cache
  RunAttentionTest(input_data, input_data, q_weight_data, kv_weight_data, bias_data, output_data,
                   batch_size, sequence_length, kv_sequence_length, 0, hidden_size, number_of_heads,
                   /*static_kv*/ false, /*use_past*/ false, /*has_layer_state*/ false, /*has_key_padding_mask*/ true,
                   nullptr, nullptr, nullptr, nullptr, &key_padding_mask_data);
}

}  // namespace test
}  // namespace onnxruntime
