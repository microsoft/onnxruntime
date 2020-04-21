// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void RunAttentionTest(
    const std::vector<float>& input_data,         // input:      [batch_size, sequence_length, hidden_size]
    const std::vector<float>& weights_data,       // weights:    [hidden_size, 3 * hidden_size]
    const std::vector<float>& bias_data,          // bias:       [3 * hidden_size]
    const std::vector<int32_t>& mask_index_data,  // mask_index: [batch_size] or empty
    const std::vector<float>& output_data,        // output:     [batch_size, sequence_length, hidden_size]
    int batch_size,
    int sequence_length,
    int hidden_size,
    int number_of_heads,
    bool use_float16 = false,
    bool is_unidirectional = false) {
  int min_cuda_architecture = use_float16 ? 530 : 0;

  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get()) && !use_float16;

  if (enable_cpu || enable_cuda) {
    OpTester tester("Attention", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));
    tester.AddAttribute<int64_t>("unidirectional", static_cast<int64_t>(is_unidirectional ? 1 : 0));

    std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
    std::vector<int64_t> weights_dims = {hidden_size, 3 * hidden_size};
    std::vector<int64_t> bias_dims = {3 * hidden_size};
    std::vector<int64_t> mask_index_dims = {batch_size};
    std::vector<int64_t> output_dims = input_dims;

    if (use_float16) {
      tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input_data));
      tester.AddInput<MLFloat16>("weight", weights_dims, ToFloat16(weights_data));
      tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    } else {
      tester.AddInput<float>("input", input_dims, input_data);
      tester.AddInput<float>("weight", weights_dims, weights_data);
      tester.AddInput<float>("bias", bias_dims, bias_data);
      tester.AddOutput<float>("output", output_dims, output_data);
    }

    if (mask_index_data.size() > 0) {  // mask index is optional.
      tester.AddInput<int32_t>("mask_index", mask_index_dims, mask_index_data);
    }

    if (enable_cuda) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_cpu) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCpuExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
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
      -0.12125026f, -0.0066160089f, 0.38809127f, -0.22455512f
  };

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
      0.11210034042596817f
};

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
      0.28048714995384216f
  };

  // No mask_index
  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      0.28109729f, 0.069518551f, 0.0038009658f, 0.29213354f, 0.3692801f, 0.029495837f, -0.084964074f, 0.28169215f
  };

  bool is_unidirectional = true;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional);
}

}  // namespace test
}  // namespace onnxruntime
