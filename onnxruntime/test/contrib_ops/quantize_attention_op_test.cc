// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cfenv>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/quantization_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/qmath.h"
#include "core/quantization/quantization.h"

namespace onnxruntime {
namespace test {

enum class EP : char {
  CPU,
  CUDA
};


// input:      [batch_size, sequence_length, hidden_size]
// weights:    [hidden_size, 3 * hidden_size]
// bias:       [3 * hidden_size]
// mask_index: [batch_size]
// output:     [batch_size, sequence_length, hidden_size]
template <typename QInput, typename QWeight, EP ep>
void RunQAttention(const std::vector<float>& input_data,
                   const std::vector<float>& weights_data,
                   const std::vector<float>& bias_data,
                   const std::vector<int32_t>& mask_index_data,
                   const std::vector<float>& output_data,
                   quantization::Params<QInput>& input_quant_params,
                   quantization::Params<QWeight>& weight_quant_params,
                   int batch_size,
                   int sequence_length,
                   int hidden_size,
                   int number_of_heads,
                   bool is_unidirectional = false,
                   bool use_float16 = false,
                   int input_hidden_size = 0) {
  input_hidden_size = (input_hidden_size == 0) ? hidden_size : input_hidden_size;

  OpTester tester("QAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));
  if (is_unidirectional) {
    tester.AddAttribute<int64_t>("unidirectional", 1);
  }

  std::vector<int64_t> input_dims = {batch_size, sequence_length, input_hidden_size};
  std::vector<int64_t> weights_dims = {input_hidden_size, static_cast<int64_t>(3 * hidden_size)};
  std::vector<int64_t> bias_dims = {static_cast<int64_t>(3 * hidden_size)};
  std::vector<int64_t> mask_index_dims = {batch_size};
  std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size};

  if (input_quant_params.scale != 0.0f) {
    tester.AddInput<QInput>("input",
                            input_dims,
                            QuantizeTestVector<QInput>(input_data, input_quant_params));
    tester.AddInput<QWeight>("weight",
                             weights_dims,
                             QuantizeTestVector<QWeight>(weights_data, weight_quant_params));
  } else {
    bool force_symmetric = false;
    if constexpr (ep == EP::CUDA) {
      force_symmetric = true;
    }
    tester.AddInput<QInput>(
        "input",
        input_dims,
        QuantizeLinearTestVector<QInput>(input_data, input_quant_params, force_symmetric));
    tester.AddInput<QWeight>(
        "weight",
        weights_dims,
        QuantizeLinearTestVector<QWeight>(weights_data, weight_quant_params, force_symmetric));
  }
  if (use_float16) {
    tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
    tester.AddInput<MLFloat16>("input_scale", {1}, ToFloat16({input_quant_params.scale}));
    tester.AddInput<MLFloat16>("weight_scale", {1}, ToFloat16({weight_quant_params.scale}));
    tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
  } else {
    tester.AddInput<float>("bias", bias_dims, bias_data);
    tester.AddInput<float>("input_scale", {1}, {input_quant_params.scale});
    tester.AddInput<float>("weight_scale", {1}, {weight_quant_params.scale});
    tester.AddOutput<float>("output", output_dims, output_data);
  }

  if (mask_index_data.size() > 0) {
    tester.AddInput<int32_t>("mask_index", mask_index_dims, mask_index_data);
  } else {
    // mask index is optional.
    tester.AddOptionalInputEdge<int32_t>();
  }

  tester.AddInput<QInput>("input_zero_point", {1}, {input_quant_params.zero_point});
  tester.AddInput<QWeight>("weight_zero_point", {1}, {weight_quant_params.zero_point});

  if constexpr (ep == EP::CUDA) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  } else {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

static void RunQAttentionCUDA(
    const std::vector<float>& input_data,
    const std::vector<float>& weights_data,
    const std::vector<float>& bias_data,
    const std::vector<int32_t>& mask_index_data,
    const std::vector<float>& output_data,
    int batch_size,
    int sequence_length,
    int hidden_size,
    int number_of_heads,
    bool use_special_quantize_parameter = true,
    bool is_unidirectional = false,
    bool use_float16 = false,
    int input_hidden_size = 0) {
  int min_cuda_architecture = 530;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);

  if (enable_cuda) {
    quantization::Params<int8_t> input_quant_params = {0.0f, 0};
    quantization::Params<int8_t> weights_quant_params = {0.0f, 0};
    if (use_special_quantize_parameter) {
      input_quant_params.scale = 0.1f;
      weights_quant_params.scale = 0.1f;
    }
    RunQAttention<int8_t, int8_t, EP::CUDA>(
        input_data, weights_data, bias_data, mask_index_data, output_data, input_quant_params, weights_quant_params,
        batch_size, sequence_length, hidden_size, number_of_heads, is_unidirectional, use_float16, input_hidden_size);
  }
}

static void RunQAttentionU8U8(
    const std::vector<float>& input_data,
    const std::vector<float>& weights_data,
    const std::vector<float>& bias_data,
    const std::vector<int32_t>& mask_index_data,
    const std::vector<float>& output_data,
    int batch_size,
    int sequence_length,
    int hidden_size,
    int number_of_heads,
    bool use_special_quantize_parameter = true,
    bool is_unidirectional = false,
    int input_hidden_size = 0) {
  quantization::Params<uint8_t> input_quant_params = {0.0f, 0};
  quantization::Params<uint8_t> weights_quant_params = {0.0f, 0};
  if (use_special_quantize_parameter) {
    input_quant_params.scale = 0.1f;
    weights_quant_params.scale = 0.1f;
    input_quant_params.zero_point = 128;
    weights_quant_params.zero_point = 128;
  }

  RunQAttention<uint8_t, uint8_t, EP::CPU>(
      input_data, weights_data, bias_data, mask_index_data, output_data, input_quant_params, weights_quant_params,
      batch_size, sequence_length, hidden_size, number_of_heads, is_unidirectional, false, input_hidden_size);
}

static void RunQAttentionU8S8(
    const std::vector<float>& input_data,
    const std::vector<float>& weights_data,
    const std::vector<float>& bias_data,
    const std::vector<int32_t>& mask_index_data,
    const std::vector<float>& output_data,
    int batch_size,
    int sequence_length,
    int hidden_size,
    int number_of_heads,
    bool use_special_quantize_parameter = true,
    bool is_unidirectional = false,
    int input_hidden_size = 0) {
  quantization::Params<uint8_t> input_quant_params(/*scale=*/0.0f, /*zero_point=*/0);
  quantization::Params<int8_t> weights_quant_params(/*scale=*/0.0f, /*zero_point=*/0);
  if (use_special_quantize_parameter) {
    input_quant_params.scale = 0.1f;
    weights_quant_params.scale = 0.1f;
    input_quant_params.zero_point = 128;
    weights_quant_params.zero_point = 1;
  }

  RunQAttention<uint8_t, int8_t, EP::CPU>(
      input_data, weights_data, bias_data, mask_index_data, output_data, input_quant_params, weights_quant_params,
      batch_size, sequence_length, hidden_size, number_of_heads, is_unidirectional, false, input_hidden_size);
}

static void RunQAttentionAll(
    const std::vector<float>& input_data,
    const std::vector<float>& weight_data,
    const std::vector<float>& bias_data,
    const std::vector<int32_t>& mask_index_data,
    const std::vector<float>& output_data,
    int batch_size,
    int sequence_length,
    int hidden_size,
    int number_of_heads,
    bool use_special_quantize_parameter = true,
    bool is_unidirectional = false,
    bool use_float16 = false,
    int input_hidden_size = 0) {
  RunQAttentionU8U8(input_data, weight_data, bias_data, mask_index_data, output_data,
                    batch_size, sequence_length, hidden_size, number_of_heads,
                    use_special_quantize_parameter, is_unidirectional, input_hidden_size);
  RunQAttentionU8S8(input_data, weight_data, bias_data, mask_index_data, output_data,
                    batch_size, sequence_length, hidden_size, number_of_heads,
                    use_special_quantize_parameter, is_unidirectional, input_hidden_size);
  RunQAttentionCUDA(input_data, weight_data, bias_data, mask_index_data, output_data,
                    batch_size, sequence_length, hidden_size, number_of_heads,
                    use_special_quantize_parameter, is_unidirectional, use_float16, input_hidden_size);
}

TEST(QAttentionTest, QAttentionBatch1) {
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

  RunQAttentionAll(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(QAttentionTest, QAttentionBatch1_Float16) {
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
      3.15039f, 0.1082763671875f, 4.24609375f, 5.6484375f,
      3.96679f, 0.072998046875f, 4.24609f, 5.6484375f};

  RunQAttentionCUDA(input_data, weight_data, bias_data, mask_index_data, output_data,
                    batch_size, sequence_length, hidden_size, number_of_heads,
                    true /*use_special_quantize_parameter*/,
                    false /*is_unidirectional*/,
                    true /*use_float16*/);
}

TEST(QAttentionTest, QAttentionBatch2) {
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

  RunQAttentionAll(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(QAttentionTest, QAttentionMaskPartialSequence) {
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

  RunQAttentionAll(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(QAttentionTest, QAttentionMaskExceedSequence) {
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

  RunQAttentionAll(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(QAttentionTest, QAttentionNoMaskIndex) {
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

  RunQAttentionAll(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(QAttentionTest, QAttentionUnidirectional_U8U8) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.91099896f, -0.18294459f, -0.36594841f, 0.28410032f,
      -0.12125026f, -0.66160089f, 0.38809127f, -0.22455512f};

  std::vector<float> weight_data = {
      -0.2659236192703247f,
      0.2789675071835518f,
      0.7280516624450684f,
      0.50951678305864334f,
      0.20417947322130203f,
      -0.4751841351389885f,
      0.43815530836582184f,
      0.6015353370457888f,
      -0.11496957391500473f,
      -0.1773347705602646f,
      0.30928605794906616f,
      0.5648412741720676f,

      0.8960387855768204f,
      -0.27270448207855225f,
      0.14847396314144135f,
      -0.17960812151432037f,
      0.1788954995572567f,
      0.9993876516819f,
      0.3943513706326485f,
      -0.2484400011599064f,
      -0.12958766520023346f,
      0.220433309674263f,
      0.1720484346151352f,
      0.22024005651474f,

      0.59368450194597244f,
      0.1710093915462494f,
      -0.3967452347278595f,
      -0.1591450721025467f,
      0.1446179747581482f,
      -0.20505407452583313f,
      0.12749597430229187f,
      0.32139700651168823f,
      0.139958456158638f,
      -0.10619817674160004f,
      0.4528557509183884f,
      0.45598603785037994f,

      -0.7152545265853405f,
      0.109454445540905f,
      -0.1582530289888382f,
      -0.2646341919898987f,
      0.920850858092308f,
      0.701494812965393f,
      -0.19062495231628418f,
      -0.24360455572605133f,
      -0.9368397295475006f,
      0.7878211885690689f,
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
      -0.027716159820556641f, 0.091021925210952759f,
      0.080938525497913361f, 0.61913836002349854f,
      0.36089283227920532f, -0.11653690040111542f,
      -0.030121456831693649f, 0.40923327207565308f};

  RunQAttentionU8U8(input_data, weight_data, bias_data, mask_index_data, output_data,
                    batch_size, sequence_length, hidden_size, number_of_heads,
                    false /*use_special_quantize_parameter*/,
                    true /*is_unidirectional*/);
}

TEST(QAttentionTest, QAttentionUnidirectional_U8S8) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.91099896f, -0.18294459f, -0.36594841f, 0.28410032f,
      -0.12125026f, -0.66160089f, 0.38809127f, -0.22455512f};

  std::vector<float> weight_data = {
      -0.2659236192703247f,
      0.2789675071835518f,
      0.7280516624450684f,
      0.50951678305864334f,
      0.20417947322130203f,
      -0.4751841351389885f,
      0.43815530836582184f,
      0.6015353370457888f,
      -0.11496957391500473f,
      -0.1773347705602646f,
      0.30928605794906616f,
      0.5648412741720676f,

      0.8960387855768204f,
      -0.27270448207855225f,
      0.14847396314144135f,
      -0.17960812151432037f,
      0.1788954995572567f,
      0.9993876516819f,
      0.3943513706326485f,
      -0.2484400011599064f,
      -0.12958766520023346f,
      0.220433309674263f,
      0.1720484346151352f,
      0.22024005651474f,

      0.59368450194597244f,
      0.1710093915462494f,
      -0.3967452347278595f,
      -0.1591450721025467f,
      0.1446179747581482f,
      -0.20505407452583313f,
      0.12749597430229187f,
      0.32139700651168823f,
      0.139958456158638f,
      -0.10619817674160004f,
      0.4528557509183884f,
      0.45598603785037994f,

      -0.7152545265853405f,
      0.109454445540905f,
      -0.1582530289888382f,
      -0.2646341919898987f,
      0.920850858092308f,
      0.701494812965393f,
      -0.19062495231628418f,
      -0.24360455572605133f,
      -0.9368397295475006f,
      0.7878211885690689f,
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
      -0.029270321130752563f, 0.089105717837810516f,
      0.084381766617298126f, 0.62047165632247925f,
      0.36089283227920532f, -0.11732138693332672f,
      -0.029981952160596848f, 0.40998253226280212f};

  RunQAttentionU8S8(input_data, weight_data, bias_data, mask_index_data, output_data,
                    batch_size, sequence_length, hidden_size, number_of_heads,
                    false /*use_special_quantize_parameter*/,
                    true /*is_unidirectional*/);
}

TEST(QAttentionTest, QAttentionUnidirectional_CUDA) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.91099896f, -0.18294459f, -0.36594841f, 0.28410032f,
      -0.12125026f, -0.66160089f, 0.38809127f, -0.22455512f};

  std::vector<float> weight_data = {
      -0.2659236192703247f,
      0.2789675071835518f,
      0.7280516624450684f,
      0.50951678305864334f,
      0.20417947322130203f,
      -0.4751841351389885f,
      0.43815530836582184f,
      0.6015353370457888f,
      -0.11496957391500473f,
      -0.1773347705602646f,
      0.30928605794906616f,
      0.5648412741720676f,

      0.8960387855768204f,
      -0.27270448207855225f,
      0.14847396314144135f,
      -0.17960812151432037f,
      0.1788954995572567f,
      0.9993876516819f,
      0.3943513706326485f,
      -0.2484400011599064f,
      -0.12958766520023346f,
      0.220433309674263f,
      0.1720484346151352f,
      0.22024005651474f,

      0.59368450194597244f,
      0.1710093915462494f,
      -0.3967452347278595f,
      -0.1591450721025467f,
      0.1446179747581482f,
      -0.20505407452583313f,
      0.12749597430229187f,
      0.32139700651168823f,
      0.139958456158638f,
      -0.10619817674160004f,
      0.4528557509183884f,
      0.45598603785037994f,

      -0.7152545265853405f,
      0.109454445540905f,
      -0.1582530289888382f,
      -0.2646341919898987f,
      0.920850858092308f,
      0.701494812965393f,
      -0.19062495231628418f,
      -0.24360455572605133f,
      -0.9368397295475006f,
      0.7878211885690689f,
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
      -0.037525445222854614f, 0.089105717837810516f,
      0.076988570392131805f, 0.62047165632247925f,
      0.36089283227920532f, -0.11732138693332672f,
      -0.029981952160596848f, 0.40998253226280212f};

  RunQAttentionCUDA(input_data, weight_data, bias_data, mask_index_data, output_data,
                    batch_size, sequence_length, hidden_size, number_of_heads,
                    false /*use_special_quantize_parameter*/,
                    true /*is_unidirectional*/);
}

template <typename InputT, typename WeightT>
void TestQuantizedAttentionPastState(int64_t batch,
                                     int64_t seq_len,
                                     int64_t past_seq_len,
                                     int64_t hidden_size,
                                     int64_t head_number,
                                     int64_t head_size,
                                     const std::string& reference_model,
                                     bool is_weight_constant,
                                     bool per_column = false) {
  // create rand inputs
  RandomValueGenerator random{};

  constexpr InputT input_min = std::numeric_limits<InputT>::min();
  constexpr InputT input_max = std::numeric_limits<InputT>::max();
  constexpr int32_t input_range = input_max - input_min;

  int64_t weight_scale_zp_size = per_column ? 3 * hidden_size : 1;

  InputT input_mean = (input_min + input_max) / 2 + 1;
  std::vector<InputT> input_zero_point{input_mean};

  std::vector<int64_t> input_dims{batch, seq_len, hidden_size};
  std::vector<InputT> input_data = random.Gaussian<InputT>(input_dims, input_mean, static_cast<InputT>(input_range / 6), input_min, input_max);

  constexpr WeightT weight_min = std::numeric_limits<WeightT>::min();
  constexpr WeightT weight_max = std::numeric_limits<WeightT>::max();
  constexpr int32_t weight_range = weight_max - weight_min;

  std::vector<WeightT> weight_zero_point(weight_scale_zp_size);
  for (auto& zp : weight_zero_point) {
    zp = static_cast<WeightT>(random.Uniform<int32_t>({1}, weight_min, weight_max)[0]);
  }

  WeightT weight_mean = (weight_min + weight_max) / 2 + 1;
  std::vector<int64_t> weight_dims{hidden_size, 3 * hidden_size};
  std::vector<WeightT> weight_data = random.Gaussian<WeightT>(weight_dims, weight_mean, static_cast<WeightT>(weight_range / 6), weight_min, weight_max);

  std::vector<int64_t> bias_dims{3 * hidden_size};
  std::vector<float> bias_data = random.Gaussian<float>(bias_dims, 0.0f, 0.3f);

  std::vector<float> input_scale{0.005f};
  std::vector<float> weight_scale(random.Uniform<float>({weight_scale_zp_size}, -0.01f, 0.01f));

  std::vector<int64_t> past_dims{2, batch, head_number, past_seq_len, head_size};
  std::vector<float> past_data = random.Gaussian<float>(past_dims, 0.0f, 0.3f);

  OpTester test("QAttention", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("num_heads", head_number);
  test.AddAttribute<int64_t>("unidirectional", 1);
  test.AddInput<InputT>("input", input_dims, input_data);
  test.AddInput<WeightT>("weight", weight_dims, weight_data, is_weight_constant);
  test.AddInput<float>("bias", bias_dims, bias_data);
  test.AddInput<float>("input_scale", {1}, input_scale);
  test.AddInput<float>("weight_scale", {weight_scale_zp_size}, weight_scale);
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<InputT>("input_zero_point", {1}, input_zero_point);
  test.AddInput<WeightT>("weight_zero_point", {weight_scale_zp_size}, weight_zero_point);
  test.AddInput<float>("past", past_dims, past_data);

  test.AddReferenceOutputs(reference_model);
  test.Run();
}

TEST(QAttentionTest, QAttentionPastState_u8u8) {
  TestQuantizedAttentionPastState<uint8_t, uint8_t>(2, 5, 15, 768, 12, 64,
                                                    "testdata/attention_past_state.u8u8.onnx",
                                                    false /*is_weight_constant*/);

  TestQuantizedAttentionPastState<uint8_t, uint8_t>(2, 5, 15, 768, 12, 64,
                                                    "testdata/attention_past_state.u8u8.onnx",
                                                    true /*is_weight_constant*/);

  TestQuantizedAttentionPastState<uint8_t, uint8_t>(2, 5, 15, 768, 12, 64,
                                                    "testdata/attention_past_state.u8u8.onnx",
                                                    false /*is_weight_constant*/,
                                                    true /*per_column*/);

  TestQuantizedAttentionPastState<uint8_t, uint8_t>(2, 5, 15, 768, 12, 64,
                                                    "testdata/attention_past_state.u8u8.onnx",
                                                    true /*is_weight_constant*/,
                                                    true /*per_column*/);
}

TEST(QAttentionTest, QAttentionPastState_u8s8) {
  TestQuantizedAttentionPastState<uint8_t, int8_t>(2, 5, 15, 768, 12, 64,
                                                   "testdata/attention_past_state.u8s8.onnx",
                                                   false /*is_weight_constant*/);

  TestQuantizedAttentionPastState<uint8_t, int8_t>(2, 5, 15, 768, 12, 64,
                                                   "testdata/attention_past_state.u8s8.onnx",
                                                   true /*is_weight_constant*/);

  TestQuantizedAttentionPastState<uint8_t, int8_t>(2, 5, 15, 768, 12, 64,
                                                   "testdata/attention_past_state.u8s8.onnx",
                                                   false /*is_weight_constant*/,
                                                   true /*per_column*/);

  TestQuantizedAttentionPastState<uint8_t, int8_t>(2, 5, 15, 768, 12, 64,
                                                   "testdata/attention_past_state.u8s8.onnx",
                                                   true /*is_weight_constant*/,
                                                   true /*per_column*/);
}

TEST(QAttentionTest, QAttentionPrunedModel) {
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
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f,
      0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f,
      1.2f, 1.1f, 1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_data = {};

  std::vector<float> output_data = {
      11.689527f, 2.769937f, 7.05f, 8.35f,
      11.69f, 2.77f, 7.05f, 8.35f,
      14.276558f, 5.374159f, 9.65f, 10.95f,
      14.289073f, 5.370287f, 9.65f, 10.95f};

  bool use_special_quantize_parameter = true;
  bool is_unidirectional = false;
  bool use_float16 = false;
  RunQAttentionAll(input_data, weight_data, bias_data, mask_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_special_quantize_parameter, is_unidirectional, use_float16,
                   input_hidden_size);
}

#ifndef ENABLE_TRAINING  // Prepacking is enabled only on non-training builds
TEST(QAttentionTest, SharedPrepackedWeights) {
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

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> weights_dims = {hidden_size, 3 * hidden_size};
  std::vector<int64_t> bias_dims = {3 * hidden_size};
  std::vector<int64_t> mask_index_dims = {batch_size};
  std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size};

  OpTester tester("QAttention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));

  tester.AddInput<uint8_t>(
      "input",
      input_dims,
      QuantizeTestVector<uint8_t>(
          input_data,
          quantization::Params<uint8_t>(/*scale=*/0.1f, /*zero_point=*/128)));

  auto weight_data_converted_to_int = QuantizeTestVector<uint8_t>(
      weight_data,
      quantization::Params<uint8_t>(/*scale=*/0.1f, /*zero_point=*/128));
  tester.AddInput<uint8_t>("weight",
                           weights_dims,
                           weight_data_converted_to_int,
                           /*is_initializer=*/true);  // Trigger pre-packing

  tester.AddInput<float>("bias", bias_dims, bias_data);
  tester.AddInput<float>("input_scale", {1}, {0.1f});
  tester.AddInput<float>("weight_scale", {1}, {0.1f});
  tester.AddOutput<float>("output", output_dims, output_data);

  tester.AddInput<int32_t>("mask_index", mask_index_dims, mask_index_data);

  tester.AddInput<uint8_t>("input_zero_point", {1}, {128});
  tester.AddInput<uint8_t>("weight_zero_point", {1}, {128});

  auto p_tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<uint8_t>(), TensorShape(weights_dims),
                                           weight_data_converted_to_int.data(),
                                           OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator));
  OrtValue weight;

  weight.Init(p_tensor.release(), DataTypeImpl::GetType<Tensor>(),
              DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

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
