// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
constexpr float epsilon_ = 1e-12f;

static void RunTest(
    const std::vector<float>& input_data,
    const std::vector<float>& skip_data,
    const std::vector<float>& gamma_data,
    const std::vector<float>& beta_data,
    const std::vector<float>& bias_data,
    const std::vector<float>& output_data,
    const std::vector<float>& skip_input_bias_add_output_data,
    float epsilon,
    int batch_size,
    int sequence_length,
    int hidden_size,
    bool use_float16 = false,
    bool no_beta = false,
    bool simplified = false) {
  // Input and output shapes
  //   Input 0 - input: (batch_size, sequence_length, hidden_size)
  //   Input 1 - skip : (batch_size, sequence_length, hidden_size)
  //   Input 2 - gamma: (hidden_size)
  //   Input 3 - beta : (hidden_size)
  //   Output         : (batch_size, sequence_length, hidden_size)
  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> skip_dims = input_dims;
  std::vector<int64_t> gamma_dims = {hidden_size};
  std::vector<int64_t> beta_dims = gamma_dims;
  std::vector<int64_t> bias_dims = gamma_dims;
  std::vector<int64_t> output_dims = input_dims;

  std::string op_type = simplified ? "SkipSimplifiedLayerNormalization" : "SkipLayerNormalization";

  auto rocm_ep = DefaultRocmExecutionProvider();
  auto dml_ep = DefaultDmlExecutionProvider();
  if (!use_float16) {
    OpTester test(op_type.c_str(), 1, onnxruntime::kMSDomain);
    test.AddInput<float>("input", input_dims, input_data);
    test.AddInput<float>("skip", skip_dims, skip_data);
    test.AddInput<float>("gamma", gamma_dims, gamma_data);
    if (!simplified) {
      if (!no_beta) {
        test.AddInput<float>("beta", beta_dims, beta_data);
      } else {
        test.AddOptionalInputEdge<float>();
      }
    }
    test.AddAttribute("epsilon", epsilon);
    if (!bias_data.empty()) {
      test.AddInput<float>("bias", bias_dims, bias_data);
    }

    test.AddOutput<float>("output", output_dims, output_data);

    if (skip_input_bias_add_output_data.size() != 0) {
      // The second and third outputs are reserved for something else
      test.AddOptionalOutputEdge<float>();
      test.AddOptionalOutputEdge<float>();

      test.AddOutput<float>("skip_input_bias_add_output",
                            output_dims,
                            skip_input_bias_add_output_data);
    }

    test.Run();
  } else if (HasCudaEnvironment(530 /*min_cuda_architecture*/) ||
             dml_ep != nullptr ||
             rocm_ep != nullptr) {
    OpTester test(op_type.c_str(), 1, onnxruntime::kMSDomain);
    test.AddInput<MLFloat16>("input", input_dims, ToFloat16(input_data));
    test.AddInput<MLFloat16>("skip", skip_dims, ToFloat16(skip_data));
    test.AddInput<MLFloat16>("gamma", gamma_dims, ToFloat16(gamma_data));
    if (!simplified) {
      if (!no_beta) {
        test.AddInput<MLFloat16>("beta", beta_dims, ToFloat16(beta_data));
      } else {
        test.AddOptionalInputEdge<float>();
      }
    }
    test.AddAttribute("epsilon", epsilon);
    if (!bias_data.empty()) {
      test.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
    }

    test.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));

    if (skip_input_bias_add_output_data.size() != 0) {
      // The second and third outputs are reserved for something else
      test.AddOptionalOutputEdge<MLFloat16>();
      test.AddOptionalOutputEdge<MLFloat16>();

      test.AddOutput<MLFloat16>("skip_input_bias_add_output",
                                output_dims,
                                ToFloat16(skip_input_bias_add_output_data));
    }

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    if (dml_ep != nullptr) {
      execution_providers.push_back(DefaultDmlExecutionProvider());
    } else if (rocm_ep != nullptr) {
      execution_providers.push_back(DefaultRocmExecutionProvider());
    } else {
      execution_providers.push_back(DefaultCudaExecutionProvider());
    }

    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(SkipLayerNormTest, SkipLayerNormNullInput) {
  int batch_size = 1;
  int sequence_length = 0;
  int hidden_size = 4;

  std::vector<float> input_data = {};

  std::vector<float> skip_data = {};

  std::vector<float> gamma_data = {
      0.3f, 0.2f, 4.0f, 2.2f};

  std::vector<float> beta_data = {
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> output_data = {};

  RunTest(input_data,
          skip_data,
          gamma_data,
          beta_data,
          std::vector<float>(),
          output_data,
          {},
          epsilon_,
          batch_size,
          sequence_length,
          hidden_size);
}

TEST(SkipLayerNormTest, SkipLayerNormBatch1) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> skip_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f};

  std::vector<float> gamma_data = {
      0.3f, 0.2f, 4.0f, 2.2f};

  std::vector<float> beta_data = {
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> output_data = {
      0.28433859348297119, -0.17090578377246857, -0.92897164821624756, 4.6924152374267578,
      0.46111652255058289, -0.21333980560302734, -0.29631003737449646, 3.5148544311523438};

  RunTest(input_data,
          skip_data,
          gamma_data,
          beta_data,
          std::vector<float>(),
          output_data,
          {},
          epsilon_,
          batch_size,
          sequence_length,
          hidden_size);
}

TEST(SkipLayerNormTest, SkipLayerNormBatch1_Float16) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> skip_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f};

  std::vector<float> gamma_data = {
      0.3f, 0.2f, 4.0f, 2.2f};

  std::vector<float> beta_data = {
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> output_data = {
      0.28433859348297119, -0.17090578377246857, -0.92897164821624756, 4.6924152374267578,
      0.46111652255058289, -0.21333980560302734, -0.29631003737449646, 3.5148544311523438};

  RunTest(input_data,
          skip_data,
          gamma_data,
          beta_data,
          std::vector<float>(),
          output_data,
          {},
          epsilon_,
          batch_size,
          sequence_length,
          hidden_size,
          true);
}

TEST(SkipLayerNormTest, SkipLayerNormBatch1_Float16_vec) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 64;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f, 0.5f, 0.2f, 0.3f, -0.6f,  // 1
      -0.8f, -0.5f, 2.0f, 1.f, 0.5f, 0.2f, 0.3f, 0.2f,  // 2
      0.8f, -0.5f, 0.0f, 1.f, -0.5f, 0.2f, 0.3f, 0.6f,  // 3
      0.8f, -0.5f, 0.0f, 1.f, 0.5f, 0.1f, 0.3f, -0.3f,  // 4
      0.8f, -3.5f, 0.9f, 1.f, 0.5f, 0.2f, 0.2f, -0.6f,  // 5
      0.8f, -0.5f, 0.0f, 1.f, 0.5f, 0.2f, 0.3f, -0.6f,  // 6
      0.9f, -0.5f, 0.8f, 2.f, 0.3f, 0.3f, 0.3f, -0.6f,  // 7
      0.8f, -0.8f, 3.0f, 1.f, 0.5f, 0.2f, 0.3f, -0.6f,  // 8
      0.8f, -0.5f, 0.1f, 1.f, 0.5f, 0.2f, 0.3f, -0.6f,  // 9
      0.8f, -1.5f, 0.0f, 6.f, 0.5f, 0.2f, 0.3f, -0.6f,  // 10
      0.8f, -0.5f, 0.0f, 2.f, 0.5f, 0.2f, 0.3f, -0.6f,  // 11
      0.8f, -0.2f, 7.0f, 1.f, -0.2f, 0.2f, 0.3f, 0.6f,  // 12
      0.8f, -0.5f, 0.0f, 1.f, 0.6f, 0.2f, 0.3f, -0.6f,  // 13
      0.8f, -0.5f, 0.0f, 1.f, 0.5f, 0.3f, 0.3f, -0.6f,  // 14
      0.8f, -0.5f, 0.0f, 1.f, 0.5f, 0.2f, -0.4f, 0.6f,  // 15
      0.8f, -0.5f, 0.0f, 1.f, 0.5f, 0.2f, 0.3f, 0.1f};  // 16

  std::vector<float> skip_data = {
      0.8f, -0.5f, 0.0f, 1.f, 0.5f, 0.2f, 0.3f, -0.6f,  // 1
      -0.8f, -0.5f, 2.0f, 1.f, 0.5f, 0.2f, 0.3f, 0.2f,  // 2
      0.8f, -0.5f, 0.0f, 1.f, -0.5f, 0.2f, 0.3f, 0.6f,  // 3
      0.8f, -0.5f, 0.0f, 3.f, 0.5f, 0.1f, 0.3f, -0.4f,  // 4
      0.8f, -3.5f, 2.9f, -0.f, 0.5f, 0.2f, 0.2f, 0.6f,  // 5
      0.8f, -0.5f, 0.0f, 1.f, 0.5f, -0.2f, 0.3f, 0.6f,  // 6
      0.9f, -0.5f, 0.8f, 2.f, 0.3f, 0.3f, 0.3f, -0.6f,  // 7
      0.8f, -1.8f, 3.0f, 1.f, 0.5f, 0.2f, 0.3f, -0.6f,  // 8
      0.8f, -0.5f, 0.1f, 1.f, 0.5f, 0.2f, 0.3f, -0.6f,  // 9
      0.8f, -1.5f, 0.0f, 6.f, 0.5f, 0.2f, -1.2f, 0.6f,  // 10
      0.8f, -3.5f, 0.0f, 2.f, -0.9f, 0.2f, 0.3f, 0.6f,  // 11
      0.8f, -0.2f, 7.0f, 0.f, -0.2f, 0.2f, 0.3f, 0.6f,  // 12
      0.8f, -0.5f, 4.0f, 1.f, 1.6f, 0.2f, 1.3f, -0.6f,  // 13
      0.8f, -0.5f, 0.1f, 1.f, 0.5f, 0.3f, 0.3f, -0.6f,  // 14
      0.8f, -0.5f, 1.0f, 0.f, 0.5f, 2.2f, -0.4f, 0.6f,  // 15
      0.8f, -0.5f, 0.2f, 1.f, 0.5f, 0.2f, 0.3f, 0.1f};  // 16

  std::vector<float> gamma_data = {
      0.8f, -0.5f, 0.0f, 1.f, 0.5f, 0.2f, 4.3f, -0.6f,  // 1
      -0.8f, -3.5f, 2.0f, 1.f, 0.2f, 0.2f, 0.3f, 0.2f,  // 2
      0.8f, -0.5f, 0.0f, 1.f, -0.5f, 0.2f, 0.3f, 0.6f,  // 3
      0.8f, -0.5f, 0.0f, 1.f, 0.5f, 0.1f, 0.3f, -0.3f,  // 4
      0.2f, -3.5f, 0.9f, -2.f, 0.5f, 1.2f, 0.2f, 0.6f,  // 5
      0.8f, -0.5f, 0.0f, 1.f, 0.5f, 0.2f, 3.3f, -0.6f,  // 6
      0.9f, -0.5f, -0.8f, 2.f, 0.3f, 0.3f, 0.3f, 0.6f,  // 7
      0.1f, -0.5f, 0.0f, 1.f, 0.5f, 0.2f, 0.3f, 0.1f};  // 8

  std::vector<float> beta_data = {
      0.8f, -0.5f, 0.0f, 1.f, 0.5f, 0.2f, 0.3f, -0.6f,  // 1
      -0.8f, -0.5f, 2.0f, 0.f, 0.5f, 0.2f, 4.9f, 0.2f,  // 2
      0.2f, -0.5f, 0.0f, 1.f, -0.5f, 0.2f, 0.3f, 0.6f,  // 3
      0.8f, -0.5f, 0.0f, 1.f, 0.5f, 0.2f, 0.3f, -0.3f,  // 4
      0.1f, -3.5f, 4.9f, 0.f, 0.5f, 0.2f, 0.2f, -0.6f,  // 5
      0.8f, -1.5f, 0.0f, 3.f, 0.5f, 0.7f, 0.8f, -0.6f,  // 6
      0.9f, -0.5f, 0.8f, 0.f, 0.3f, 0.3f, 0.3f, -0.6f,  // 7
      0.8f, -0.5f, 0.0f, 1.f, 0.5f, 0.2f, 0.3f, 0.1f};  // 8

  std::vector<float> output_data = {
      1.2490234f, -0.044372559f, 0.f, 1.7890625f, 0.61132812f, 0.1763916f, 0.28100586f, 0.014961243f,
      0.20117188f, 2.6894531f, 5.8476562f, 0.78955078f, 0.54443359f, 0.1763916f, 4.8984375f, 0.1763916f,
      0.64941406f, -0.044372559f, 0.f, 1.7890625f, -0.044372559f, 0.1763916f, 0.29882812f, 0.80175781f,
      1.2490234f, -0.044372559f, 0.f, 2.9238281f, 0.61132812f, 0.17687988f, 0.29882812f, -0.077514648f,
      0.21240234f, 11.59375f, 6.5273438f, -0.44458008f, 0.61132812f, 0.058441162f, 0.1763916f, -0.80664062f,
      1.2490234f, -1.0439453f, 0.f, 3.7890625f, 0.61132812f, 0.63134766f, 0.78515625f, -0.39331055f,
      1.5078125f, -0.044372559f, 0.3503418f, 3.8457031f, 0.29882812f, 0.29882812f, 0.29882812f, -1.2148438f,
      0.85595703f, 0.40893555f, 0.f, 1.7890625f, 0.61132812f, 0.1763916f, 0.29882812f, -0.0025119781f,
      1.0097656f, -0.12133789f, 0.f, 1.4189453f, 0.51367188f, 0.1583252f, -0.25830078f, -0.098876953f,
      -1.0097656f, 4.8945312f, 1.2695312f, 4.3398438f, 0.50537109f, 0.1583252f, 4.6835938f, 0.12695312f,
      0.40966797f, 0.46655273f, 0.f, 2.203125f, -0.23901367f, 0.1583252f, 0.26123047f, 0.38110352f,
      1.0097656f, -0.23901367f, 0.f, 1.0273438f, 0.23901367f, 0.17907715f, 0.26123047f, -0.33178711f,
      0.15234375f, -0.84863281f, 5.9804688f, -0.83789062f, 0.74853516f, -0.050079346f, 0.25244141f, -1.1015625f,
      1.0097656f, -1.1210938f, 0.f, 3.4179688f, 0.51367188f, 0.67431641f, 0.37133789f, -0.098876953f,
      1.1357422f, -0.12133789f, 0.77832031f, 0.053985596f, 0.30810547f, 0.47265625f, 0.096557617f, -0.53662109f,
      0.82617188f, -0.12133789f, 0.f, 1.4189453f, 0.51367188f, 0.1583252f, 0.26123047f, 0.071289062f};

  RunTest(input_data,
          skip_data,
          gamma_data,
          beta_data,
          std::vector<float>(),
          output_data,
          {},
          epsilon_,
          batch_size,
          sequence_length,
          hidden_size,
          true);
}

TEST(SkipLayerNormTest, SkipLayerNormBatch1_NoBeta) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> skip_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f};

  std::vector<float> gamma_data = {
      0.3f, 0.2f, 4.0f, 2.2f};

  std::vector<float> beta_data = {};

  std::vector<float> output_data = {
      0.08433859348297119f, -0.27090578377246857f, -1.32897164821624756f, 3.0924152374267578f,
      0.26111652255058289f, -0.31333980560302734f, -0.69631003737449646f, 1.9148544311523438f};

  RunTest(input_data,
          skip_data,
          gamma_data,
          beta_data,
          std::vector<float>(),
          output_data,
          {},
          epsilon_,
          batch_size,
          sequence_length,
          hidden_size,
          false,
          true);
}

TEST(SkipLayerNormTest, SkipLayerNormBatch2) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> skip_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      1.8f, -0.3f, 0.0f, 1.f,
      -0.5f, 0.4f, 0.8f, -0.6f};

  std::vector<float> gamma_data = {
      0.3f, 0.2f, 4.0f, 2.2f};

  std::vector<float> beta_data = {
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> output_data = {
      0.28433859348297119, -0.17090578377246857, -0.92897164821624756, 4.6924152374267578,
      0.46111652255058289, -0.21333980560302734, -0.29631003737449646, 3.5148544311523438,
      0.55470430850982666, -0.15080101788043976, -2.3229825496673584, 3.255286693572998,
      0.15631480515003204, 0.21066918969154358, 4.9432611465454102, -1.7957965135574341};

  RunTest(input_data,
          skip_data,
          gamma_data,
          beta_data,
          std::vector<float>(),
          output_data,
          {},
          epsilon_,
          batch_size,
          sequence_length,
          hidden_size);
}

TEST(SkipLayerNormTest, SkipLayerNormBatch2_Bias) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.7f, -0.4f, -0.2f, 1.2f,
      0.4f, 0.3f, 0.1f, -0.4f,
      0.7f, -0.4f, -0.2f, 1.2f,
      0.4f, 0.3f, 0.1f, -0.4f};

  std::vector<float> skip_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      1.8f, -0.3f, 0.0f, 1.f,
      -0.5f, 0.4f, 0.8f, -0.6f};

  std::vector<float> gamma_data = {
      0.3f, 0.2f, 4.0f, 2.2f};

  std::vector<float> beta_data = {
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> bias_data = {
      0.1f, -0.1f, 0.2f, -0.2f};

  std::vector<float> output_data = {
      0.28433859348297119, -0.17090578377246857, -0.92897164821624756, 4.6924152374267578,
      0.46111652255058289, -0.21333980560302734, -0.29631003737449646, 3.5148544311523438,
      0.55470430850982666, -0.15080101788043976, -2.3229825496673584, 3.255286693572998,
      0.15631480515003204, 0.21066918969154358, 4.9432611465454102, -1.7957965135574341};

  RunTest(input_data,
          skip_data,
          gamma_data,
          beta_data,
          bias_data,
          output_data,
          {},
          epsilon_,
          batch_size,
          sequence_length,
          hidden_size);
}

// Don't enable this test for DML/ROCM builds as these EP doesn't produce the new optional output yet
#if !defined(USE_ROCM) && !defined(USE_DML)
TEST(SkipLayerNormTest, SkipLayerNormBatch2_Bias_ProducingOptionalOutput) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.7f, -0.4f, -0.2f, 1.2f,
      0.4f, 0.3f, 0.1f, -0.4f,
      0.7f, -0.4f, -0.2f, 1.2f,
      0.4f, 0.3f, 0.1f, -0.4f};

  std::vector<float> skip_data = {
      0.1f, -0.2f, 0.3f, 1.0f,
      0.5f, 0.1f, 0.4f, 1.6f,
      1.8f, -0.3f, 0.0f, 1.f,
      -0.5f, 0.4f, 0.8f, -0.6f};

  std::vector<float> gamma_data = {
      0.3f, 0.2f, 4.0f, 2.2f};

  std::vector<float> beta_data = {
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> bias_data = {
      0.1f, -0.1f, 0.2f, -0.2f};

  std::vector<float> output_data = {
      0.28433859348297119, -0.17090578377246857, -0.92897164821624756, 4.6924152374267578,
      0.46111652255058289, -0.21333980560302734, -0.29631003737449646, 3.5148544311523438,
      0.55470430850982666, -0.15080101788043976, -2.3229825496673584, 3.255286693572998,
      0.15631480515003204, 0.21066918969154358, 4.9432611465454102, -1.7957965135574341};

  std::vector<float> input_skip_bias_add_output_data(batch_size * sequence_length * hidden_size, 0.f);
  std::transform(input_data.cbegin(), input_data.cend(), skip_data.cbegin(), input_skip_bias_add_output_data.begin(), [](const float& i, const float& s) { return i + s; });
  // Add bias
  for (int i = 0; i < batch_size * sequence_length; ++i) {
    int offset = i * hidden_size;

    for (int j = 0; j < hidden_size; ++j) {
      input_skip_bias_add_output_data[offset + j] += bias_data[j];
    }
  }

  RunTest(input_data,
          skip_data,
          gamma_data,
          beta_data,
          bias_data,
          output_data,
          input_skip_bias_add_output_data,
          epsilon_,
          batch_size,
          sequence_length,
          hidden_size);
}

TEST(SkipLayerNormTest, SkipSimplifiedLayerNormBatch1_Float16) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> skip_data = {
      0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f};

  std::vector<float> gamma_data = {
      0.3f, 0.2f, 4.0f, 2.2f};

  std::vector<float> output_data = {
      0.3491f, -0.1455f, 0.0000f, 3.2005f,
      0.3487f, 0.0930f, 2.7899f, -3.0689f};

  RunTest(input_data,
          skip_data,
          gamma_data,
          std::vector<float>(),
          std::vector<float>(),
          output_data,
          {},
          epsilon_,
          batch_size,
          sequence_length,
          hidden_size,
          true,
          true,
          true);
}
#endif

}  // namespace test
}  // namespace onnxruntime
