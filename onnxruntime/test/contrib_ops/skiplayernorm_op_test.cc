// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
constexpr float epsilon_ = 1e-12f;

static void RunOneTest(
    bool strict,
    const std::vector<float>& input_data,
    const std::vector<float>& skip_data,
    const std::vector<float>& gamma_data,
    const std::vector<float>& beta_data,
    const std::vector<float>& bias_data,
    const std::vector<float>& output_data,
    const std::vector<float>& sum_output_data,
    float epsilon,
    int batch_size,
    int sequence_length,
    int hidden_size,
    bool use_float16 = false,
    bool no_beta = false,
    bool simplified = false,
    bool use_token_count = false,
    bool broadcast_skip = false,
    bool no_batch_size = false) {
  // Input and output shapes
  //   Input 0 - input: (batch_size, sequence_length, hidden_size) or (batch_size * sequence_length, hidden_size)
  //   Input 1 - skip : (batch_size, sequence_length, hidden_size) or (batch_size * sequence_length, hidden_size) or (1, sequence_length, hidden_size) or (sequence_length, hidden_size)
  //   Input 2 - gamma: (hidden_size)
  //   Input 3 - beta : (hidden_size)
  //   Output         : (batch_size, sequence_length, hidden_size) or (batch_size * sequence_length, hidden_size)
  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> skip_dims = input_dims;

  if (use_token_count) {
    input_dims = {batch_size * sequence_length, hidden_size};
    skip_dims = input_dims;
  }

  if (broadcast_skip) {
    skip_dims = {1, sequence_length, hidden_size};
  }

  if (no_batch_size) {
    skip_dims = {sequence_length, hidden_size};
  }

  std::vector<int64_t> gamma_dims = {hidden_size};
  std::vector<int64_t> beta_dims = gamma_dims;
  std::vector<int64_t> bias_dims = gamma_dims;
  std::vector<int64_t> output_dims = input_dims;

  std::string op_type = simplified ? "SkipSimplifiedLayerNormalization" : "SkipLayerNormalization";

  auto rocm_ep = DefaultRocmExecutionProvider();
  auto dml_ep = DefaultDmlExecutionProvider();
  auto cpu_ep = DefaultCpuExecutionProvider();
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
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

    if (sum_output_data.size() != 0) {
      // The second and third outputs are reserved for something else
      test.AddOptionalOutputEdge<float>();
      test.AddOptionalOutputEdge<float>();

      test.AddOutput<float>("skip_input_bias_add_output",
                            output_dims,
                            sum_output_data);
    }

    if (cpu_ep != nullptr) {
      execution_providers.push_back(DefaultCpuExecutionProvider());
    }
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
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

    // Use larger threshold for fp16
    if (use_float16) {
      test.SetOutputAbsErr("output", 0.01f);
    }

    if (sum_output_data.size() != 0) {
      // The second and third outputs are reserved for something else
      test.AddOptionalOutputEdge<MLFloat16>();
      test.AddOptionalOutputEdge<MLFloat16>();

      test.AddOutput<MLFloat16>("skip_input_bias_add_output",
                                output_dims,
                                ToFloat16(sum_output_data));
    }

    if (dml_ep != nullptr) {
      execution_providers.push_back(DefaultDmlExecutionProvider());
    } else if (rocm_ep != nullptr) {
      execution_providers.push_back(DefaultRocmExecutionProvider());
    } else {
      if (strict) {
        const auto& api = Ort::GetApi();
        OrtCUDAProviderOptionsV2* cuda_options = nullptr;
        ASSERT_TRUE(api.CreateCUDAProviderOptions(&cuda_options) == nullptr);
        std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)>
            rel_cuda_options(cuda_options, api.ReleaseCUDAProviderOptions);
        std::vector<const char*> keys{"enable_skip_layer_norm_strict_mode"};
        std::vector<const char*> values{"1"};
        ASSERT_TRUE(api.UpdateCUDAProviderOptions(rel_cuda_options.get(), keys.data(), values.data(), 1) == nullptr);
        execution_providers.push_back(CudaExecutionProviderWithOptions(std::move(rel_cuda_options.get())));
      } else {
        execution_providers.push_back(DefaultCudaExecutionProvider());
      }
    }

    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

static void RunTest(
    const std::vector<float>& input_data,
    const std::vector<float>& skip_data,
    const std::vector<float>& gamma_data,
    const std::vector<float>& beta_data,
    const std::vector<float>& bias_data,
    const std::vector<float>& output_data,
    const std::vector<float>& sum_output_data,
    float epsilon,
    int batch_size,
    int sequence_length,
    int hidden_size,
    bool use_float16 = false,
    bool no_beta = false,
    bool simplified = false,
    bool use_token_count = false,
    bool broadcast_skip = false,
    bool no_batch_size = false) {
  RunOneTest(false, input_data, skip_data, gamma_data, beta_data, bias_data, output_data, sum_output_data,
             epsilon, batch_size, sequence_length, hidden_size, use_float16, no_beta, simplified,
             use_token_count, broadcast_skip, no_batch_size);

  // strict mode does not support skip broadcasting.
  if (!broadcast_skip) {
    RunOneTest(true, input_data, skip_data, gamma_data, beta_data, bias_data, output_data, sum_output_data,
               epsilon, batch_size, sequence_length, hidden_size, use_float16, no_beta, simplified,
               use_token_count, broadcast_skip, no_batch_size);
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

  // Update test data result which use internal fp32 calculation for fp16 input/parameters.
  // Following pytorch code snippet are used to generate the result: (Not torch uses fp32 internal calculation for this)
  //
  // gamma_tensor = torch.tensor(gamma_data, dtype=torch.float32).reshape(hidden_size).to('cuda:0').to(torch.float16)
  // beta_tensor = torch.tensor(beta_data, dtype=torch.float32).reshape(hidden_size).to('cuda:0').to(torch.float16)
  // input_tensor = torch.tensor(input_data, dtype=torch.float32).reshape(
  //     batch_size, sequence_length, hidden_size).to('cuda:0').to(torch.float16)
  // skip_tensor = torch.tensor(skip_data, dtype=torch.float32).reshape(
  //     batch_size, sequence_length, hidden_size).to('cuda:0').to(torch.float16)
  // added_input = torch.add(input_tensor, skip_tensor)
  // out32 = torch.layer_norm(added_input, [hidden_size], gamma_tensor, beta_tensor, eps=epsilon_).to(torch.float32)
  //
  std::vector<float> output_data = {
      1.25000000f, -0.04403687f, 0.00000000f, 1.79003906f, 0.61132812f, 0.17639160f, 0.28125000f, 0.01530457f,
      0.20166016f, 2.69140625f, 5.84765625f, 0.78955078f, 0.54443359f, 0.17639160f, 4.89843750f, 0.17639160f,
      0.64990234f, -0.04403687f, 0.00000000f, 1.79003906f, -0.04403687f, 0.17639160f, 0.29882812f, 0.80175781f,
      1.25000000f, -0.04403687f, 0.00000000f, 2.92382812f, 0.61132812f, 0.17687988f, 0.29882812f, -0.07745361f,
      0.21240234f, 11.60156250f, 6.52734375f, -0.44482422f, 0.61132812f, 0.05844116f, 0.17639160f, -0.80712891f,
      1.25000000f, -1.04394531f, 0.00000000f, 3.78906250f, 0.61132812f, 0.63134766f, 0.78564453f, -0.39331055f,
      1.50878906f, -0.04403687f, 0.34985352f, 3.84765625f, 0.29882812f, 0.29882812f, 0.29882812f, -1.21582031f,
      0.85595703f, 0.40966797f, 0.00000000f, 1.79003906f, 0.61132812f, 0.17639160f, 0.29882812f, -0.00255013f,
      1.00976562f, -0.12152100f, 0.00000000f, 1.41894531f, 0.51367188f, 0.15832520f, -0.25805664f, -0.09875488f,
      -1.00976562f, 4.89453125f, 1.26953125f, 4.33984375f, 0.50537109f, 0.15832520f, 4.68359375f, 0.12695312f,
      0.40942383f, 0.46655273f, 0.00000000f, 2.20312500f, -0.23913574f, 0.15832520f, 0.26123047f, 0.38110352f,
      1.00976562f, -0.23913574f, 0.00000000f, 1.02734375f, 0.23913574f, 0.17907715f, 0.26123047f, -0.33178711f,
      0.15234375f, -0.85058594f, 5.98046875f, -0.83789062f, 0.74853516f, -0.04998779f, 0.25244141f, -1.10156250f,
      1.00976562f, -1.12109375f, 0.00000000f, 3.41992188f, 0.51367188f, 0.67431641f, 0.37133789f, -0.09875488f,
      1.13574219f, -0.12152100f, 0.77832031f, 0.05398560f, 0.30810547f, 0.47265625f, 0.09643555f, -0.53662109f,
      0.82617188f, -0.12152100f, 0.00000000f, 1.41894531f, 0.51367188f, 0.15832520f, 0.26123047f, 0.07135010f};

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
          true /*use_float16*/,
          false /*no_beta*/,
          false /*simplified*/,
          false /*use_token_count*/);
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

TEST(SkipLayerNormTest, SkipLayerNormBatch1_Float16_vec_token_count) {
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

  // Update test data result which use internal fp32 calculation for fp16 input/parameters.
  // Following pytorch code snippet are used to generate the result: (Not torch uses fp32 internal calculation for this)
  //
  // gamma_tensor = torch.tensor(gamma_data, dtype=torch.float32).reshape(hidden_size).to('cuda:0').to(torch.float16)
  // beta_tensor = torch.tensor(beta_data, dtype=torch.float32).reshape(hidden_size).to('cuda:0').to(torch.float16)
  // input_tensor = torch.tensor(input_data, dtype=torch.float32).reshape(
  //     batch_size, sequence_length, hidden_size).to('cuda:0').to(torch.float16)
  // skip_tensor = torch.tensor(skip_data, dtype=torch.float32).reshape(
  //     batch_size, sequence_length, hidden_size).to('cuda:0').to(torch.float16)
  // added_input = torch.add(input_tensor, skip_tensor)
  // out32 = torch.layer_norm(added_input, [hidden_size], gamma_tensor, beta_tensor, eps=epsilon_).to(torch.float32)
  //
  std::vector<float> output_data = {
      1.25000000f, -0.04403687f, 0.00000000f, 1.79003906f, 0.61132812f, 0.17639160f, 0.28125000f, 0.01530457f,
      0.20166016f, 2.69140625f, 5.84765625f, 0.78955078f, 0.54443359f, 0.17639160f, 4.89843750f, 0.17639160f,
      0.64990234f, -0.04403687f, 0.00000000f, 1.79003906f, -0.04403687f, 0.17639160f, 0.29882812f, 0.80175781f,
      1.25000000f, -0.04403687f, 0.00000000f, 2.92382812f, 0.61132812f, 0.17687988f, 0.29882812f, -0.07745361f,
      0.21240234f, 11.60156250f, 6.52734375f, -0.44482422f, 0.61132812f, 0.05844116f, 0.17639160f, -0.80712891f,
      1.25000000f, -1.04394531f, 0.00000000f, 3.78906250f, 0.61132812f, 0.63134766f, 0.78564453f, -0.39331055f,
      1.50878906f, -0.04403687f, 0.34985352f, 3.84765625f, 0.29882812f, 0.29882812f, 0.29882812f, -1.21582031f,
      0.85595703f, 0.40966797f, 0.00000000f, 1.79003906f, 0.61132812f, 0.17639160f, 0.29882812f, -0.00255013f,
      1.00976562f, -0.12152100f, 0.00000000f, 1.41894531f, 0.51367188f, 0.15832520f, -0.25805664f, -0.09875488f,
      -1.00976562f, 4.89453125f, 1.26953125f, 4.33984375f, 0.50537109f, 0.15832520f, 4.68359375f, 0.12695312f,
      0.40942383f, 0.46655273f, 0.00000000f, 2.20312500f, -0.23913574f, 0.15832520f, 0.26123047f, 0.38110352f,
      1.00976562f, -0.23913574f, 0.00000000f, 1.02734375f, 0.23913574f, 0.17907715f, 0.26123047f, -0.33178711f,
      0.15234375f, -0.85058594f, 5.98046875f, -0.83789062f, 0.74853516f, -0.04998779f, 0.25244141f, -1.10156250f,
      1.00976562f, -1.12109375f, 0.00000000f, 3.41992188f, 0.51367188f, 0.67431641f, 0.37133789f, -0.09875488f,
      1.13574219f, -0.12152100f, 0.77832031f, 0.05398560f, 0.30810547f, 0.47265625f, 0.09643555f, -0.53662109f,
      0.82617188f, -0.12152100f, 0.00000000f, 1.41894531f, 0.51367188f, 0.15832520f, 0.26123047f, 0.07135010f};

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
          true /*use_float16*/,
          false /*no_beta*/,
          false /*simplified*/,
          true /*use_token_count*/);
}

TEST(SkipLayerNormTest, SkipLayerNormBatch2_TokenCount) {
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
          hidden_size,
          false,
          false,
          false,
          true);
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

#if !defined(USE_ROCM)
TEST(SkipLayerNormTest, SkipLayerNormBatch2_Skip_Broadcast_No_Batch_Size) {
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
      0.5f, 0.1f, 0.4f, 1.6f};

  std::vector<float> gamma_data = {
      0.3f, 0.2f, 4.0f, 2.2f};

  std::vector<float> beta_data = {
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> output_data = {
      0.28433859348297119, -0.17090578377246857, -0.92897164821624756, 4.6924152374267578,
      0.46111652255058289, -0.21333980560302734, -0.29631003737449646, 3.5148544311523438,
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
          false,  // use_float16
          false,  // no_beta
          false,  // simplified
          false,  // use_token_count
          true,   // broadcast_skip
          true);  // no_batch_size
}

TEST(SkipLayerNormTest, SkipLayerNormBatch2_Skip_Broadcast_Batch_Size_1) {
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
      0.5f, 0.1f, 0.4f, 1.6f};

  std::vector<float> gamma_data = {
      0.3f, 0.2f, 4.0f, 2.2f};

  std::vector<float> beta_data = {
      0.2f, 0.1f, 0.4f, 1.6f};

  std::vector<float> output_data = {
      0.28433859348297119, -0.17090578377246857, -0.92897164821624756, 4.6924152374267578,
      0.46111652255058289, -0.21333980560302734, -0.29631003737449646, 3.5148544311523438,
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
          false,   // use_float16
          false,   // no_beta
          false,   // simplified
          false,   // use_token_count
          true,    // broadcast_skip
          false);  // no_batch_size
}
#endif

}  // namespace test
}  // namespace onnxruntime
