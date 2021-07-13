// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//#include "embed_layer_norm_test_vectors.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

class OpData {
 public:
  explicit OpData(
      const int64_t batch_size,
      const int64_t sequence_length,
      const int64_t hidden_size,
      const std::vector<float>& input_data,
      const std::vector<float>& skip_data,
      const std::vector<float>& gamma_data,
      const std::vector<float>& beta_data,
      const std::vector<float>& bias_data,
      const std::vector<float>& matmul_1_b_data,
      const std::vector<float>& bias_gelu_bias_data,
      const std::vector<float>& matmul_2_b_data)
      : batch_size(batch_size)
      , sequence_length(sequence_length)
      , hidden_size(hidden_size)
      , input_data(input_data)
      , skip_data(skip_data)
      , gamma_data(gamma_data)
      , beta_data(beta_data)
      , bias_data(bias_data)
      , matmul_1_b_data(matmul_1_b_data)
      , bias_gelu_bias_data(bias_gelu_bias_data)
      , matmul_2_b_data(matmul_2_b_data) {}

  OpData() = delete;
  OpData(const OpData&) = delete;

  const int64_t batch_size;
  const int64_t sequence_length;
  const int64_t hidden_size;
  const std::vector<float>& input_data;
  const std::vector<float>& skip_data;
  const std::vector<float>& gamma_data;
  const std::vector<float>& beta_data;
  const std::vector<float>& bias_data;
  const std::vector<float>& matmul_1_b_data;
  const std::vector<float>& bias_gelu_bias_data;
  const std::vector<float>& matmul_2_b_data;
};

void TestInvokeOp(const OpData& data) {
  std::vector<int64_t> input_dims =
      {data.batch_size, data.sequence_length, data.hidden_size};
  std::vector<int64_t> skip_dims =
      {data.batch_size, data.sequence_length, data.hidden_size};
  std::vector<int64_t> gamma_dims = {data.hidden_size};
  std::vector<int64_t> beta_dims = {data.hidden_size};
  std::vector<int64_t> bias_dims = {data.hidden_size};

  OpTester tester("EmbedLayerNormBiasGelu", 1, onnxruntime::kMSDomain);

  // 5: MatMul #1 Input 1
  // 6: BiasGelu Input 1
  // 7: MatMul #2 Input 1

  tester.AddInput<float>("input", input_dims, data.input_data);
  tester.AddInput<float>("skip", skip_dims, data.skip_data);
  tester.AddInput<float>(
      "gamma", gamma_dims, data.gamma_data, /*is_initializer=*/true);
  tester.AddInput<float>(
      "beta", beta_dims, data.beta_data, /*is_initializer=*/true);
  tester.AddInput<float>(
      "bias", bias_dims, data.bias_data, /*is_initializer=*/true);
}

}  // namespace

TEST(EmbedLayerNormBiasGelu, ShouldWork) {
  const int64_t batch_size = 1;
  const int64_t sequence_length = 8;
  const int64_t hidden_size = 4;

  // clang-format off
  const std::vector<float> input_data = {
      0.877292f, 0.686702f, 0.26983f, 0.410746f,
      0.299725f, 0.314675f, 0.233526f, 0.0724602f,
      0.56409f, 0.30251f, 0.94781f, 0.706121f,
      0.0640292f, 0.301493f, 0.964735f, 0.636928f,
      0.608366f, 0.622913f, 0.395354f, 0.40979f,
      0.427133f, 0.920674f, 0.428131f, 0.626119f,
      0.73837f, 0.480352f, 0.421635f, 0.287868f,
      0.451956f, 0.167709f, 0.235804f, 0.873725f,
  };

  const std::vector<float> skip_data = {
      0.548171f, 0.993236f, 0.214446f, 0.359704f,
      0.686656f, 0.0831026f, 0.236436f, 0.434125f,
      0.685505f, 0.906413f, 0.547314f, 0.105903f,
      0.689571f, 0.0312862f, 0.696121f, 0.51158f,
      0.0757117f, 0.536431f, 0.68158f, 0.566632f,
      0.381293f, 0.823399f, 0.223761f, 0.861105f,
      0.5068f, 0.693589f, 0.831246f, 0.681481f,
      0.0792981f, 0.234749f, 0.920688f, 0.0283016f,
  };

  const std::vector<float> output_data = {
      0.931759f, 0.338765f, -0.131476f, 1.25103f,
      0.998018f, 0.271481f, -0.0677575f, 1.2027f,
      0.609153f, 0.311766f, -0.0742768f, 1.39308f,
      0.912295f, 0.453042f, -0.203592f, 1.14993f,
      0.973502f, 0.484089f, -0.217615f, 1.05523f,
      0.822251f, 0.49487f, -0.21649f, 1.13241f,
      0.869146f, 0.611431f, -0.232114f, 0.797104f,
      0.148023f, 0.656236f, -0.182884f, 1.04682f,
  };
  // clang-format on

  const std::vector<float> gamma_data = {
      0.7857108116149902f,
      0.6215428709983826f,
      0.6965682506561279f,
      0.15536907315254211f,
  };

  const std::vector<float> beta_data = {
      0.7055606842041016f,
      0.7170179486274719f,
      0.8810263872146606f,
      0.8210763931274414f,
  };

  const std::vector<float> bias_data = {
      0.21390849351882935f,
      0.7236972451210022f,
      0.7466588616371155f,
      0.29950183629989624f,
  };

  //
  //
  // TODO(kreeger): LEFT OFF RIGHT HERE. NEED CUT OVER VALUES FROM THE MODEL!
  // MIGHT BE HELPFUL TO MAKE A PYTHON TOOL TO CONVERT MATRIX VALUES TO INLINE
  // FLOAT32 VECTOR VALUES!
  //
  //
  const std::vector<float> matmul_1_b_data = {
  };
  const std::vector<float> bias_gelu_bias_data = {};
  const std::vector<float> matmul_2_b_data = {};

  OpData data(batch_size,
              sequence_length,
              hidden_size,
              input_data,
              skip_data,
              gamma_data,
              beta_data,
              bias_data,
              matmul_1_b_data,
              bias_gelu_bias_data,
              matmul_2_b_data);

  TestInvokeOp(data);
}

}  // namespace test
}  // namespace onnxruntime

