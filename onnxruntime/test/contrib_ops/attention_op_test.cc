// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include <iterator>
#include <vector>
#include <string>
#include <algorithm>

namespace onnxruntime {
namespace test {

//   Input 0 - input       : (batch_size, sequence_length, hidden_size)
//   Input 1 - weights     : (hidden_size, 3 * hidden_size)
//   Input 2 - bias        : (3 * hidden_size)
//   Input 3 - mask_index  : (batch_size)
//   Output                : (batch_size, sequence_length, hidden_size)

static void RunAttentionTest(
    const std::vector<float>& input_data,        // input:      [batch_size, sequence_length, hidden_size]
    const std::vector<float>& weights_data,      // weights:    [hidden_size, 3 * hidden_size]
    const std::vector<float>& bias_data,         // bias:       [3 * hidden_size]
    const std::vector<int32_t>& mask_index_data, // mask_index: [batch_size]
    const std::vector<float>& output_data,       // output:     [batch_size, sequence_length, hidden_size]
    int batch_size,
    int sequence_length,
    int hidden_size,
    int number_of_heads)
  {
      OpTester test("Attention", 1, onnxruntime::kOnnxDomain);
      test.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));

      std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
      std::vector<int64_t> weights_dims = {hidden_size, 3 * hidden_size};
      std::vector<int64_t> bias_dims = {3 * hidden_size};
      std::vector<int64_t> mask_index_dims = {batch_size};
      std::vector<int64_t> output_dims = input_dims;

      test.AddInput<float>("input", input_dims, input_data);
      test.AddInput<float>("weight", weights_dims, weights_data);
      test.AddInput<float>("bias", bias_dims, bias_data);
      test.AddInput<int32_t>("mask_index", mask_index_dims, mask_index_data);
      test.AddOutput<float>("output", output_dims, output_data);

      test.Run();
}

TEST(AttentionTest, AttentionValidInput_1) {
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
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f
      };

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

}  // namespace test
}  // namespace onnxruntime
