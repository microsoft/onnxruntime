// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
static void RunTest(
    const std::vector<int32_t>& input_ids_data,
    const std::vector<int32_t>& segment_ids_data,
    const std::vector<int32_t>& mask_data,
    const std::vector<float>& word_embedding_data,
    const std::vector<float>& position_embedding_data,
    const std::vector<float>& segment_embedding_data,
    const std::vector<float>& gamma_data,
    const std::vector<float>& beta_data,
    const std::vector<float>& output_data,
    const std::vector<int32_t>& mask_index_data,
    int batch_size,
    int sequence_length,
    int hidden_size,
    bool use_float16 = false) {
  int min_cuda_architecture = use_float16 ? 530 : 0;

  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = !use_float16;

  if (enable_cpu || enable_cuda) {
    // Input and output shapes
    //   Input 0 - input_ids          : (batch_size, sequence_length)
    //   Input 1 - segment_ids        : (batch_size, sequence_length)
    //   Input 2 - mask               : (batch_size, sequence_length)
    //   Input 3 - word_embedding     : (,hidden_size)
    //   Input 4 - position_embedding : (,hidden_size)
    //   Input 5 - segment_embedding  : (,hidden_size)
    //   Input 6 - gamma              : (hidden_size)
    //   Input 7 - beta               : (hidden_size)
    //   Output 0 - output            : (batch_size, sequence_length, hidden_size)
    //   Output 1 - mask_index        : (batch_size)

    std::vector<int64_t> input_ids_dims = {batch_size, sequence_length};
    std::vector<int64_t> segment_ids_dims = {batch_size, sequence_length};
    std::vector<int64_t> mask_dims = {batch_size, sequence_length};

    ASSERT_TRUE(word_embedding_data.size() % hidden_size == 0);
    std::vector<int64_t> word_embedding_dims = {static_cast<int64_t>(word_embedding_data.size() / hidden_size), hidden_size};

    ASSERT_TRUE(position_embedding_data.size() % hidden_size == 0);
    std::vector<int64_t> position_embedding_dims = {static_cast<int64_t>(position_embedding_data.size() / hidden_size), hidden_size};

    ASSERT_TRUE(segment_embedding_data.size() % hidden_size == 0);
    std::vector<int64_t> segment_embedding_dims = {static_cast<int64_t>(segment_embedding_data.size() / hidden_size), hidden_size};

    std::vector<int64_t> gamma_dims = {hidden_size};
    std::vector<int64_t> beta_dims = gamma_dims;
    std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size};
    std::vector<int64_t> mask_index_dims = {batch_size};

    OpTester tester("EmbedLayerNormalization", 1, onnxruntime::kMSDomain);
    tester.AddInput<int32_t>("input_ids", input_ids_dims, input_ids_data);
    tester.AddInput<int32_t>("segment_ids", segment_ids_dims, segment_ids_data);
    tester.AddInput<int32_t>("mask", mask_dims, mask_data);
    if (use_float16) {
      tester.AddInput<MLFloat16>("word_embedding", word_embedding_dims, ToFloat16(word_embedding_data));
      tester.AddInput<MLFloat16>("position_embedding", position_embedding_dims, ToFloat16(position_embedding_data));
      tester.AddInput<MLFloat16>("segment_embedding", segment_embedding_dims, ToFloat16(segment_embedding_data));
      tester.AddInput<MLFloat16>("gamma", gamma_dims, ToFloat16(gamma_data));
      tester.AddInput<MLFloat16>("beta", beta_dims, ToFloat16(beta_data));
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    } else {
      tester.AddInput<float>("word_embedding", word_embedding_dims, word_embedding_data);
      tester.AddInput<float>("position_embedding", position_embedding_dims, position_embedding_data);
      tester.AddInput<float>("segment_embedding", segment_embedding_dims, segment_embedding_data);
      tester.AddInput<float>("gamma", gamma_dims, gamma_data);
      tester.AddInput<float>("beta", beta_dims, beta_data);
      tester.AddOutput<float>("output", output_dims, output_data);
    }
    tester.AddOutput<int32_t>("mask_index", mask_index_dims, mask_index_data);

    tester.Run();
  }
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch1) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<int32_t> input_ids_data = {
      1, 3};

  std::vector<int32_t> segment_ids_data = {
      0, 1};

  std::vector<int32_t> mask_data = {
      1, 1};

  std::vector<float> word_embedding_data = {
      0.2f, 0.1f, 0.4f, -0.6f,
      0.3f, 0.2f, 0.5f, 0.6f,
      0.6f, 0.7f, 0.0f, -0.1f,
      0.8f, 0.6f, 0.9f, 1.2f,
      0.1f, 0.3f, 0.5f, 0.9f,
      1.0f, -2.0f, 1.1f, 0.8f};

  std::vector<float> position_embedding_data = {
      0.1f, 0.1f, 0.4f, 0.6f,
      0.6f, 0.0f, 0.8f, 0.6f,
      0.3f, 0.9f, -2.0f, 0.8f};

  std::vector<float> segment_embedding_data = {
      0.3f, 0.4f, 0.9f, 0.1f,
      0.7f, 0.3f, 0.5f, 0.2f};

  std::vector<float> gamma_data = {
      0.25f, 0.15f, 0.45f, -0.66f};

  std::vector<float> beta_data = {
      0.6f, 0.2f, 0.5f, -0.6f};

  std::vector<float> output_data = {
      0.36917170882225037, 0.061503000557422638, 1.1598974466323853, -0.85092413425445557,
      0.74301940202713013, -0.057434864342212677, 0.84324657917022705, -0.85171419382095337};

  std::vector<int32_t> mask_index_data = {
      2};

  RunTest(input_ids_data,
          segment_ids_data,
          mask_data,
          word_embedding_data,
          position_embedding_data,
          segment_embedding_data,
          gamma_data,
          beta_data,
          output_data,
          mask_index_data,
          batch_size,
          sequence_length,
          hidden_size);
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch1_Float16) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<int32_t> input_ids_data = {
      1, 3};

  std::vector<int32_t> segment_ids_data = {
      0, 1};

  std::vector<int32_t> mask_data = {
      1, 1};

  std::vector<float> word_embedding_data = {
      0.2f, 0.1f, 0.4f, -0.6f,
      0.3f, 0.2f, 0.5f, 0.6f,
      0.6f, 0.7f, 0.0f, -0.1f,
      0.8f, 0.6f, 0.9f, 1.2f,
      0.1f, 0.3f, 0.5f, 0.9f,
      1.0f, -2.0f, 1.1f, 0.8f};

  std::vector<float> position_embedding_data = {
      0.1f, 0.1f, 0.4f, 0.6f,
      0.6f, 0.0f, 0.8f, 0.6f,
      0.3f, 0.9f, -2.0f, 0.8f};

  std::vector<float> segment_embedding_data = {
      0.3f, 0.4f, 0.9f, 0.1f,
      0.7f, 0.3f, 0.5f, 0.2f};

  std::vector<float> gamma_data = {
      0.25f, 0.15f, 0.45f, -0.66f};

  std::vector<float> beta_data = {
      0.6f, 0.2f, 0.5f, -0.6f};

  std::vector<float> output_data = {
      0.369873046875, 0.061676025390625, 1.1591796875, -0.8515625,
      0.7431640625, -0.057586669921875, 0.84326171875, -0.8525390625};

  std::vector<int32_t> mask_index_data = {
      2};

  RunTest(input_ids_data,
          segment_ids_data,
          mask_data,
          word_embedding_data,
          position_embedding_data,
          segment_embedding_data,
          gamma_data,
          beta_data,
          output_data,
          mask_index_data,
          batch_size,
          sequence_length,
          hidden_size,
          true);
}

TEST(EmbedLayerNormTest, EmbedLayerNormBatch2) {
  int batch_size = 3;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<int32_t> input_ids_data = {
      1, 3,
      1, 3,
      2, 0};

  std::vector<int32_t> segment_ids_data = {
      0, 1,
      0, 1,
      0, 0};

  std::vector<int32_t> mask_data = {
      1, 1,
      1, 1,
      1, 0};

  std::vector<float> word_embedding_data = {
      0.2f, 0.1f, 0.4f, -0.6f,
      0.3f, 0.2f, 0.5f, 0.6f,
      0.6f, 0.7f, 0.0f, -0.1f,
      0.8f, 0.6f, 0.9f, 1.2f,
      0.1f, 0.3f, 0.5f, 0.9f,
      1.0f, -2.0f, 1.1f, 0.8f};

  std::vector<float> position_embedding_data = {
      0.1f, 0.1f, 0.4f, 0.6f,
      0.6f, 0.0f, 0.8f, 0.6f,
      0.3f, 0.9f, -2.0f, 0.8f};

  std::vector<float> segment_embedding_data = {
      0.3f, 0.4f, 0.9f, 0.1f,
      0.7f, 0.3f, 0.5f, 0.2f};

  std::vector<float> gamma_data = {
      0.25f, 0.15f, 0.45f, -0.66f};

  std::vector<float> beta_data = {
      0.6f, 0.2f, 0.5f, -0.6f};

  std::vector<float> output_data = {
      0.36917170882225037, 0.061503000557422638, 1.1598974466323853, -0.85092413425445557,
      0.74301940202713013, -0.057434864342212677, 0.84324657917022705, -0.85171419382095337,
      0.36917170882225037, 0.061503000557422638, 1.1598974466323853, -0.85092413425445557,
      0.74301940202713013, -0.057434864342212677, 0.84324657917022705, -0.85171419382095337,
      0.57668739557266235, 0.2979130744934082, 0.96158987283706665, 0.44627034664154053,
      0.64977931976318359, 0.11039737612009048, 1.1869535446166992, 0.14469735324382782};

  std::vector<int32_t> mask_index_data = {
      2, 2, 1};

  RunTest(input_ids_data,
          segment_ids_data,
          mask_data,
          word_embedding_data,
          position_embedding_data,
          segment_embedding_data,
          gamma_data,
          beta_data,
          output_data,
          mask_index_data,
          batch_size,
          sequence_length,
          hidden_size);
}

// BatchSize > HiddenSize to reproduce mask processing bug
TEST(EmbedLayerNormTest, EmbedLayerNormLargeBatchSmallHiddenSize) {
  int batch_size = 5;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<int32_t> input_ids_data = {
      1, 3,
      1, 3,
      2, 0,
      1, 3,
      2, 0};

  std::vector<int32_t> segment_ids_data = {
      0, 1,
      0, 1,
      0, 0,
      0, 1,
      0, 0};

  std::vector<int32_t> mask_data = {
      1, 1,
      1, 1,
      1, 0,
      1, 1,
      1, 0};

  std::vector<float> word_embedding_data = {
      0.2f, 0.1f, 0.4f, -0.6f,
      0.3f, 0.2f, 0.5f, 0.6f,
      0.6f, 0.7f, 0.0f, -0.1f,
      0.8f, 0.6f, 0.9f, 1.2f,
      0.1f, 0.3f, 0.5f, 0.9f,
      1.0f, -2.0f, 1.1f, 0.8f};

  std::vector<float> position_embedding_data = {
      0.1f, 0.1f, 0.4f, 0.6f,
      0.6f, 0.0f, 0.8f, 0.6f,
      0.3f, 0.9f, -2.0f, 0.8f};

  std::vector<float> segment_embedding_data = {
      0.3f, 0.4f, 0.9f, 0.1f,
      0.7f, 0.3f, 0.5f, 0.2f};

  std::vector<float> gamma_data = {
      0.25f, 0.15f, 0.45f, -0.66f};

  std::vector<float> beta_data = {
      0.6f, 0.2f, 0.5f, -0.6f};

  std::vector<float> output_data = {
      0.36917170882225037, 0.061503000557422638, 1.1598974466323853, -0.85092413425445557,
      0.74301940202713013, -0.057434864342212677, 0.84324657917022705, -0.85171419382095337,
      0.36917170882225037, 0.061503000557422638, 1.1598974466323853, -0.85092413425445557,
      0.74301940202713013, -0.057434864342212677, 0.84324657917022705, -0.85171419382095337,
      0.57668739557266235, 0.2979130744934082, 0.96158987283706665, 0.44627034664154053,
      0.64977931976318359, 0.11039737612009048, 1.1869535446166992, 0.14469735324382782,
      0.36917170882225037, 0.061503000557422638, 1.1598974466323853, -0.85092413425445557,
      0.74301940202713013, -0.057434864342212677, 0.84324657917022705, -0.85171419382095337,
      0.57668739557266235, 0.2979130744934082, 0.96158987283706665, 0.44627034664154053,
      0.64977931976318359, 0.11039737612009048, 1.1869535446166992, 0.14469735324382782
  };

  std::vector<int32_t> mask_index_data = {
      2, 2, 1, 2, 1};

  RunTest(input_ids_data,
          segment_ids_data,
          mask_data,
          word_embedding_data,
          position_embedding_data,
          segment_embedding_data,
          gamma_data,
          beta_data,
          output_data,
          mask_index_data,
          batch_size,
          sequence_length,
          hidden_size);
}
}  // namespace test
}  // namespace onnxruntime
