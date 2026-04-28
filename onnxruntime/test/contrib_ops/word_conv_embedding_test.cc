// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <vector>
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

void InitializeTestWithoutAttribute(OpTester& test) {
  // sequence has 2 words and each words has 5 chars
  std::vector<int64_t> seq_words_shape = {2, 5};
  std::vector<int> seq_words{1, 2, 3, 4, 0,
                             4, 3, 2, 1, 0};

  // Charset has 5 chars and each char is represented with a vector of 3
  std::vector<int64_t> W_char_embedding_shape = {5, 3};
  std::vector<float> W_char_embedding{0.1f, 0.2f, 0.3f,
                                      0.2f, 0.3f, 0.1f,
                                      0.3f, 0.1f, 0.2f,
                                      0.4f, 0.5f, 0.6f,
                                      0.7f, 0.8f, 0.9f};

  std::vector<int64_t> W_conv_shape = {2, 1, 2, 3};
  std::vector<float> W_conv{0.1f, 0.2f, 0.3f,
                            0.2f, 0.3f, 0.1f,
                            0.3f, 0.1f, 0.2f,
                            1.0f, 1.1f, 1.2f};

  std::vector<int64_t> B_conv_shape = {2};
  std::vector<float> B_conv{0.1f, 0.2f};

  std::vector<int64_t> output_shape = {2, 2};
  std::vector<float> output{0.711393774f, 0.996334076f, 0.711393774f, 0.981612563f};

  test.AddInput<int>("Sequence", seq_words_shape, seq_words);
  test.AddInput<float>("W", W_conv_shape, W_conv);
  test.AddInput<float>("B", B_conv_shape, B_conv);
  test.AddInput<float>("C", W_char_embedding_shape, W_char_embedding);
  test.AddOutput<float>("Y", output_shape, output);
}

TEST(ContribOpTest, WordConvEmbedding) {
  // Invalid input dimensions
  OpTester test("WordConvEmbedding", 1, onnxruntime::kMSDomain);
  InitializeTestWithoutAttribute(test);
  test.Run();
}

TEST(ContribOpTest, WordConvEmbedding_valid_attribute) {
  // Invalid input dimensions
  OpTester test("WordConvEmbedding", 1, onnxruntime::kMSDomain);
  InitializeTestWithoutAttribute(test);
  test.AddAttribute<int64_t>("embedding_size", 2LL);
  test.AddAttribute<int64_t>("conv_window_size", 2LL);
  test.AddAttribute<int64_t>("char_embedding_size", 3LL);
  test.Run();
}

TEST(ContribOpTest, WordConvEmbedding_embedding_size_mismatch) {
  // Invalid input dimensions
  OpTester test("WordConvEmbedding", 1, onnxruntime::kMSDomain);
  InitializeTestWithoutAttribute(test);
  test.AddAttribute<int64_t>("embedding_size", 3LL);
  test.AddAttribute<int64_t>("conv_window_size", 2LL);
  test.AddAttribute<int64_t>("char_embedding_size", 3LL);
  test.Run(OpTester::ExpectResult::kExpectFailure);
}

TEST(ContribOpTest, WordConvEmbedding_conv_window_size_mismatch) {
  // Invalid input dimensions
  OpTester test("WordConvEmbedding", 1, onnxruntime::kMSDomain);
  InitializeTestWithoutAttribute(test);
  test.AddAttribute<int64_t>("embedding_size", 2LL);
  test.AddAttribute<int64_t>("conv_window_size", 1LL);
  test.AddAttribute<int64_t>("char_embedding_size", 3LL);
  test.Run(OpTester::ExpectResult::kExpectFailure);
}

TEST(ContribOpTest, WordConvEmbedding_char_embedding_size_mismatch) {
  // Invalid input dimensions
  OpTester test("WordConvEmbedding", 1, onnxruntime::kMSDomain);
  InitializeTestWithoutAttribute(test);
  test.AddAttribute<int64_t>("embedding_size", 2LL);
  test.AddAttribute<int64_t>("conv_window_size", 2LL);
  test.AddAttribute<int64_t>("char_embedding_size", 4LL);
  test.Run(OpTester::ExpectResult::kExpectFailure);
}

TEST(ContribOpTest, WordConvEmbedding_char_embedding_shape_conv_shape_not_match) {
  // Invalid input dimensions
  OpTester test("WordConvEmbedding", 1, onnxruntime::kMSDomain);

  // sequence has 2 words and each words has 5 chars
  std::vector<int64_t> seq_words_shape = {2, 5};
  std::vector<int> seq_words{1, 2, 3, 4, 0,
                             4, 3, 2, 1, 0};

  // Charset has 5 chars and each char is represented with a vector of 3
  std::vector<int64_t> W_char_embedding_shape = {5, 3};
  std::vector<float> W_char_embedding{0.1f, 0.2f, 0.3f,
                                      0.2f, 0.3f, 0.1f,
                                      0.3f, 0.1f, 0.2f,
                                      0.4f, 0.5f, 0.6f,
                                      0.7f, 0.8f, 0.9f};

  std::vector<int64_t> W_conv_shape = {2, 1, 2, 2};
  std::vector<float> W_conv{0.1f, 0.2f,
                            0.2f, 0.3f,
                            0.3f, 0.1f,
                            1.0f, 1.1f};

  std::vector<int64_t> B_conv_shape = {2};
  std::vector<float> B_conv{0.1f, 0.2f};

  std::vector<int64_t> output_shape = {2, 2};
  std::vector<float> output{0.711393774f, 0.996334076f, 0.711393774f, 0.981612563f};

  test.AddInput<int>("Sequence", seq_words_shape, seq_words);
  test.AddInput<float>("W", W_conv_shape, W_conv);
  test.AddInput<float>("B", B_conv_shape, B_conv);
  test.AddInput<float>("C", W_char_embedding_shape, W_char_embedding);
  test.AddOutput<float>("Y", output_shape, output);

  test.Run(OpTester::ExpectResult::kExpectFailure);
}

TEST(ContribOpTest, WordConvEmbedding_out_of_range_char_index_treated_as_padding) {
  OpTester test("WordConvEmbedding", 1, onnxruntime::kMSDomain);

  test.AddAttribute<int64_t>("embedding_size", 1LL);
  test.AddAttribute<int64_t>("conv_window_size", 2LL);
  test.AddAttribute<int64_t>("char_embedding_size", 1LL);

  test.AddInput<int>("Sequence", {1, 2}, {1, 99});
  test.AddInput<float>("W", {1, 1, 2, 1}, {1.0f, 1.0f});
  test.AddInput<float>("B", {1}, {0.0f});
  test.AddInput<float>("C", {2, 1}, {123.0f, 2.0f});
  test.AddOutput<float>("Y", {1, 1}, {std::tanh(2.0f)});

  test.Run();
}

TEST(ContribOpTest, WordConvEmbedding_rejects_filter_width_larger_than_word_length) {
  OpTester test("WordConvEmbedding", 1, onnxruntime::kMSDomain);

  test.AddInput<int>("Sequence", {1, 2}, {1, 2});
  test.AddInput<float>("W", {1, 1, 3, 1}, {1.0f, 1.0f, 1.0f});
  test.AddInput<float>("B", {1}, {0.0f});
  test.AddInput<float>("C", {3, 1}, {0.0f, 1.0f, 2.0f});
  test.AddOutput<float>("Y", {1, 1}, {0.0f});

  test.Run(OpTester::ExpectResult::kExpectFailure, "Conv kernel width must not exceed word length");
}

TEST(ContribOpTest, WordConvEmbedding_rejects_sequence_rank_one) {
  OpTester test("WordConvEmbedding", 1, onnxruntime::kMSDomain);

  test.AddInput<int>("Sequence", {2}, {1, 2});
  test.AddInput<float>("W", {1, 1, 2, 1}, {1.0f, 1.0f});
  test.AddInput<float>("B", {1}, {0.0f});
  test.AddInput<float>("C", {3, 1}, {0.0f, 1.0f, 2.0f});
  test.AddOutput<float>("Y", {1, 1}, {0.0f});

  test.Run(OpTester::ExpectResult::kExpectFailure, "Sequence input must have rank greater than 1");
}

}  // namespace test
}  // namespace onnxruntime
