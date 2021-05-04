// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

TEST(ReverseSequenceTest, BatchMajor) {
  OpTester test("ReverseSequence", 10);
  std::vector<int64_t> input = {0, 1, 2, 3,
                                4, 5, 6, 7};
  std::vector<int64_t> sequence_lens = {4, 3};
  std::vector<int64_t> expected_output = {3, 2, 1, 0,
                                          6, 5, 4, 7};

  test.AddAttribute("batch_axis", int64_t(0));
  test.AddAttribute("time_axis", int64_t(1));

  test.AddInput<int64_t>("input", {2, 4, 1}, input);
  test.AddInput<int64_t>("sequence_lens", {2}, sequence_lens);
  test.AddOutput<int64_t>("Y", {2, 4, 1}, expected_output);
  test.Run();
}

TEST(ReverseSequenceTest, TimeMajor) {
  OpTester test("ReverseSequence", 10);
  std::vector<int64_t> input = {0, 4,
                                1, 5,
                                2, 6,
                                3, 7};

  std::vector<int64_t> sequence_lens = {4, 3};
  std::vector<int64_t> expected_output = {3, 6,
                                          2, 5,
                                          1, 4,
                                          0, 7};

  test.AddAttribute("batch_axis", int64_t(1));
  test.AddAttribute("time_axis", int64_t(0));

  test.AddInput<int64_t>("input", {4, 2, 1}, input);
  test.AddInput<int64_t>("sequence_lens", {2}, sequence_lens);
  test.AddOutput<int64_t>("Y", {4, 2, 1}, expected_output);
  test.Run();
}

TEST(ReverseSequenceTest, LargerDim2) {
  OpTester test("ReverseSequence", 10);
  std::vector<float> input = {0.f, 1.f,
                              2.f, 3.f,
                              4.f, 5.f,

                              6.f, 7.f,
                              8.f, 9.f,
                              10.f, 11.f};
  std::vector<int64_t> sequence_lens = {2, 3};
  std::vector<float> expected_output = {2.f, 3.f,
                                        0.f, 1.f,
                                        4.f, 5.f,

                                        10.f, 11.f,
                                        8.f, 9.f,
                                        6.f, 7.f};

  test.AddAttribute("batch_axis", int64_t(0));
  test.AddAttribute("time_axis", int64_t(1));

  test.AddInput<float>("input", {2, 3, 2}, input);
  test.AddInput<int64_t>("sequence_lens", {2}, sequence_lens);
  test.AddOutput<float>("Y", {2, 3, 2}, expected_output);
  test.Run();
}

TEST(ReverseSequenceTest, Strings) {
  OpTester test("ReverseSequence", 10);
  std::vector<std::string> input = {"0", "4 string longer than 16 chars that requires its own buffer",
                                    "1", "5",
                                    "2", "6",
                                    "3", "7"};

  std::vector<int64_t> sequence_lens = {4, 3};
  std::vector<std::string> expected_output = {"3", "6",
                                              "2", "5",
                                              "1", "4 string longer than 16 chars that requires its own buffer",
                                              "0", "7"};

  test.AddAttribute("batch_axis", int64_t(1));
  test.AddAttribute("time_axis", int64_t(0));

  test.AddInput<std::string>("input", {4, 2, 1}, input);
  test.AddInput<int64_t>("sequence_lens", {2}, sequence_lens);
  test.AddOutput<std::string>("Y", {4, 2, 1}, expected_output);
  test.Run();
}

TEST(ReverseSequenceTest, InvalidInput) {
  {
    int64_t batch_size = 2, seq_size = 4;

    // Bad axis values
    auto check_bad_axis = [&](int64_t batch_dim, int64_t seq_dim,
                              const std::vector<int64_t>& input_shape,
                              const std::string err_msg) {
      OpTester test("ReverseSequence", 10);
      std::vector<int64_t> input(batch_size * seq_size, 0);
      std::vector<int64_t> sequence_lens(batch_size, 1);
      std::vector<int64_t> expected_output = input;

      test.AddAttribute("batch_axis", batch_dim);
      test.AddAttribute("time_axis", seq_dim);

      test.AddInput<int64_t>("input", input_shape, input);
      test.AddInput<int64_t>("sequence_lens", {batch_size}, sequence_lens);
      test.AddOutput<int64_t>("Y", input_shape, expected_output);
      test.Run(test::OpTester::ExpectResult::kExpectFailure, err_msg, {kTensorrtExecutionProvider});  //TensorRT engine build error
    };

    check_bad_axis(2, 1, {1, seq_size, batch_size}, "Invalid batch_axis of 2. Must be 0 or 1");
    check_bad_axis(0, 2, {batch_size, 1, seq_size}, "Invalid time_axis of 2. Must be 0 or 1");
    check_bad_axis(1, 1, {batch_size, seq_size, 1}, "time_axis and batch_axis must have different values but both are 1");
  }

  // invalid sequence_lens size
  {
    OpTester test("ReverseSequence", 10);

    // Bad data_format value
    std::vector<int64_t> input = {0, 1, 2, 3,
                                  4, 5, 6, 7};
    std::vector<int64_t> sequence_lens = {4, 3, 4};
    std::vector<int64_t> expected_output = {3, 2, 1, 0,
                                            6, 5, 4, 7};

    test.AddAttribute("batch_axis", int64_t(0));
    test.AddAttribute("time_axis", int64_t(1));

    test.AddInput<int64_t>("input", {2, 4, 1}, input);
    test.AddInput<int64_t>("sequence_lens", {3}, sequence_lens);
    test.AddOutput<int64_t>("Y", {2, 4, 1}, expected_output);
    test.Run(test::OpTester::ExpectResult::kExpectFailure,
             "sequence_lens shape must be {batch_size}. Got:{3}. batch_size=2", {kTensorrtExecutionProvider});  //TensorRT engine build error
  }
}

TEST(ReverseSequenceTest, BadLength) {
  auto run_test = [](bool use_negative) {
    OpTester test("ReverseSequence", 10);
    std::vector<int64_t> input = {0, 1, 2, 3,
                                  4, 5, 6, 7};

    std::vector<int64_t> sequence_lens = {4, 3};

    // make sequence_lens invalid for the input
    sequence_lens[1] = use_negative ? -2 : 6;

    test.AddAttribute("batch_axis", int64_t(0));
    test.AddAttribute("time_axis", int64_t(1));

    test.AddInput<int64_t>("input", {2, 4, 1}, input);
    test.AddInput<int64_t>("sequence_lens", {2}, sequence_lens);
    test.AddOutput<int64_t>("Y", {0}, {});

    // the bad length check is just in the CPU EP
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(DefaultCpuExecutionProvider());

    test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid sequence length", {}, nullptr, &eps);
  };

  run_test(true);
  run_test(false);
}

}  // namespace test
}  // namespace onnxruntime
