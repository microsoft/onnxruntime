// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// SequenceLength
TEST(SequenceOpsTest, SequenceLengthPositiveFloat) {
  OpTester test("SequenceLength", 11);
  SeqTensors<float> input;
  input.AddTensor({3, 2}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  input.AddTensor({3, 3}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.AddSeqInput("S", input);
  test.AddOutput<int64_t>("I", {}, {2});
  test.Run();
}

TEST(SequenceOpsTest, SequenceLengthPositiveInt64) {
  OpTester test("SequenceLength", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqInput("S", input);
  test.AddOutput<int64_t>("I", {}, {2});
  test.Run();
}

// SequenceAt
TEST(SequenceOpsTest, SequenceAtPositiveIdx) {
  OpTester test("SequenceAt", 11);
  SeqTensors<float> input;
  std::vector<float> output_vec{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<int64_t> output_shape{3, 3};
  input.AddTensor({3, 2}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
  input.AddTensor(output_shape, output_vec);
  test.AddSeqInput("S", input);
  test.AddInput("I", {}, {1});
  test.AddOutput<float>("T", output_shape, output_vec);
  test.Run();
}

TEST(SequenceOpsTest, SequenceAtNegativeIdx) {
  OpTester test("SequenceAt", 11);
  SeqTensors<float> input;
  std::vector<float> output_vec{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<int64_t> output_shape{3, 3};
  input.AddTensor({3, 2}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
  input.AddTensor(output_shape, output_vec);
  test.AddSeqInput("S", input);
  test.AddInput("I", {}, {-1});
  test.AddOutput<float>("T", output_shape, output_vec);
  test.Run();
}

TEST(SequenceOpsTest, SequenceAtInvalidPositiveIdx) {
  OpTester test("SequenceAt", 11);
  SeqTensors<float> input;
  std::vector<float> output_vec{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<int64_t> output_shape{3, 3};
  input.AddTensor({3, 2}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
  input.AddTensor(output_shape, output_vec);
  test.AddSeqInput("S", input);
  test.AddInput("I", {}, {10});
  test.AddOutput<float>("T", output_shape, output_vec);
  test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid sequence index");
}

TEST(SequenceOpsTest, SequenceAtInvalidNegativeIdx) {
  OpTester test("SequenceAt", 11);
  SeqTensors<float> input;
  std::vector<float> output_vec{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<int64_t> output_shape{3, 3};
  input.AddTensor({3, 2}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f});
  input.AddTensor(output_shape, output_vec);
  test.AddSeqInput("S", input);
  test.AddInput("I", {}, {-10});
  test.AddOutput<float>("T", output_shape, output_vec);
  test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid sequence index");
}

// SequenceEmpty
TEST(SequenceOpsTest, SequenceEmptyDefault) {
  OpTester test("SequenceEmpty", 11);
  test.AddSeqOutput("S", SeqTensors<float>{});
  test.Run();
}

TEST(SequenceOpsTest, SequenceEmptyInt64) {
  OpTester test("SequenceEmpty", 11);
  test.AddAttribute("dtype", static_cast<int64_t>(7));
  test.AddSeqOutput("S", SeqTensors<int64_t>{});
  test.Run();
}

// SequenceInsert
TEST(SequenceOpsTest, SequenceInsertPositiveDefaultFloat) {
  OpTester test("SequenceInsert", 11);
  SeqTensors<float> input;
  input.AddTensor({3, 2}, {1., 2., 3., 4., 5., 6.});
  input.AddTensor({3, 3}, {1., 2., 3., 4., 5., 6., 7., 8., 9.});
  test.AddSeqInput("S", input);
  test.AddInput<float>("T", {3, 2}, {10., 20., 30., 40., 50., 60.});

  SeqTensors<float> output;
  output.AddTensor({3, 2}, {1., 2., 3., 4., 5., 6.});
  output.AddTensor({3, 3}, {1., 2., 3., 4., 5., 6., 7., 8., 9.});
  output.AddTensor({3, 2}, {10., 20., 30., 40., 50., 60.});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SequenceInsertPositiveDefault) {
  OpTester test("SequenceInsert", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("T", {3, 2}, {10, 20, 30, 40, 50, 60});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  output.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  output.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SequenceInsertValidPositiveIdx) {
  OpTester test("SequenceInsert", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("T", {3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddInput<int64_t>("I", {}, {1});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  output.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  output.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SequenceInsertValidNegativeIdx) {
  OpTester test("SequenceInsert", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("T", {3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddInput<int64_t>("I", {}, {-2});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  output.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SequenceInsertInvalidPositiveIdx) {
  OpTester test("SequenceInsert", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("T", {3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddInput<int64_t>("I", {}, {99});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  output.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  output.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqOutput("S2", output);
  test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid sequence index");
}

TEST(SequenceOpsTest, SequenceInsertInvalidNegativeIdx) {
  OpTester test("SequenceInsert", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("T", {3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddInput<int64_t>("I", {}, {-99});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  output.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  output.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqOutput("S2", output);
  test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid sequence index");
}

// SequenceErase
TEST(SequenceOpsTest, SequenceErasePositiveDefault) {
  OpTester test("SequenceErase", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  input.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddSeqInput("S", input);

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  output.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SequenceEraseValidPositiveIdx) {
  OpTester test("SequenceErase", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  input.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("I", {}, {1});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  output.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SequenceEraseValidNegativeIdx) {
  OpTester test("SequenceErase", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  input.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  input.AddTensor({2, 2}, {2, 4, 6, 8});
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("I", {}, {-2});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  output.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  output.AddTensor({2, 2}, {2, 4, 6, 8});
  test.AddSeqOutput("S2", output);
  test.Run();
}

TEST(SequenceOpsTest, SequenceEraseInvalidPositiveIdx) {
  OpTester test("SequenceErase", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  input.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  input.AddTensor({2, 2}, {2, 4, 6, 8});
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("I", {}, {99});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  input.AddTensor({2, 2}, {2, 4, 6, 8});
  test.AddSeqOutput("S2", output);
  test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid sequence index");
}

TEST(SequenceOpsTest, SequenceEraseInvalidNegativeIdx) {
  OpTester test("SequenceErase", 11);
  SeqTensors<int64_t> input;
  input.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  input.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  input.AddTensor({2, 2}, {2, 4, 6, 8});
  test.AddSeqInput("S", input);
  test.AddInput<int64_t>("I", {}, {-99});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  input.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  input.AddTensor({2, 2}, {2, 4, 6, 8});
  test.AddSeqOutput("S2", output);
  test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid sequence index");
}

// SequenceConstruct
TEST(SequenceOpsTest, SequenceConstructPositive) {
  OpTester test("SequenceConstruct", 11);
  test.AddInput<int64_t>("input_1", {3, 2}, {1, 2, 3, 4, 5, 6});
  test.AddInput<int64_t>("input_2", {3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddInput<int64_t>("input_3", {3, 2}, {10, 20, 30, 40, 50, 60});

  SeqTensors<int64_t> output;
  output.AddTensor({3, 2}, {1, 2, 3, 4, 5, 6});
  output.AddTensor({3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  output.AddTensor({3, 2}, {10, 20, 30, 40, 50, 60});
  test.AddSeqOutput("S2", output);
  test.Run();
}
}  // namespace test
}  // namespace onnxruntime