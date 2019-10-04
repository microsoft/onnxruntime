// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

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

}  // namespace test
}  // namespace onnxruntime