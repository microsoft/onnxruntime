// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"

namespace onnxruntime {
namespace test {

static void SequencePoolingTest(
  const std::vector<float>& batch_input,
  const std::vector<int64_t>& batch_sequence_lengths,
  const std::vector<float>& output,
  const std::vector<float>& masks,
  const int batch_size,
  const int sequence_length_for_split,
  const int hidden_size,
  const int num_sequence) {

  OpTester tester("SequencePooling", 1, onnxruntime::kMSDomain);

  tester.AddInput<float>("batch_input_tensor", {batch_size, sequence_length_for_split, hidden_size}, batch_input);
  tester.AddInput<int64_t>("batch_sentence_lengthes", {batch_size, num_sequence}, batch_sequence_lengths);
  tester.AddOutput<float>("output", {batch_size, num_sequence, hidden_size}, output);
  tester.AddOutput<float>("masks", {batch_size, num_sequence}, masks);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  execution_providers.push_back(DefaultCudaExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);

  if (HasCudaEnvironment(530 /*min_cuda_architecture*/)) {
    OpTester tester_1("SequencePooling", 1, onnxruntime::kMSDomain);

    tester_1.AddInput<MLFloat16>("batch_input_tensor", {batch_size, sequence_length_for_split, hidden_size}, ToFloat16(batch_input));
    tester_1.AddInput<int64_t>("batch_sentence_lengthes", {batch_size, num_sequence}, batch_sequence_lengths);
    tester_1.AddOutput<MLFloat16>("output", {batch_size, num_sequence, hidden_size}, ToFloat16(output));
    tester_1.AddOutput<MLFloat16>("masks", {batch_size, num_sequence}, ToFloat16(masks));

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    tester_1.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}


TEST(SequencePoolingTest, Test_1) {
  std::vector<float> batch_input{1.0f, 2.0f, 3.0f, 3.0f, 5.0f, 5.0f, 4.0f, 3.0f, 6.0f,
                                 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                                 1.0f, 2.0f, 3.0f, 3.0f, 5.0f, 5.0f, 4.0f, 3.0f, 6.0f,
                                 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<int64_t> batch_sequence_lengths{1, 2, 3, 1, 2, 3};
  std::vector<float> output{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                            1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<float> masks{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  SequencePoolingTest(batch_input, batch_sequence_lengths, output, masks, 2, 6, 3, 3);
}

}  // namespace test
}  // namespace onnxruntime
