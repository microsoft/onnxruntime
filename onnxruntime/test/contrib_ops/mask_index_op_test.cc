// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template<typename T>
void RunTest(
    const std::vector<T>& mask_data,
    const std::vector<int32_t>& mask_index_data,
    int batch_size,
    int sequence_length) {
  if (HasCudaEnvironment(0)) { // MaskIndex has only Cuda implementation right now.
    // Input and output shapes
    std::vector<int64_t> mask_dims = {batch_size, sequence_length};
    std::vector<int64_t> mask_index_dims = {batch_size};

    OpTester tester("MaskIndex", 1, onnxruntime::kMSDomain);
    tester.AddInput<T>("mask", mask_dims, mask_data);
    tester.AddOutput<int32_t>("mask_index", mask_index_dims, mask_index_data);

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(MaskIndexTest, MaskIndexTest_batch1_int32) {
  int batch_size = 1;
  int sequence_length = 2;

  std::vector<int32_t> mask_data = {
      1, 1};

  std::vector<int32_t> mask_index_data = {
      2};

  RunTest(mask_data,
          mask_index_data,
          batch_size,
          sequence_length);
}

TEST(MaskIndexTest, MaskIndexTest_batch1_int64) {
  int batch_size = 1;
  int sequence_length = 2;

  std::vector<int64_t> mask_data = {
      1, 1};

  std::vector<int32_t> mask_index_data = {
      2};

  RunTest(mask_data,
          mask_index_data,
          batch_size,
          sequence_length);
}

TEST(MaskIndexTest, MaskIndexTest_batch3_int32) {
  int batch_size = 3;
  int sequence_length = 2;

  std::vector<int32_t> mask_data = {
      1, 1,
      1, 1,
      1, 0};

  std::vector<int32_t> mask_index_data = {
      2, 2, 1};

  RunTest(mask_data,
          mask_index_data,
          batch_size,
          sequence_length);
}


TEST(MaskIndexTest, MaskIndexTest_batch5_int64) {
  int batch_size = 5;
  int sequence_length = 2;

  std::vector<int64_t> mask_data = {
      0, 0,  // All 0
      1, 1,  // All 1
      1, 0,
      1, 1,
      1, 0};

  std::vector<int32_t> mask_index_data = {
      0, 2, 1, 2, 1};

  RunTest(mask_data,
          mask_index_data,
          batch_size,
          sequence_length);
}
}  // namespace test
}  // namespace onnxruntime
