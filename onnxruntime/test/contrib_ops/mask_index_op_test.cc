// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
void RunTest(
    const std::vector<T>& mask_data,
    const std::vector<int32_t>& mask_index_data,
    int batch_size,
    int sequence_length,
    bool invalid_mask = false) {
  bool enable_cuda = HasCudaEnvironment(0);
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get());

  if (enable_cuda || enable_cpu) {
    // Input and output shapes
    std::vector<int64_t> mask_dims = {batch_size, sequence_length};
    std::vector<int64_t> mask_index_dims = {2 * batch_size};

    OpTester tester("MaskIndex", 1, onnxruntime::kMSDomain);
    tester.AddInput<T>("mask", mask_dims, mask_data);
    tester.AddOutput<int32_t>("mask_index", mask_index_dims, mask_index_data);

    if (enable_cuda) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_cpu) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCpuExecutionProvider());
      if (invalid_mask) {
        tester.Run(OpTester::ExpectResult::kExpectFailure, "", {}, nullptr, &execution_providers);
      } else {
        tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
      }
    }
  }
}

TEST(MaskIndexTest, MaskIndexTest_batch1_int64) {
  int batch_size = 1;
  int sequence_length = 2;

  std::vector<int64_t> mask_data = {
      1, 1};

  std::vector<int32_t> mask_index_data = {
      2, 0};

  RunTest(mask_data,
          mask_index_data,
          batch_size,
          sequence_length);
}

TEST(MaskIndexTest, MaskIndexTest_batch4_int32) {
  int batch_size = 4;
  int sequence_length = 2;

  std::vector<int32_t> mask_data = {
      0, 0,   // All 0
      1, 1,   // All 1
      1, 0,   // Right padding
      0, 1};  // Left padding

  std::vector<int32_t> mask_index_data = {
      0, 2, 1, 2,
      2, 0, 0, 1};

  RunTest(mask_data,
          mask_index_data,
          batch_size,
          sequence_length);
}

TEST(MaskIndexTest, MaskIndexTest_batch4_int64) {
  int batch_size = 4;
  int sequence_length = 2;

  std::vector<int64_t> mask_data = {
      0, 0,   // All 0
      1, 1,   // All 1
      1, 0,   // Right padding
      0, 1};  // Left padding

  std::vector<int32_t> mask_index_data = {
      0, 2, 1, 2,
      2, 0, 0, 1};

  RunTest(mask_data,
          mask_index_data,
          batch_size,
          sequence_length);
}

TEST(MaskIndexTest, MaskIndexTest_float) {
  int batch_size = 4;
  int sequence_length = 3;

  std::vector<float> mask_data = {
      0.0, 0.0, 0.0,  // All 0
      1.0, 1.0, 1.0,  // All 1
      1.0, 0.0, 0.0,  // Right padding
      0.0, 1.0, 1.0   // Left padding
  };

  std::vector<int32_t> mask_index_data = {
      0, 3, 1, 3,
      3, 0, 0, 1};

  RunTest(mask_data,
          mask_index_data,
          batch_size,
          sequence_length);
}

TEST(MaskIndexTest, MaskIndexTest_invalid_mask) {
  int batch_size = 1;
  int sequence_length = 3;

  std::vector<float> mask_data = {
      1.0, 0.0, 1.0  // Middle padding (invalid)
  };

  std::vector<int32_t> mask_index_data = {
      3, 0};  // Expected result for cuda.

  RunTest(mask_data,
          mask_index_data,
          batch_size,
          sequence_length,
          true);  // For CPU, expect failuare.
}
}  // namespace test
}  // namespace onnxruntime
