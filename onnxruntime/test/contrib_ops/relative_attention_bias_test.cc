// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void RunRelAttnBiasTest(
    const std::vector<float>& bias_table,           // Shape = [num_buckets, num_heads]
    const std::vector<int64_t>& sequence_length,    // Shape = []
    const std::vector<float>& output_data,          // Shape = [1, num_heads, sequence_length, sequence_length]
    int max_distance,
    int num_buckets,
    int num_heads,
    int seq_len,
    bool is_bidirectional,
    bool use_float16 = false) {
  int min_cuda_architecture = use_float16 ? 530 : 0;

  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = false;
  if (enable_cpu || enable_cuda) {
    OpTester tester("RelPosAttnBiasGen", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("max_distance", static_cast<int64_t>(max_distance));
    tester.AddAttribute<int64_t>("is_bidirectional", static_cast<int64_t>(is_bidirectional));

    std::vector<int64_t> bias_table_dims = {num_buckets, num_heads};
    std::vector<int64_t> sequence_length_dims = {1};
    std::vector<int64_t> output_dims = {1, num_heads, seq_len, seq_len};

    if (use_float16) {
      tester.AddInput<MLFloat16>("bias_table", bias_table_dims, ToFloat16(bias_table));
      tester.AddInput<int64_t>("sequence_length", sequence_length_dims, sequence_length);
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    } else {
      tester.AddInput<float>("bias_table", bias_table_dims, bias_table);
      tester.AddInput<int64_t>("sequence_length", sequence_length_dims, sequence_length);
      tester.AddOutput<float>("output", output_dims, output_data);
    }

    if (enable_cuda) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_cpu) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCpuExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
}

TEST(RelationalAttentionBiasTest, RelationalAttentionBiasTest_FP32) {
  int max_distance = 8;
  int num_buckets = 2;
  int num_heads = 2;
  int seq_len = 2;
  bool is_bidirectional = true;

  std::vector<float> bias_table = {3.5f, 0.453125f, 3.1875f, 0.97265625f};
  std::vector<int64_t> sequence_length = {2};

  std::vector<float> output_data = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  RunRelAttnBiasTest(bias_table,
                     sequence_length,
                     output_data,
                     max_distance,
                     num_buckets,
                     num_heads,
                     seq_len,
                     is_bidirectional);
}


}  // namespace test
}  // namespace onnxruntime
