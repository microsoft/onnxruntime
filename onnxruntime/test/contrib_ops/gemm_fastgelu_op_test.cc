// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/util/math.h"
#include <gtest/gtest.h>
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "core/platform/threadpool.h"
#include "core/util/thread_utils.h"

using namespace onnxruntime::test;

namespace onnxruntime {
namespace test {
namespace gemmfastgelu {

#if defined(USE_ROCM)
static void RunGemmFastGeluGpuTest(const std::vector<float>& input_data, const std::vector<float>& weight_data,
                               const std::vector<float>& bias_data, const std::vector<float>& output_data,
                               const std::vector<int64_t>& input_dims, const std::vector<int64_t>& weight_dims,
                               const std::vector<int64_t>& bias_dims, const std::vector<int64_t>& output_dims,
                               bool has_bias, bool use_float16 = false) {
  OpTester tester("GemmFastGelu", 1, onnxruntime::kMSDomain);

  if (use_float16) {
    tester.AddInput<MLFloat16>("X", input_dims, ToFloat16(input_data));
    tester.AddInput<MLFloat16>("W", weight_dims, ToFloat16(weight_data));
    if (has_bias) {
      tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
    }
    tester.AddOutput<MLFloat16>("Y", output_dims, ToFloat16(output_data));
  } else {
    tester.AddInput<float>("X", input_dims, input_data);
    tester.AddInput<float>("W", weight_dims, weight_data);
    if (has_bias) {
      tester.AddInput<float>("bias", bias_dims, bias_data);
    }
    tester.AddOutput<float>("Y", output_dims, output_data);
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultRocmExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

TEST(GemmFastGeluTest, GemmFastGeluWithoutBiasFloat32) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int dense_size = 6;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.7f, -0.5f, 0.7f, 1.2f,
      0.3f, 0.1f, 0.8f, -1.6f,
      0.9f, -0.1f, 3.0f, 2.f,
      0.4f, -0.7f, -0.3f, 0.6f};

  std::vector<float> bias_data = {};

  std::vector<float> output_data = {
    3.4894,  1.8455,  0.0260,  0.2229, -0.1003,  0.0902,
    -0.1323, -0.0953,  0.0778,  0.2152,  0.6715, -0.0240
  };


  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> weight_dims = {hidden_size, dense_size};
  std::vector<int64_t> bias_dims = {dense_size};
  std::vector<int64_t> output_dims = {batch_size, sequence_length, dense_size};
#if defined(USE_ROCM)
  RunGemmFastGeluGpuTest(input_data, weight_data, bias_data, output_data, input_dims, weight_dims, bias_dims, output_dims, false);
#endif
}

TEST(GemmFastGeluTest, GemmFastGeluWithBiasFloat32) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int dense_size = 6;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.7f, -0.5f, 0.7f, 1.2f,
      0.3f, 0.1f, 0.8f, -1.6f,
      0.9f, -0.1f, 3.0f, 2.f,
      0.4f, -0.7f, -0.3f, 0.6f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, -0.6f, 0.4f};

  std::vector<float> output_data = {
    2.9862,  2.4849,  1.1177,  2.4329, -0.1681,  0.3988,
    -0.0702, -0.1633,  1.2190,  2.4225,  0.1428,  0.2229
  };

   std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> weight_dims = {hidden_size, dense_size};
  std::vector<int64_t> bias_dims = {dense_size};
  std::vector<int64_t> output_dims = {batch_size, sequence_length, dense_size};
#if defined(USE_ROCM)
  RunGemmFastGeluGpuTest(input_data, weight_data, bias_data, output_data, input_dims, weight_dims, bias_dims, output_dims, true);
#endif
}


// CUDA and ROCm only for Float16 and BFloat16 type.
#if defined(USE_ROCM)
TEST(GemmFastGeluTest, GemmFastGeluWithoutBiasFloat16) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int dense_size = 6;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.7f, -0.5f, 0.7f, 1.2f,
      0.3f, 0.1f, 0.8f, -1.6f,
      0.9f, -0.1f, 3.0f, 2.f,
      0.4f, -0.7f, -0.3f, 0.6f};

  std::vector<float> bias_data = {};

  std::vector<float> output_data = {
    3.4902,  1.8467,  0.0259,  0.2227, -0.1005,  0.0901,
    -0.1324, -0.0955,  0.0778,  0.2156,  0.6714, -0.0241
  };


  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> weight_dims = {hidden_size, dense_size};
  std::vector<int64_t> bias_dims = {dense_size};
  std::vector<int64_t> output_dims = {batch_size, sequence_length, dense_size};
  RunGemmFastGeluGpuTest(input_data, weight_data, bias_data, output_data, input_dims, weight_dims, bias_dims, output_dims, false);
}

TEST(GemmFastGeluTest, GemmFastGeluWithBiasFloat16) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int dense_size = 6;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.7f, -0.5f, 0.7f, 1.2f,
      0.3f, 0.1f, 0.8f, -1.6f,
      0.9f, -0.1f, 3.0f, 2.f,
      0.4f, -0.7f, -0.3f, 0.6f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, -0.6f, 0.4f};

  std::vector<float> output_data = {
    2.9883,  2.4844,  1.1182,  2.4316, -0.1680,  0.3984,
    -0.0701, -0.1633,  1.2178,  2.4219,  0.1426,  0.2227
  };

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> weight_dims = {hidden_size, dense_size};
  std::vector<int64_t> bias_dims = {dense_size};
  std::vector<int64_t> output_dims = {batch_size, sequence_length, dense_size};
  RunGemmFastGeluGpuTest(input_data, weight_data, bias_data, output_data, input_dims, weight_dims, bias_dims, output_dims, true);
}

TEST(GemmFastGeluTest, GemmFastGeluWithBias_BFloat16) {
  OpTester tester("GemmFastGelu", 1, onnxruntime::kMSDomain);

  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int dense_size = 6;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.7f, -0.5f, 0.7f, 1.2f,
      0.3f, 0.1f, 0.8f, -1.6f,
      0.9f, -0.1f, 3.0f, 2.f,
      0.4f, -0.7f, -0.3f, 0.6f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, -0.6f, 0.4f};

  std::vector<float> output_data = {
    2.9883,  2.4844,  1.1182,  2.4316, -0.1680,  0.3984,
    -0.0701, -0.1633,  1.2178,  2.4219,  0.1426,  0.2227
  };

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> weight_dims = {hidden_size, dense_size};
  std::vector<int64_t> bias_dims = {dense_size};
  std::vector<int64_t> output_dims = {batch_size, sequence_length, dense_size};

  std::vector<BFloat16> f_X = FloatsToBFloat16s(input_data);
  std::vector<BFloat16> f_W = FloatsToBFloat16s(weight_data);
  std::vector<BFloat16> f_B = FloatsToBFloat16s(bias_data);
  std::vector<BFloat16> f_Y = FloatsToBFloat16s(output_data);

  tester.AddInput<BFloat16>("X", input_dims, f_X);
  tester.AddInput<BFloat16>("W", weight_dims, f_W);
  tester.AddInput<BFloat16>("bias", bias_dims, f_B);
  tester.AddOutput<BFloat16>("Y", output_dims, f_Y);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultRocmExecutionProvider());
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

}  // namespace gemmfastgelu
}  // namespace test
}  // namespace onnxruntime
