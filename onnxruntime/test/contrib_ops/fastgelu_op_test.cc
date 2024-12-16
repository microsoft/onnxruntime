// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/dnnl_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

using namespace onnxruntime::test;

namespace onnxruntime {
namespace test {

const std::vector<float> ComputeGelu(const std::vector<float>& input_data) {
  std::vector<float> output;
  output.reserve(input_data.size());

  for (size_t i = 0; i < input_data.size(); i++) {
    float x = input_data[i];
    float y = x * (0.5f + 0.5f * tanh(x * (0.035677408136300125f * x * x + 0.7978845608028654f)));
    output.push_back(y);
  }
  return output;
}

const std::vector<float> AddBias(const std::vector<float>& input_data, const std::vector<float>& bias_data) {
  size_t bias_length = bias_data.size();

  std::vector<float> output;
  output.reserve(input_data.size());

  for (size_t i = 0; i < input_data.size(); i++) {
    output.push_back(input_data[i] + bias_data[i % bias_length]);
  }
  return output;
}

const std::vector<float> GetExpectedResult(const std::vector<float>& input_data, const std::vector<float>& bias_data) {
  std::vector<float> add_bias_data = AddBias(input_data, bias_data);
  return ComputeGelu(add_bias_data);
}

#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_WEBGPU)
static void RunFastGeluGpuTest(const std::vector<float>& input_data, const std::vector<float>& bias_data,
                               const std::vector<float>& output_data, const std::vector<int64_t>& input_dims,
                               const std::vector<int64_t>& bias_dims, const std::vector<int64_t>& output_dims,
                               bool has_bias = true, bool use_float16 = false) {
#ifdef USE_CUDA
  // Test CUDA operator.
  int min_cuda_architecture = use_float16 ? 530 : 0;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support FP16";
    return;
  }
#endif
  OpTester tester("FastGelu", 1, onnxruntime::kMSDomain);

  if (use_float16) {
    tester.AddInput<MLFloat16>("X", input_dims, ToFloat16(input_data));
    if (has_bias) {
      tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
    }
    tester.AddOutput<MLFloat16>("Y", output_dims, ToFloat16(output_data));
  } else {
    tester.AddInput<float>("X", input_dims, input_data);
    if (has_bias) {
      tester.AddInput<float>("bias", bias_dims, bias_data);
    }
    tester.AddOutput<float>("Y", output_dims, output_data);
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  execution_providers.push_back(DefaultRocmExecutionProvider());
#elif USE_WEBGPU
  execution_providers.push_back(DefaultWebGpuExecutionProvider());
#endif
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

#if defined(USE_DNNL)
static void RunFastGeluTest_bf16(const std::vector<float>& input_data, const std::vector<float>& bias_data,
                                 const std::vector<float>& output_data, const std::vector<int64_t>& input_dims,
                                 const std::vector<int64_t>& bias_dims, const std::vector<int64_t>& output_dims,
                                 bool has_bias = true) {
  OpTester tester("FastGelu", 1, onnxruntime::kMSDomain);

  tester.AddInput<BFloat16>("X", input_dims, FloatsToBFloat16s(input_data));
  if (has_bias) {
    tester.AddInput<BFloat16>("bias", bias_dims, FloatsToBFloat16s(bias_data));
  }
  tester.AddOutput<BFloat16>("Y", output_dims, FloatsToBFloat16s(output_data));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  //  USE_DNNL

static void RunFastGeluCpuTest(const std::vector<float>& input_data, const std::vector<float>& bias_data,
                               const std::vector<float>& output_data, const std::vector<int64_t>& input_dims,
                               const std::vector<int64_t>& bias_dims, const std::vector<int64_t>& output_dims,
                               bool has_bias = true) {
  // Test CPU operator: only float32 is implemented for FastGelu CPU.
  if (nullptr != DefaultCpuExecutionProvider().get()) {
    OpTester tester("FastGelu", 1, onnxruntime::kMSDomain);

    tester.AddInput<float>("X", input_dims, input_data);
    if (has_bias) {
      tester.AddInput<float>("bias", bias_dims, bias_data);
    }
    tester.AddOutput<float>("Y", output_dims, output_data);

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
    execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif
    execution_providers.push_back(DefaultCpuExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

// This test simulates Gelu in Bert model for float32
static void RunFastGeluTest(
    const std::vector<float>& input_data,
    const std::vector<float>& bias_data,
    int batch_size,
    int sequence_length,
    int hidden_size) {
  std::vector<float> output_data;

  bool has_bias = (bias_data.size() > 0);
  if (has_bias) {
    output_data = GetExpectedResult(input_data, bias_data);
  } else {
    output_data = ComputeGelu(input_data);
  }
  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> bias_dims = {hidden_size};
  std::vector<int64_t> output_dims = input_dims;
#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_WEBGPU)
  RunFastGeluGpuTest(input_data, bias_data, output_data, input_dims, bias_dims, output_dims, has_bias);
#endif
  RunFastGeluCpuTest(input_data, bias_data, output_data, input_dims, bias_dims, output_dims, has_bias);
}

TEST(FastGeluTest, FastGeluWithNullInput) {
  int batch_size = 1;
  int sequence_length = 0;
  int hidden_size = 4;

  std::vector<float> input_data = {};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f};

  RunFastGeluTest(input_data, bias_data, batch_size, sequence_length, hidden_size);
}
#if defined(USE_DNNL)
TEST(FastGeluTest, FastGeluWithBias_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> output_data = GetExpectedResult(input_data, bias_data);

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> bias_dims = {hidden_size};
  std::vector<int64_t> output_dims = input_dims;

  RunFastGeluTest_bf16(input_data, bias_data, output_data, input_dims, bias_dims, output_dims, true);
}

TEST(FastGeluTest, FastGeluWithoutBias_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> output_data = ComputeGelu(input_data);

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> bias_dims = {hidden_size};
  std::vector<int64_t> output_dims = input_dims;

  RunFastGeluTest_bf16(input_data, bias_data, output_data, input_dims, bias_dims, output_dims, false);
}
#endif  //  USE_DNNL

TEST(FastGeluTest, FastGeluWithBiasFloat32) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f};

  RunFastGeluTest(input_data, bias_data, batch_size, sequence_length, hidden_size);
}

TEST(FastGeluTest, FastGeluWithoutBiasFloat32) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> bias_data = {};

  RunFastGeluTest(input_data, bias_data, batch_size, sequence_length, hidden_size);
}

// CUDA, ROCm and WebGPU only for Float16 type.
#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_WEBGPU)
TEST(FastGeluTest, FastGeluWithBiasFloat16_2) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f,
      0.5f, 0.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f};

  std::vector<float> output_data = {
      0.1851806640625f, 0.054046630859375f,
      0, 0.63037109375f};

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> bias_dims = {hidden_size};
  std::vector<int64_t> output_dims = input_dims;

  RunFastGeluGpuTest(input_data, bias_data, output_data, input_dims, bias_dims, output_dims, true, true);
}

TEST(FastGeluTest, FastGeluWithoutBiasFloat16_2) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f,
      0.5f, 0.2f};

  std::vector<float> bias_data = {};

  std::vector<float> output_data = {
      0.63037109375f, -0.154296875f,
      0.345703125f, 0.11578369140625f};

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> bias_dims = {};
  std::vector<int64_t> output_dims = input_dims;

  RunFastGeluGpuTest(input_data, bias_data, output_data, input_dims, bias_dims, output_dims, false, true);
}

TEST(FastGeluTest, FastGeluWithBiasFloat16_4) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> output_data = {
      0.1851806640625f, 0.054046630859375f, 1.0615234375f, 3.095703125f,
      0, 0.63037109375f, 1.3984375f, 1.3984375f};

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> bias_dims = {hidden_size};
  std::vector<int64_t> output_dims = input_dims;

  RunFastGeluGpuTest(input_data, bias_data, output_data, input_dims, bias_dims, output_dims, true, true);
}

TEST(FastGeluTest, FastGeluWithoutBiasFloat16_4) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> bias_data = {};

  std::vector<float> output_data = {
      0.63037109375f, -0.154296875f, 0.0f, 0.8408203125f,
      0.345703125f, 0.11578369140625f, 0.1854248046875f, -0.1646728515625f};

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> bias_dims = {};
  std::vector<int64_t> output_dims = input_dims;

  RunFastGeluGpuTest(input_data, bias_data, output_data, input_dims, bias_dims, output_dims, false, true);
}

TEST(FastGeluTest, FastGeluWithBiasFloat16_8) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 8;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f, 1.3f, 2.1f, -0.2f, 1.1f,
      0.5f, 0.2f, 0.3f, -0.6f, 3.1f, 2.2f, -1.1f, 0.0f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 1.3f, -1.0f, 0.0f, 3.1f};

  std::vector<float> output_data = {
      0.18537094f, 0.053982764f, 1.061703f, 3.0973732f, 2.5883462f, 0.95058095f, -0.084148578f, 4.1999736f,
      0.0f, 0.63043171f, 1.3995714f, 1.3995714f, 4.3999906f, 1.061703f, -0.14941895f, 3.0973732f};

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> bias_dims = {hidden_size};
  std::vector<int64_t> output_dims = input_dims;

  RunFastGeluGpuTest(input_data, bias_data, output_data, input_dims, bias_dims, output_dims, true, true);
}

TEST(FastGeluTest, FastGeluWithoutBiasFloat16_8) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 8;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f, 1.3f, 2.1f, -0.2f, 1.1f,
      0.5f, 0.2f, 0.3f, -0.6f, 3.1f, 2.2f, -1.1f, 0.0f};

  std::vector<float> bias_data = {};

  std::vector<float> output_data = {
      0.63043171f, -0.15428598f, 0.0f, 0.84119201f, 1.173929f, 2.062669f, -0.084148578f, 0.95058107f,
      0.345714f, 0.11585142f, 0.18537094f, -0.1645848f, 3.0973732f, 2.1696784f, -0.14941895f, 0.0f};

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> bias_dims = {hidden_size};
  std::vector<int64_t> output_dims = input_dims;

  RunFastGeluGpuTest(input_data, bias_data, output_data, input_dims, bias_dims, output_dims, false, true);
}
#endif

// CUDA and ROCm only for BFloat16 type.
#if defined(USE_CUDA) || defined(USE_ROCM)
TEST(FastGeluTest, FastGeluWithBias_BFloat16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester tester("FastGelu", 1, onnxruntime::kMSDomain);

  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;

  std::vector<float> X = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> B = {
      -0.5f, 0.6f, 1.2f, 2.1f};

  std::vector<float> Y = {
      0.1851806640625f, 0.054046630859375f, 1.0615234375f, 3.095703125f,
      0, 0.63037109375f, 1.3984375f, 1.3984375f};

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> bias_dims = {hidden_size};
  std::vector<int64_t> output_dims = input_dims;

  std::vector<BFloat16> f_X = FloatsToBFloat16s(X);
  std::vector<BFloat16> f_B = FloatsToBFloat16s(B);
  std::vector<BFloat16> f_Y = FloatsToBFloat16s(Y);

  tester.AddInput<BFloat16>("X", input_dims, f_X);
  tester.AddInput<BFloat16>("bias", bias_dims, f_B);
  tester.AddOutput<BFloat16>("Y", output_dims, f_Y);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#ifdef USE_CUDA
  execution_providers.push_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  execution_providers.push_back(DefaultRocmExecutionProvider());
#endif
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

}  // namespace test
}  // namespace onnxruntime
