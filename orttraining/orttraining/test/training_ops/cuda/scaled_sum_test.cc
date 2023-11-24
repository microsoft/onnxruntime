// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_CUDA) || defined(USE_ROCM)

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void PrepareInputAndOutputData(const std::vector<std::vector<float>>& input,
                                      const std::vector<float>& scales,
                                      std::vector<float>& output) {
  output.resize(input[0].size());
  size_t scale_size = scales.size();
  for (size_t i = 0; i < input[0].size(); ++i) {
    output[i] = input[0][i] * scales[0] + input[1][i] * scales[1] + (scale_size == 3 ? input[2][i] * scales[2] : 0.0f);
  }
}

template <typename T>
static void RunScaledSumOpTester(const std::vector<std::vector<T>>& inputs,
                                 const std::vector<float>& scales,
                                 const std::vector<T>& output,
                                 const std::vector<int64_t>& shape) {
  OpTester test("ScaledSum", 1, onnxruntime::kMSDomain);
  test.AddInput<T>("input0", shape, inputs[0]);
  test.AddInput<T>("input1", shape, inputs[1]);
  if (scales.size() == 3) {
    test.AddInput<T>("input2", shape, inputs[2]);
  }

  test.AddOutput<T>("output", shape, output);
  test.AddAttribute<float>("scale_0", scales[0]);
  test.AddAttribute<float>("scale_1", scales[1]);
  if (scales.size() == 3) {
    test.AddAttribute<float>("scale_2", scales[2]);
  }

  // Exclude CPU EP since it is not implemented yet.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider});
}

static void RunScaledSumWithFloatAndMLFloat16(const std::vector<std::vector<float>>& inputs,
                                              const std::vector<float>& scales,
                                              const std::vector<int64_t>& shape) {
  std::vector<float> output;
  PrepareInputAndOutputData(inputs, scales, output);
  RunScaledSumOpTester(inputs, scales, output, shape);

  std::vector<std::vector<MLFloat16>> inputs_half;
  inputs_half.resize(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    inputs_half[i].resize(inputs[i].size());
    ConvertFloatToMLFloat16(inputs[i].data(), inputs_half[i].data(), static_cast<int>(inputs[i].size()));
  }

  std::vector<MLFloat16> output_half;
  output_half.resize(output.size());
  ConvertFloatToMLFloat16(output.data(), output_half.data(), static_cast<int>(output.size()));

  RunScaledSumOpTester(inputs_half, scales, output_half, shape);
}

TEST(ScaledSumTest, SmallTensor1D) {
  std::vector<std::vector<float>> inputs = {{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f},
                                            {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f},
                                            {0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f}};

  float scale_0 = 0.25f;
  float scale_1 = 0.25f;
  float scale_2 = 0.5f;

  std::vector<int64_t> shape{static_cast<int64_t>(inputs[0].size())};
  RunScaledSumWithFloatAndMLFloat16(inputs, {scale_0, scale_1, scale_2}, shape);

  RunScaledSumWithFloatAndMLFloat16(inputs, {scale_0, scale_1}, shape);
}  // namespace test

TEST(ScaledSumTest, SmallTensorVectorized1D) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f, 7.0f, 8.0f,
                              9.0f, 10.0f, 11.0f, 12.f, 13.0f, 14.0f, 15.0f, 16.0f,
                              17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.f, 23.0f, 24.0f,
                              25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.f, 31.0f, 32.0f};
  std::vector<std::vector<float>> inputs{input, input, input};
  float scale_0 = 0.25f;
  float scale_1 = 0.25f;
  float scale_2 = 0.5f;

  std::vector<int64_t> shape{static_cast<int64_t>(input.size())};
  RunScaledSumWithFloatAndMLFloat16(inputs, {scale_0, scale_1, scale_2}, shape);

  RunScaledSumWithFloatAndMLFloat16(inputs, {scale_0, scale_1}, shape);
}

TEST(ScaledSumTest, SmallTensor2D) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f, 7.f, 8.f, 9.f};
  std::vector<std::vector<float>> inputs{input, input, input};
  float scale_0 = 0.25f;
  float scale_1 = 0.25f;
  float scale_2 = 0.5f;

  std::vector<int64_t> shape{3, 3};
  RunScaledSumWithFloatAndMLFloat16(inputs, {scale_0, scale_1, scale_2}, shape);

  RunScaledSumWithFloatAndMLFloat16(inputs, {scale_0, scale_1}, shape);
}

TEST(ScaledSumTest, SmallTensorVectorized2D) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f, 7.0f, 8.0f,
                              9.0f, 10.0f, 11.0f, 12.f, 13.0f, 14.0f, 15.0f, 16.0f,
                              17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.f, 23.0f, 24.0f,
                              25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.f, 31.0f, 32.0f};
  std::vector<std::vector<float>> inputs{input, input, input};
  float scale_0 = 0.25f;
  float scale_1 = 0.25f;
  float scale_2 = 0.5f;

  std::vector<int64_t> shape{4, 8};
  RunScaledSumWithFloatAndMLFloat16(inputs, {scale_0, scale_1, scale_2}, shape);

  RunScaledSumWithFloatAndMLFloat16(inputs, {scale_0, scale_1}, shape);
}

}  // namespace test
}  // namespace onnxruntime

#endif
