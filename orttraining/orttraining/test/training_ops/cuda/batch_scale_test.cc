// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(USE_CUDA) || defined(USE_ROCM)

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void PrepareInputAndOutputData(const std::vector<float>& input,
                                      const std::vector<float>& scales,
                                      std::vector<std::vector<float>>& outputs) {
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs.at(i).resize(input.size());
  }

  for (size_t i = 0; i < input.size(); ++i) {
    outputs[0][i] = input[i] * scales[0];
    outputs[1][i] = input[i] * scales[1];
    if (outputs.size() == 3)
      outputs[2][i] = input[i] * scales[2];
  }
}

template <typename T>
static void RunBatchScaleOpTester(const std::vector<T>& input,
                                  const std::vector<float>& scales,
                                  const std::vector<std::vector<T>>& outputs,
                                  const std::vector<int64_t>& shape) {
  ORT_ENFORCE(scales.size() == outputs.size(), "scales and outputs should have the same size.");
  OpTester test("BatchScale", 1, onnxruntime::kMSDomain);
  test.AddInput<T>("input", shape, input);
  test.AddOutput<T>("output_0", shape, outputs[0]);
  test.AddOutput<T>("output_1", shape, outputs[1]);
  if (outputs.size() == 3) {
    test.AddOutput<T>("output_2", shape, outputs[2]);
  }
  test.AddAttribute<float>("scale_0", scales[0]);
  test.AddAttribute<float>("scale_1", scales[1]);
  if (scales.size() == 3) {
    test.AddAttribute<float>("scale_2", scales[2]);
  }

  // Exclude CPU EP since it is not implemented yet.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider});
}

static void RunBatchScaleTestWithFloatAndMLFloat16(const std::vector<float>& input,
                                                   const std::vector<float>& scales,
                                                   const std::vector<int64_t>& shape) {
  std::vector<std::vector<float>> outputs;
  outputs.resize(scales.size());
  PrepareInputAndOutputData(input, scales, outputs);
  RunBatchScaleOpTester(input, scales, outputs, shape);

  std::vector<MLFloat16> input_half;
  input_half.resize(input.size());
  ConvertFloatToMLFloat16(input.data(), input_half.data(), static_cast<int>(input.size()));

  std::vector<std::vector<MLFloat16>> outputs_half;
  outputs_half.resize(scales.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs_half[i].resize(outputs[i].size());
    ConvertFloatToMLFloat16(outputs[i].data(), outputs_half[i].data(), static_cast<int>(outputs[i].size()));
  }

  RunBatchScaleOpTester(input_half, scales, outputs_half, shape);
}

TEST(BatchScaleTest, SmallTensor1D) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f};
  float scale_0 = 0.25f;
  float scale_1 = 0.25f;
  float scale_2 = 0.5f;
  std::vector<int64_t> shape{static_cast<int64_t>(input.size())};
  RunBatchScaleTestWithFloatAndMLFloat16(input, {scale_0, scale_1, scale_2}, shape);
  RunBatchScaleTestWithFloatAndMLFloat16(input, {scale_0, scale_1}, shape);
}

TEST(BatchScaleTest, SmallTensorVectorized1D) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f, 7.0f, 8.0f,
                              9.0f, 10.0f, 11.0f, 12.f, 13.0f, 14.0f, 15.0f, 16.0f,
                              17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.f, 23.0f, 24.0f,
                              25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.f, 31.0f, 32.0f};
  float scale_0 = 0.25f;
  float scale_1 = 0.25f;
  float scale_2 = 0.5f;
  std::vector<int64_t> shape{static_cast<int64_t>(input.size())};
  RunBatchScaleTestWithFloatAndMLFloat16(input, {scale_0, scale_1, scale_2}, shape);
  RunBatchScaleTestWithFloatAndMLFloat16(input, {scale_0, scale_1}, shape);
}

TEST(BatchScaleTest, SmallTensor2D) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f, 7.f, 8.f, 9.f};
  float scale_0 = 0.25f;
  float scale_1 = 0.25f;
  float scale_2 = 0.5f;
  std::vector<int64_t> shape{3, 3};
  RunBatchScaleTestWithFloatAndMLFloat16(input, {scale_0, scale_1, scale_2}, shape);
  RunBatchScaleTestWithFloatAndMLFloat16(input, {scale_0, scale_1}, shape);
}

TEST(BatchScaleTest, SmallTensorVectorized2D) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f, 7.0f, 8.0f,
                              9.0f, 10.0f, 11.0f, 12.f, 13.0f, 14.0f, 15.0f, 16.0f,
                              17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.f, 23.0f, 24.0f,
                              25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.f, 31.0f, 32.0f};
  float scale_0 = 0.25f;
  float scale_1 = 0.25f;
  float scale_2 = 0.5f;
  std::vector<int64_t> shape{4, 8};
  RunBatchScaleTestWithFloatAndMLFloat16(input, {scale_0, scale_1, scale_2}, shape);
  RunBatchScaleTestWithFloatAndMLFloat16(input, {scale_0, scale_1}, shape);
}

}  // namespace test
}  // namespace onnxruntime

#endif
