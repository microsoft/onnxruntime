// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

struct ScaleInputOutput {
  ScaleInputOutput() {
    input_half.resize(input.size());
    output_half.resize(output.size());
    scale_half.resize(scale_float.size());
    ConvertFloatToMLFloat16(input.data(), input_half.data(), int(input.size()));
    ConvertFloatToMLFloat16(output.data(), output_half.data(), int(output.size()));
    ConvertFloatToMLFloat16(scale_float.data(), scale_half.data(), int(scale_float.size()));
  }

  // Fp32 Inputs/Output
  std::vector<float> scale_float = {2.0f};
  std::vector<int64_t> scale_int64 = {2LL};
  std::vector<float> input = {1.0f, 2.0f, 3.0f};
  std::vector<float> output = {0.5f, 1.0f, 1.5f};

  // Fp16 Inputs/Outputs
  std::vector<MLFloat16> input_half;
  std::vector<MLFloat16> output_half;
  std::vector<MLFloat16> scale_half;
};

TEST(CudaKernelTest, ScaleFloatFloat) {
  ScaleInputOutput data;
  OpTester test("Scale", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("input", {3}, data.input);
  test.AddInput<float>("scale", {1}, data.scale_float);
  test.AddOutput<float>("output", {3}, data.output);
  test.Run();
}

TEST(CudaKernelTest, ScaleFloatInt64) {
  ScaleInputOutput data;
  OpTester test("Scale", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("input", {3}, data.input);
  test.AddInput<int64_t>("scale", {1}, data.scale_int64);
  test.AddOutput<float>("output", {3}, data.output);
  test.Run();
}

TEST(CudaKernelTest, ScaleHalfHalf) {
  ScaleInputOutput data;
  OpTester test("Scale", 1, onnxruntime::kMSDomain);
  test.AddInput<MLFloat16>("input", {3}, data.input_half);
  test.AddInput<MLFloat16>("scale", {1}, data.scale_half);
  test.AddOutput<MLFloat16>("output", {3}, data.output_half);
  test.Run();
}

TEST(CudaKernelTest, ScaleHalfInt64) {
  ScaleInputOutput data;
  OpTester test("Scale", 1, onnxruntime::kMSDomain);
  test.AddInput<MLFloat16>("input", {3}, data.input_half);
  test.AddInput<int64_t>("scale", {1}, data.scale_int64);
  test.AddOutput<MLFloat16>("output", {3}, data.output_half);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime