// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

#if defined(USE_CUDA) || defined(USE_ROCM)

TEST(PadAndUnflattenTest, FloatType1D) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f};
  std::vector<int64_t> indices = {1, 3, 5, 7, 9, 11};
  std::vector<int64_t> unflatten_dims = {5, 3};

  std::vector<float> output = {0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f,
                               0.0f, 5.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f};

  OpTester test("PadAndUnflatten", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("input", {6}, input);
  test.AddInput<int64_t>("indices", {6}, indices);
  test.AddInput<int64_t>("unflatten_dims", {2}, unflatten_dims);
  test.AddOutput<float>("output", {5, 3}, output);
  test.Run();
}

TEST(PadAndUnflattenTest, FloatType2D) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f, 7.f, 8.f, 9.f};
  std::vector<int64_t> indices = {1, 3, 4};
  std::vector<int64_t> unflatten_dims = {2, 3};

  std::vector<float> output = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f,
                               4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 0.0f, 0.0f, 0.0f};

  OpTester test("PadAndUnflatten", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("input", {3, 3}, input);
  test.AddInput<int64_t>("indices", {3}, indices);
  test.AddInput<int64_t>("unflatten_dims", {2}, unflatten_dims);
  test.AddOutput<float>("output", {2, 3, 3}, output);
  test.Run();
}

TEST(PadAndUnflattenTest, MLFloat16Type1D) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f};
  std::vector<int64_t> indices = {1, 3, 5, 7, 9, 11};
  std::vector<int64_t> unflatten_dims = {5, 3};

  std::vector<float> output = {0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f,
                               0.0f, 5.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f};

  std::vector<MLFloat16> input_half;
  input_half.resize(input.size());
  ConvertFloatToMLFloat16(input.data(), input_half.data(), int(input.size()));
  std::vector<MLFloat16> output_half;
  output_half.resize(output.size());
  ConvertFloatToMLFloat16(output.data(), output_half.data(), int(output.size()));

  OpTester test("PadAndUnflatten", 1, onnxruntime::kMSDomain);
  test.AddInput<MLFloat16>("input", {6}, input_half);
  test.AddInput<int64_t>("indices", {6}, indices);
  test.AddInput<int64_t>("unflatten_dims", {2}, unflatten_dims);
  test.AddOutput<MLFloat16>("output", {5, 3}, output_half);
  test.Run();
}

TEST(PadAndUnflattenTest, MLFloat16Type2D) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f, 7.f, 8.f, 9.f};
  std::vector<int64_t> indices = {1, 3, 4};
  std::vector<int64_t> unflatten_dims = {2, 3};

  std::vector<float> output = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f,
                               4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 0.0f, 0.0f, 0.0f};

  std::vector<MLFloat16> input_half;
  input_half.resize(input.size());
  ConvertFloatToMLFloat16(input.data(), input_half.data(), int(input.size()));
  std::vector<MLFloat16> output_half;
  output_half.resize(output.size());
  ConvertFloatToMLFloat16(output.data(), output_half.data(), int(output.size()));

  OpTester test("PadAndUnflatten", 1, onnxruntime::kMSDomain);
  test.AddInput<MLFloat16>("input", {3, 3}, input_half);
  test.AddInput<int64_t>("indices", {3}, indices);
  test.AddInput<int64_t>("unflatten_dims", {2}, unflatten_dims);
  test.AddOutput<MLFloat16>("output", {2, 3, 3}, output_half);
  test.Run();
}

#endif

}  // namespace test
}  // namespace onnxruntime
