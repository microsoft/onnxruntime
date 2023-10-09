// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

#if defined(USE_CUDA) || defined(USE_ROCM)

TEST(FlattenAndUnpadTest, Int32Type1D) {
  std::vector<int32_t> input = {1, 1, 3, 2, 0, 3, 0, 4,
                                0, 5, 0, 6, 0, 0, 0};
  std::vector<int64_t> indices = {1, 3, 5, 7, 9, 11};

  std::vector<int32_t> output = {1, 2, 3, 4, 5, 6};
  std::vector<int64_t> unflatten_dims = {5, 3};

  OpTester test("FlattenAndUnpad", 1, onnxruntime::kMSDomain);
  test.AddInput<int32_t>("input", {5, 3}, input);
  test.AddInput<int64_t>("indices", {6}, indices);
  test.AddOutput<int32_t>("output", {6}, output);
  test.AddOutput<int64_t>("unflatten_dims", {2}, unflatten_dims);
  test.Run();
}

TEST(FlattenAndUnpadTest, Int32Type2D) {
  std::vector<int32_t> input = {0, 0, 0, 1, 2, 3, 0, 0, 0,
                                4, 5, 6, 7, 8, 9, 0, 0, 0};
  std::vector<int64_t> indices = {1, 3, 4};

  std::vector<int32_t> output = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int64_t> unflatten_dims = {2, 3};

  OpTester test("FlattenAndUnpad", 1, onnxruntime::kMSDomain);
  test.AddInput<int32_t>("input", {2, 3, 3}, input);
  test.AddInput<int64_t>("indices", {3}, indices);
  test.AddOutput<int32_t>("output", {3, 3}, output);
  test.AddOutput<int64_t>("unflatten_dims", {2}, unflatten_dims);
  test.Run();
}

TEST(FlattenAndUnpadTest, Int64Type1D) {
  std::vector<int64_t> input = {1, 1, 3, 2, 0, 3, 0, 4,
                                0, 5, 0, 6, 0, 0, 0};
  std::vector<int64_t> indices = {1, 3, 5, 7, 9, 11};

  std::vector<int64_t> output = {1, 2, 3, 4, 5, 6};
  std::vector<int64_t> unflatten_dims = {5, 3};

  OpTester test("FlattenAndUnpad", 1, onnxruntime::kMSDomain);
  test.AddInput<int64_t>("input", {5, 3}, input);
  test.AddInput<int64_t>("indices", {6}, indices);
  test.AddOutput<int64_t>("output", {6}, output);
  test.AddOutput<int64_t>("unflatten_dims", {2}, unflatten_dims);
  test.Run();
}

TEST(FlattenAndUnpadTest, Int64Type2D) {
  std::vector<int64_t> input = {0, 0, 0, 1, 2, 3, 0, 0, 0,
                                4, 5, 6, 7, 8, 9, 0, 0, 0};
  std::vector<int64_t> indices = {1, 3, 4};

  std::vector<int64_t> output = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int64_t> unflatten_dims = {2, 3};

  OpTester test("FlattenAndUnpad", 1, onnxruntime::kMSDomain);
  test.AddInput<int64_t>("input", {2, 3, 3}, input);
  test.AddInput<int64_t>("indices", {3}, indices);
  test.AddOutput<int64_t>("output", {3, 3}, output);
  test.AddOutput<int64_t>("unflatten_dims", {2}, unflatten_dims);
  test.Run();
}

TEST(FlattenAndUnpadTest, FloatType1D) {
  std::vector<float> input = {1.0f, 1.0f, 3.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f,
                              0.0f, 5.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f};
  std::vector<int64_t> indices = {1, 3, 5, 7, 9, 11};

  std::vector<float> output = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f};
  std::vector<int64_t> unflatten_dims = {5, 3};

  OpTester test("FlattenAndUnpad", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("input", {5, 3}, input);
  test.AddInput<int64_t>("indices", {6}, indices);
  test.AddOutput<float>("output", {6}, output);
  test.AddOutput<int64_t>("unflatten_dims", {2}, unflatten_dims);
  test.Run();
}

TEST(FlattenAndUnpadTest, FloatType2D) {
  std::vector<float> input = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f,
                              4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 0.0f, 0.0f, 0.0f};
  std::vector<int64_t> indices = {1, 3, 4};

  std::vector<float> output = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f, 7.f, 8.f, 9.f};
  std::vector<int64_t> unflatten_dims = {2, 3};

  OpTester test("FlattenAndUnpad", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("input", {2, 3, 3}, input);
  test.AddInput<int64_t>("indices", {3}, indices);
  test.AddOutput<float>("output", {3, 3}, output);
  test.AddOutput<int64_t>("unflatten_dims", {2}, unflatten_dims);
  test.Run();
}

TEST(FlattenAndUnpadTest, MLFloat16Type1D) {
  std::vector<float> input = {0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f,
                              0.0f, 5.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f};
  std::vector<int64_t> indices = {1, 3, 5, 7, 9, 11};

  std::vector<float> output = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f};
  std::vector<int64_t> unflatten_dims = {5, 3};

  std::vector<MLFloat16> input_half;
  input_half.resize(input.size());
  ConvertFloatToMLFloat16(input.data(), input_half.data(), static_cast<int>(input.size()));
  std::vector<MLFloat16> output_half;
  output_half.resize(output.size());
  ConvertFloatToMLFloat16(output.data(), output_half.data(), static_cast<int>(output.size()));

  OpTester test("FlattenAndUnpad", 1, onnxruntime::kMSDomain);
  test.AddInput<MLFloat16>("input", {5, 3}, input_half);
  test.AddInput<int64_t>("indices", {6}, indices);
  test.AddOutput<MLFloat16>("output", {6}, output_half);
  test.AddOutput<int64_t>("unflatten_dims", {2}, unflatten_dims);
  test.Run();
}

TEST(FlattenAndUnpadTest, MLFloat16Type2D) {
  std::vector<float> input = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f,
                              4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 0.0f, 0.0f, 0.0f};
  std::vector<int64_t> indices = {1, 3, 4};

  std::vector<float> output = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f, 7.f, 8.f, 9.f};
  std::vector<int64_t> unflatten_dims = {2, 3};

  std::vector<MLFloat16> input_half;
  input_half.resize(input.size());
  ConvertFloatToMLFloat16(input.data(), input_half.data(), static_cast<int>(input.size()));
  std::vector<MLFloat16> output_half;
  output_half.resize(output.size());
  ConvertFloatToMLFloat16(output.data(), output_half.data(), static_cast<int>(output.size()));

  OpTester test("FlattenAndUnpad", 1, onnxruntime::kMSDomain);
  test.AddInput<MLFloat16>("input", {2, 3, 3}, input_half);
  test.AddInput<int64_t>("indices", {3}, indices);
  test.AddOutput<MLFloat16>("output", {3, 3}, output_half);
  test.AddOutput<int64_t>("unflatten_dims", {2}, unflatten_dims);
  test.Run();
}

#endif

}  // namespace test
}  // namespace onnxruntime
