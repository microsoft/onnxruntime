// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void PrepareInputAndOutputData(const std::vector<float>& input,
                                      const float scale_0,
                                      const float scale_1,
                                      const float scale_2,
                                      std::vector<float>& output_0,
                                      std::vector<float>& output_1,
                                      std::vector<float>& output_2) {
  output_0.resize(input.size());
  output_1.resize(input.size());
  output_2.resize(input.size());
  std::transform(input.begin(), input.end(), output_0.begin(),
                 [&scale_0](float v) { return v / scale_0; });
  std::transform(input.begin(), input.end(), output_1.begin(),
                 [&scale_1](float v) { return v / scale_1; });
  std::transform(input.begin(), input.end(), output_2.begin(),
                 [&scale_2](float v) { return v / scale_2; });
}

TEST(BatchScaleTest, FloatType1D) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f};
  std::vector<float> scale_0 = {4.f};
  std::vector<float> scale_1 = {4.f};
  std::vector<float> scale_2 = {2.f};

  std::vector<float> output_0;
  std::vector<float> output_1;
  std::vector<float> output_2;
  PrepareInputAndOutputData(input, scale_0[0], scale_1[0], scale_2[0], output_0, output_1, output_2);

  int64_t N = static_cast<int64_t>(input.size());

  OpTester test("BatchScale", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("input", {N}, input);
  test.AddInput<float>("scale0", {}, scale_0);
  test.AddInput<float>("scale1", {}, scale_1);
  test.AddInput<float>("scale2", {}, scale_2);
  test.AddOutput<float>("output0", {N}, output_0);
  test.AddOutput<float>("output1", {N}, output_1);
  test.AddOutput<float>("output2", {N}, output_2);
  test.Run();
}

TEST(BatchScaleTest, FloatTypeVectorized1D) {
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f, 7.0f, 8.0f,
                              9.0f, 10.0f, 11.0f, 12.f, 13.0f, 14.0f, 15.0f, 16.0f,
                              17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.f, 23.0f, 24.0f,
                              25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.f, 31.0f, 32.0f};
  std::vector<float> scale_0 = {4.f};
  std::vector<float> scale_1 = {4.f};
  std::vector<float> scale_2 = {2.f};

  std::vector<float> output_0;
  std::vector<float> output_1;
  std::vector<float> output_2;
  PrepareInputAndOutputData(input, scale_0[0], scale_1[0], scale_2[0], output_0, output_1, output_2);

  int64_t N = static_cast<int64_t>(input.size());

  OpTester test("BatchScale", 1, onnxruntime::kMSDomain);
  test.AddInput<float>("input", {N}, input);
  test.AddInput<float>("scale0", {}, scale_0);
  test.AddInput<float>("scale1", {}, scale_1);
  test.AddInput<float>("scale2", {}, scale_2);
  test.AddOutput<float>("output0", {N}, output_0);
  test.AddOutput<float>("output1", {N}, output_1);
  test.AddOutput<float>("output2", {N}, output_2);
  test.Run();
}

// TEST(ScaledSumTest, FloatType2D) {
//   std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f, 7.f, 8.f, 9.f};
//   std::vector<int64_t> indices = {1, 3, 4};
//   std::vector<int64_t> unflatten_dims = {2, 3};

//   std::vector<float> output = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f,
//                                4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 0.0f, 0.0f, 0.0f};

//   std::vector<int64_t> full_flatten_dims = {6, 3};

//   OpTester test("ScaledSum", 1, onnxruntime::kMSDomain);
//   test.AddInput<float>("input", {3, 3}, input);
//   test.AddInput<int64_t>("indices", {3}, indices);
//   test.AddInput<int64_t>("unflatten_dims", {2}, unflatten_dims);
//   test.AddOutput<float>("output", {2, 3, 3}, output);
//   test.AddOutput<int64_t>("full_flatten_dims", {2}, full_flatten_dims);
//   test.Run();
// }

// TEST(ScaledSumTest, MLFloat16Type1D) {
//   std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f};
//   std::vector<int64_t> indices = {1, 3, 5, 7, 9, 11};
//   std::vector<int64_t> unflatten_dims = {5, 3};

//   std::vector<float> output = {0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f,
//                                0.0f, 5.0f, 0.0f, 6.0f, 0.0f, 0.0f, 0.0f};

//   std::vector<int64_t> full_flatten_dims = {15};

//   std::vector<MLFloat16> input_half;
//   input_half.resize(input.size());
//   ConvertFloatToMLFloat16(input.data(), input_half.data(), int(input.size()));
//   std::vector<MLFloat16> output_half;
//   output_half.resize(output.size());
//   ConvertFloatToMLFloat16(output.data(), output_half.data(), int(output.size()));

//   OpTester test("ScaledSum", 1, onnxruntime::kMSDomain);
//   test.AddInput<MLFloat16>("input", {6}, input_half);
//   test.AddInput<int64_t>("indices", {6}, indices);
//   test.AddInput<int64_t>("unflatten_dims", {2}, unflatten_dims);
//   test.AddOutput<MLFloat16>("output", {5, 3}, output_half);
//   test.AddOutput<int64_t>("full_flatten_dims", {1}, full_flatten_dims);
//   test.Run();
// }

// TEST(ScaledSumTest, MLFloat16Type2D) {
//   std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.f, 7.f, 8.f, 9.f};
//   std::vector<int64_t> indices = {1, 3, 4};
//   std::vector<int64_t> unflatten_dims = {2, 3};

//   std::vector<float> output = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f,
//                                4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 0.0f, 0.0f, 0.0f};

//   std::vector<int64_t> full_flatten_dims = {6, 3};

//   std::vector<MLFloat16> input_half;
//   input_half.resize(input.size());
//   ConvertFloatToMLFloat16(input.data(), input_half.data(), int(input.size()));
//   std::vector<MLFloat16> output_half;
//   output_half.resize(output.size());
//   ConvertFloatToMLFloat16(output.data(), output_half.data(), int(output.size()));

//   OpTester test("ScaledSum", 1, onnxruntime::kMSDomain);
//   test.AddInput<MLFloat16>("input", {3, 3}, input_half);
//   test.AddInput<int64_t>("indices", {3}, indices);
//   test.AddInput<int64_t>("unflatten_dims", {2}, unflatten_dims);
//   test.AddOutput<MLFloat16>("output", {2, 3, 3}, output_half);
//   test.AddOutput<int64_t>("full_flatten_dims", {2}, full_flatten_dims);
//   test.Run();
// }

}  // namespace test
}  // namespace onnxruntime
