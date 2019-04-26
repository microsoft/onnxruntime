// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/upsample.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// Disable TensorRT on some of the tests because TensorRT only supports "nearest" mode Upsample and limited data types

TEST(UpsampleOpTest, UpsampleOpNearestTest) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 1.0f, 2.0f, 3.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  const int64_t N = 1, C = 2, H = 2, W = 2;
  std::vector<float> X = {1.0f, 3.0f,
                          3.0f, 5.0f,

                          3.0f, 5.0f,
                          7.0f, 9.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);

  std::vector<float> Y = {
      1.0f, 1.0f, 1.0f, 3.0f, 3.0f, 3.0f,
      1.0f, 1.0f, 1.0f, 3.0f, 3.0f, 3.0f,
      3.0f, 3.0f, 3.0f, 5.0f, 5.0f, 5.0f,
      3.0f, 3.0f, 3.0f, 5.0f, 5.0f, 5.0f,

      3.0f, 3.0f, 3.0f, 5.0f, 5.0f, 5.0f,
      3.0f, 3.0f, 3.0f, 5.0f, 5.0f, 5.0f,
      7.0f, 7.0f, 7.0f, 9.0f, 9.0f, 9.0f,
      7.0f, 7.0f, 7.0f, 9.0f, 9.0f, 9.0f};

  test.AddOutput<float>("Y", {N, C, (int64_t)(H * scales[2]), (int64_t)(W * scales[3])}, Y);
  test.Run();
}

TEST(UpsampleOpTest, UpsampleOpNearestTest_int32) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 1.0f, 2.0f, 3.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  const int64_t N = 1, C = 2, H = 2, W = 2;
  std::vector<int32_t> X = {1, 3,
                            3, 5,

                            3, 5,
                            7, 9};

  test.AddInput<int32_t>("X", {N, C, H, W}, X);

  std::vector<int32_t> Y = {
      1, 1, 1, 3, 3, 3,
      1, 1, 1, 3, 3, 3,
      3, 3, 3, 5, 5, 5,
      3, 3, 3, 5, 5, 5,

      3, 3, 3, 5, 5, 5,
      3, 3, 3, 5, 5, 5,
      7, 7, 7, 9, 9, 9,
      7, 7, 7, 9, 9, 9};

  test.AddOutput<int32_t>("Y", {N, C, (int64_t)(H * scales[2]), (int64_t)(W * scales[3])}, Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(UpsampleOpTest, UpsampleOpNearestTest_uint8) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 1.0f, 2.0f, 3.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  const int64_t N = 1, C = 2, H = 2, W = 2;
  std::vector<uint8_t> X = {1, 3,
                            3, 5,

                            3, 5,
                            7, 9};

  test.AddInput<uint8_t>("X", {N, C, H, W}, X);

  std::vector<uint8_t> Y = {
      1, 1, 1, 3, 3, 3,
      1, 1, 1, 3, 3, 3,
      3, 3, 3, 5, 5, 5,
      3, 3, 3, 5, 5, 5,

      3, 3, 3, 5, 5, 5,
      3, 3, 3, 5, 5, 5,
      7, 7, 7, 9, 9, 9,
      7, 7, 7, 9, 9, 9};

  test.AddOutput<uint8_t>("Y", {N, C, (int64_t)(H * scales[2]), (int64_t)(W * scales[3])}, Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});//TensorRT: unsupported data type
}

TEST(UpsampleOpTest, UpsampleOpNearest2XTest) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 1.0f, 2.0f, 2.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  const int64_t N = 1, C = 2, H = 2, W = 2;
  std::vector<float> X = {1.0f, 3.0f,
                          3.0f, 5.0f,

                          3.0f, 5.0f,
                          7.0f, 9.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);

  std::vector<float> Y = {
      1.0f, 1.0f, 3.0f, 3.0f,
      1.0f, 1.0f, 3.0f, 3.0f,
      3.0f, 3.0f, 5.0f, 5.0f,
      3.0f, 3.0f, 5.0f, 5.0f,

      3.0f, 3.0f, 5.0f, 5.0f,
      3.0f, 3.0f, 5.0f, 5.0f,
      7.0f, 7.0f, 9.0f, 9.0f,
      7.0f, 7.0f, 9.0f, 9.0f};

  test.AddOutput<float>("Y", {N, C, (int64_t)(H * scales[2]), (int64_t)(W * scales[3])}, Y);
  test.Run();
}

TEST(UpsampleOpTest, UpsampleOpNearest222XTest) {
  OpTester test("Upsample");

  std::vector<float> scales{2.0f, 1.0f, 2.0f, 2.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  const int64_t N = 1, C = 2, H = 2, W = 2;
  std::vector<float> X = {1.0f, 3.0f,
                          3.0f, 5.0f,

                          3.0f, 5.0f,
                          7.0f, 9.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);

  std::vector<float> Y = {
      1.0f, 1.0f, 3.0f, 3.0f,
      1.0f, 1.0f, 3.0f, 3.0f,
      3.0f, 3.0f, 5.0f, 5.0f,
      3.0f, 3.0f, 5.0f, 5.0f,

      3.0f, 3.0f, 5.0f, 5.0f,
      3.0f, 3.0f, 5.0f, 5.0f,
      7.0f, 7.0f, 9.0f, 9.0f,
      7.0f, 7.0f, 9.0f, 9.0f,

      1.0f, 1.0f, 3.0f, 3.0f,
      1.0f, 1.0f, 3.0f, 3.0f,
      3.0f, 3.0f, 5.0f, 5.0f,
      3.0f, 3.0f, 5.0f, 5.0f,

      3.0f, 3.0f, 5.0f, 5.0f,
      3.0f, 3.0f, 5.0f, 5.0f,
      7.0f, 7.0f, 9.0f, 9.0f,
      7.0f, 7.0f, 9.0f, 9.0f
  };

  test.AddOutput<float>("Y", {N*2, C, (int64_t)(H * scales[2]), (int64_t)(W * scales[3])}, Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});//TensorRT parser: Assertion failed: scales[0] == 1 && scales[1] == 1
}

TEST(UpsampleOpTest, UpsampleOpNearest15XTest) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 1.0f, 2.0f, 1.5f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  const int64_t N = 1, C = 2, H = 2, W = 2;
  std::vector<float> X = {1.0f, 3.0f,
                          3.0f, 5.0f,

                          3.0f, 5.0f,
                          7.0f, 9.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);

  std::vector<float> Y = {
      1.0f, 1.0f, 3.0f,
      1.0f, 1.0f, 3.0f,
      3.0f, 3.0f, 5.0f,
      3.0f, 3.0f, 5.0f,

      3.0f, 3.0f, 5.0f,
      3.0f, 3.0f, 5.0f,
      7.0f, 7.0f, 9.0f,
      7.0f, 7.0f, 9.0f};

  test.AddOutput<float>("Y", {N, C, (int64_t)(H * scales[2]), (int64_t)(W * scales[3])}, Y);
  test.Run();
}

TEST(UpsampleOpTest, UpsampleOpNearest2XTest_int32) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 1.0f, 2.0f, 2.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  const int64_t N = 1, C = 2, H = 2, W = 2;
  std::vector<int32_t> X = {1, 3,
                            3, 5,

                            3, 5,
                            7, 9};

  test.AddInput<int32_t>("X", {N, C, H, W}, X);

  std::vector<int32_t> Y = {
      1, 1, 3, 3,
      1, 1, 3, 3,
      3, 3, 5, 5,
      3, 3, 5, 5,

      3, 3, 5, 5,
      3, 3, 5, 5,
      7, 7, 9, 9,
      7, 7, 9, 9};

  test.AddOutput<int32_t>("Y", {N, C, (int64_t)(H * scales[2]), (int64_t)(W * scales[3])}, Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(UpsampleOpTest, UpsampleOpBilinearTest) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 1.0f, 2.0f, 4.0f};
  test.AddAttribute("mode", "linear");
  test.AddAttribute("scales", scales);

  const int64_t N = 2, C = 1, H = 2, W = 2;
  std::vector<float> X = {1.0f, 3.0f,
                          3.0f, 5.0f,

                          3.0f, 5.0f,
                          7.0f, 9.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);

  std::vector<float> Y = {
      1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.0f, 3.0f, 3.0f,
      2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.0f, 4.0f, 4.0f,
      3.0f, 3.5f, 4.0f, 4.5f, 5.0f, 5.0f, 5.0f, 5.0f,
      3.0f, 3.5f, 4.0f, 4.5f, 5.0f, 5.0f, 5.0f, 5.0f,

      3.0f, 3.5f, 4.0f, 4.5f, 5.0f, 5.0f, 5.0f, 5.0f,
      5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.0f, 7.0f, 7.0f,
      7.0f, 7.5f, 8.0f, 8.5f, 9.0f, 9.0f, 9.0f, 9.0f,
      7.0f, 7.5f, 8.0f, 8.5f, 9.0f, 9.0f, 9.0f, 9.0f};

  test.AddOutput<float>("Y", {N, C, (int64_t)(H * scales[2]), (int64_t)(W * scales[3])}, Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});//TensorRT parser:Assertion failed:  mode == "nearest"
}

TEST(UpsampleOpTest, UpsampleOpBilinearTest2) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 1.0f, 2.0f, 4.0f};
  test.AddAttribute("mode", "linear");
  test.AddAttribute("scales", scales);

  const int64_t N = 2, C = 1, H = 2, W = 2;
  std::vector<float> X = {1.0f, 3.0f,
                          4.0f, 8.0f,

                          6.0f, 2.0f,
                          7.0f, 11.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);

  std::vector<float> Y = {
      1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.0f, 3.0f, 3.0f,
      2.5f, 3.25f, 4.0f, 4.75f, 5.5f, 5.5f, 5.5f, 5.5f,
      4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 8.0f, 8.0f, 8.0f,
      4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 8.0f, 8.0f, 8.0f,

      6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f,
      6.5f, 6.5f, 6.5f, 6.5f, 6.5f, 6.5f, 6.5f, 6.5f,
      7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 11.0f, 11.0f, 11.0f,
      7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 11.0f, 11.0f, 11.0f};

  test.AddOutput<float>("Y", {N, C, (int64_t)(H * scales[2]), (int64_t)(W * scales[3])}, Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});//TensorRT parser:Assertion failed:  mode == "nearest"
}

TEST(UpsampleOpTest, UpsampleOpBilinearTest_int32) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 1.0f, 2.0f, 4.0f};
  test.AddAttribute("mode", "linear");
  test.AddAttribute("scales", scales);

  const int64_t N = 2, C = 1, H = 2, W = 2;
  std::vector<int32_t> X = {1, 3,
                            3, 5,

                            3, 5,
                            7, 9};

  test.AddInput<int32_t>("X", {N, C, H, W}, X);

  std::vector<int32_t> Y = {
      1, 1, 2, 2, 3, 3, 3, 3,
      2, 2, 3, 3, 4, 4, 4, 4,
      3, 3, 4, 4, 5, 5, 5, 5,
      3, 3, 4, 4, 5, 5, 5, 5,

      3, 3, 4, 4, 5, 5, 5, 5,
      5, 5, 6, 6, 7, 7, 7, 7,
      7, 7, 8, 8, 9, 9, 9, 9,
      7, 7, 8, 8, 9, 9, 9, 9};

  test.AddOutput<int32_t>("Y", {N, C, (int64_t)(H * scales[2]), (int64_t)(W * scales[3])}, Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});//TensorRT parser:Assertion failed:  mode == "nearest"
}

TEST(UpsampleOpTest, UpsampleOpNearestTest_1D) {
  OpTester test("Upsample");

  std::vector<float> scales{2.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  std::vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  test.AddInput<float>("X", {5}, X);

  std::vector<float> Y = {
      1.0f, 1.0f,
      2.0f, 2.0f,
      3.0f, 3.0f,
      4.0f, 4.0f,
      5.0f, 5.0f};

  test.AddOutput<float>("Y", {10}, Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});//TensorRT parser:Assertion failed: tensor.getDimensions().nbDims == 3
}

TEST(UpsampleOpTest, UpsampleOpNearest2XTest_opset9) {
  OpTester test("Upsample", 9);

  std::vector<float> scales{1.0f, 1.0f, 2.0f, 2.0f};
  test.AddAttribute("mode", "nearest");

  const int64_t N = 1, C = 2, H = 2, W = 2;
  std::vector<int32_t> X = {1, 3,
                            3, 5,

                            3, 5,
                            7, 9};

  test.AddInput<int32_t>("X", {N, C, H, W}, X);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<int32_t> Y = {
      1, 1, 3, 3,
      1, 1, 3, 3,
      3, 3, 5, 5,
      3, 3, 5, 5,

      3, 3, 5, 5,
      3, 3, 5, 5,
      7, 7, 9, 9,
      7, 7, 9, 9};

  test.AddOutput<int32_t>("Y", {N, C, (int64_t)(H * scales[2]), (int64_t)(W * scales[3])}, Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});//TensorRT parser:Assertion failed: scales_input.is_weights()
}
}  // namespace test
}  // namespace onnxruntime
