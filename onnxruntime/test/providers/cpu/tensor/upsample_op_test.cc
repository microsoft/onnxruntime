// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/common/trt_op_test_utils.h"

namespace onnxruntime {
namespace test {

// Some of the tests can't run on TensorrtExecutionProvider because TensorRT only supports "nearest" mode Upsample
// and limited data types. Those tests will fallback to other EPs

TEST(UpsampleOpTest, UpsampleOpNearestTest) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 1.0f, 2.0f, 3.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 1, C = 2, H = 2, W = 2;
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

TEST(UpsampleOpTest, NhwcUpsampleOpNearestTest) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 2.0f, 3.0f, 1.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 1, H = 2, W = 2, C = 2;
  std::vector<float> X = {1.0f, 3.0f,
                          3.0f, 5.0f,

                          3.0f, 5.0f,
                          7.0f, 9.0f};

  test.AddInput<float>("X", {N, H, W, C}, X);

  std::vector<float> Y = {
      1.0f, 3.0f,
      1.0f, 3.0f,
      1.0f, 3.0f,
      3.0f, 5.0f,
      3.0f, 5.0f,
      3.0f, 5.0f,

      1.0f, 3.0f,
      1.0f, 3.0f,
      1.0f, 3.0f,
      3.0f, 5.0f,
      3.0f, 5.0f,
      3.0f, 5.0f,

      3.0f, 5.0f,
      3.0f, 5.0f,
      3.0f, 5.0f,
      7.0f, 9.0f,
      7.0f, 9.0f,
      7.0f, 9.0f,

      3.0f, 5.0f,
      3.0f, 5.0f,
      3.0f, 5.0f,
      7.0f, 9.0f,
      7.0f, 9.0f,
      7.0f, 9.0f};

  test.AddOutput<float>("Y", {N, (int64_t)(H * scales[1]), (int64_t)(W * scales[2]), C}, Y);
  // CUDA: result mismatch due to not implementing NHWC support
  // TensorRT: results mismatch
  // ROCm: results mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kTensorrtExecutionProvider, kRocmExecutionProvider});
}

TEST(UpsampleOpTest, UpsampleOpNearestTest_int32) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 1.0f, 2.0f, 3.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 1, C = 2, H = 2, W = 2;
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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: nvinfer1::query::Ports<nvinfer1::query::AbstractTensor>&): Assertion `!formats.empty()' failed
}

TEST(UpsampleOpTest, NhwcUpsampleOpNearestTest_int32) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 2.0f, 3.0f, 1.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 1, H = 2, W = 2, C = 2;
  std::vector<int32_t> X = {1, 3,
                            3, 5,

                            3, 5,
                            7, 9};

  test.AddInput<int32_t>("X", {N, H, W, C}, X);

  std::vector<int32_t> Y = {
      1, 3,
      1, 3,
      1, 3,
      3, 5,
      3, 5,
      3, 5,

      1, 3,
      1, 3,
      1, 3,
      3, 5,
      3, 5,
      3, 5,

      3, 5,
      3, 5,
      3, 5,
      7, 9,
      7, 9,
      7, 9,

      3, 5,
      3, 5,
      3, 5,
      7, 9,
      7, 9,
      7, 9};

  test.AddOutput<int32_t>("Y", {N, (int64_t)(H * scales[1]), (int64_t)(W * scales[2]), C}, Y);
  // CUDA: result mismatch due to not implementing NHWC support
  // TensorRT: results mismatch
  // ROCm: results mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kTensorrtExecutionProvider, kRocmExecutionProvider});
}

TEST(UpsampleOpTest, UpsampleOpNearestTest_uint8) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 1.0f, 2.0f, 3.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 1, C = 2, H = 2, W = 2;
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
  test.Run();
}

TEST(UpsampleOpTest, NhwcUpsampleOpNearestTest_uint8) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 2.0f, 3.0f, 1.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 1, H = 2, W = 2, C = 2;
  std::vector<uint8_t> X = {1, 3,
                            3, 5,

                            3, 5,
                            7, 9};

  test.AddInput<uint8_t>("X", {N, H, W, C}, X);

  std::vector<uint8_t> Y = {
      1, 3,
      1, 3,
      1, 3,
      3, 5,
      3, 5,
      3, 5,

      1, 3,
      1, 3,
      1, 3,
      3, 5,
      3, 5,
      3, 5,

      3, 5,
      3, 5,
      3, 5,
      7, 9,
      7, 9,
      7, 9,

      3, 5,
      3, 5,
      3, 5,
      7, 9,
      7, 9,
      7, 9};

  test.AddOutput<uint8_t>("Y", {N, (int64_t)(H * scales[1]), (int64_t)(W * scales[2]), C}, Y);
  // CUDA: result mismatch due to not implementing NHWC support
  // TensorRT: results mismatch
  // ROCm: results mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kTensorrtExecutionProvider, kRocmExecutionProvider});
}

TEST(UpsampleOpTest, UpsampleOpNearest2XTest) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 1.0f, 2.0f, 2.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 1, C = 2, H = 2, W = 2;
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

TEST(UpsampleOpTest, NhwcUpsampleOpNearest2XTest) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 2.0f, 2.0f, 1.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 1, H = 2, W = 2, C = 2;
  std::vector<float> X = {1.0f, 3.0f,
                          3.0f, 5.0f,

                          3.0f, 5.0f,
                          7.0f, 9.0f};

  test.AddInput<float>("X", {N, H, W, C}, X);

  std::vector<float> Y = {
      1.0f, 3.0f,
      1.0f, 3.0f,
      3.0f, 5.0f,
      3.0f, 5.0f,

      1.0f, 3.0f,
      1.0f, 3.0f,
      3.0f, 5.0f,
      3.0f, 5.0f,

      3.0f, 5.0f,
      3.0f, 5.0f,
      7.0f, 9.0f,
      7.0f, 9.0f,

      3.0f, 5.0f,
      3.0f, 5.0f,
      7.0f, 9.0f,
      7.0f, 9.0f};

  test.AddOutput<float>("Y", {N, (int64_t)(H * scales[1]), (int64_t)(W * scales[2]), C}, Y);
  // CUDA: result mismatch due to not implementing NHWC support
  // TensorRT: results mismatch
  // ROCm: results mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kTensorrtExecutionProvider, kRocmExecutionProvider});
}

TEST(UpsampleOpTest, UpsampleOpNearest222XTest) {
  OpTester test("Upsample");

  std::vector<float> scales{2.0f, 1.0f, 2.0f, 2.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 1, C = 2, H = 2, W = 2;
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
      7.0f, 7.0f, 9.0f, 9.0f};

  test.AddOutput<float>("Y", {N * 2, C, (int64_t)(H * scales[2]), (int64_t)(W * scales[3])}, Y);
  test.Run();
}

TEST(UpsampleOpTest, NhwcUpsampleOpNearest222XTest) {
  OpTester test("Upsample");

  std::vector<float> scales{2.0f, 2.0f, 2.0f, 1.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 1, H = 2, W = 2, C = 2;
  std::vector<float> X = {1.0f, 3.0f,
                          3.0f, 5.0f,

                          3.0f, 5.0f,
                          7.0f, 9.0f};

  test.AddInput<float>("X", {N, H, W, C}, X);

  std::vector<float> Y = {
      1.0f, 3.0f,
      1.0f, 3.0f,
      3.0f, 5.0f,
      3.0f, 5.0f,

      1.0f, 3.0f,
      1.0f, 3.0f,
      3.0f, 5.0f,
      3.0f, 5.0f,

      3.0f, 5.0f,
      3.0f, 5.0f,
      7.0f, 9.0f,
      7.0f, 9.0f,

      3.0f, 5.0f,
      3.0f, 5.0f,
      7.0f, 9.0f,
      7.0f, 9.0f,

      1.0f, 3.0f,
      1.0f, 3.0f,
      3.0f, 5.0f,
      3.0f, 5.0f,

      1.0f, 3.0f,
      1.0f, 3.0f,
      3.0f, 5.0f,
      3.0f, 5.0f,

      3.0f, 5.0f,
      3.0f, 5.0f,
      7.0f, 9.0f,
      7.0f, 9.0f,

      3.0f, 5.0f,
      3.0f, 5.0f,
      7.0f, 9.0f,
      7.0f, 9.0f};

  test.AddOutput<float>("Y", {(int64_t)(N * scales[0]), (int64_t)(H * scales[1]), (int64_t)(W * scales[2]), C}, Y);
  // CUDA: result mismatch due to not implementing NHWC support
  // TensorRT: results mismatch
  // ROCm: results mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kTensorrtExecutionProvider, kRocmExecutionProvider});
}

TEST(UpsampleOpTest, UpsampleOpNearest15XTest) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 1.0f, 2.0f, 1.5f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 1, C = 2, H = 2, W = 2;
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

TEST(UpsampleOpTest, NhwcUpsampleOpNearest15XTest) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 2.0f, 1.5f, 1.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 1, H = 2, W = 2, C = 2;
  std::vector<float> X = {1.0f, 3.0f,
                          3.0f, 5.0f,

                          3.0f, 5.0f,
                          7.0f, 9.0f};

  test.AddInput<float>("X", {N, H, W, C}, X);

  std::vector<float> Y = {
      1.0f, 3.0f,
      1.0f, 3.0f,
      3.0f, 5.0f,

      1.0f, 3.0f,
      1.0f, 3.0f,
      3.0f, 5.0f,

      3.0f, 5.0f,
      3.0f, 5.0f,
      7.0f, 9.0f,

      3.0f, 5.0f,
      3.0f, 5.0f,
      7.0f, 9.0f};

  test.AddOutput<float>("Y", {N, (int64_t)(H * scales[1]), (int64_t)(W * scales[2]), C}, Y);
  // CUDA: result mismatch due to not implementing NHWC support
  // TensorRT: results mismatch
  // ROCm: results mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kTensorrtExecutionProvider, kRocmExecutionProvider});
}

TEST(UpsampleOpTest, UpsampleOpNearestTest_NoScale) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 1.0f, 1.0f, 1.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 1, C = 2, H = 2, W = 2;
  std::vector<float> X = {1.0f, 3.0f,
                          3.0f, 5.0f,

                          3.0f, 5.0f,
                          7.0f, 9.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);

  std::vector<float> Y = {1.0f, 3.0f,
                          3.0f, 5.0f,

                          3.0f, 5.0f,
                          7.0f, 9.0f};

  test.AddOutput<float>("Y", {N, C, H, W}, Y);
  test.Run();
}

TEST(UpsampleOpTest, UpsampleOpNearest2XTest_int32) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 1.0f, 2.0f, 2.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 1, C = 2, H = 2, W = 2;
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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: nvinfer1::query::Ports<nvinfer1::query::AbstractTensor>&): Assertion `!formats.empty()' failed
}

TEST(UpsampleOpTest, NhwcUpsampleOpNearest2XTest_int32) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 2.0f, 2.0f, 1.0f};
  test.AddAttribute("mode", "nearest");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 1, H = 2, W = 2, C = 2;
  std::vector<int32_t> X = {1, 3,
                            3, 5,

                            3, 5,
                            7, 9};

  test.AddInput<int32_t>("X", {N, H, W, C}, X);

  std::vector<int32_t> Y = {
      1, 3,
      1, 3,
      3, 5,
      3, 5,

      1, 3,
      1, 3,
      3, 5,
      3, 5,

      3, 5,
      3, 5,
      7, 9,
      7, 9,

      3, 5,
      3, 5,
      7, 9,
      7, 9};

  test.AddOutput<int32_t>("Y", {N, (int64_t)(H * scales[1]), (int64_t)(W * scales[2]), C}, Y);
  // CUDA: result mismatch due to not implementing NHWC support
  // TensorRT: results mismatch
  // ROCm: results mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kTensorrtExecutionProvider, kRocmExecutionProvider});
}

TEST(UpsampleOpTest, UpsampleOp4DBilinearTest) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: The difference between expected[i] and output[i] is 0.5, which exceeds threshold";
  }

  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 1.0f, 2.0f, 4.0f};
  test.AddAttribute("mode", "linear");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 2, C = 1, H = 2, W = 2;
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
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  // TensorRT: results mismatch
}

TEST(UpsampleOpTest, NhwcUpsampleOp4D1CBilinearTest) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: The difference between expected[i] and output[i] is 0.25, which exceeds threshold";
  }

  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 2.0f, 4.0f, 1.0f};
  test.AddAttribute("mode", "linear");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 2, H = 2, W = 3, C = 1;
  std::vector<float> X = {1.0f, 2.0f, 3.0f,
                          4.0f, 5.0f, 6.0f,

                          7.0f, 8.0f, 9.0f,
                          10.0f, 11.0f, 12.0f};

  test.AddInput<float>("X", {N, H, W, C}, X);

  std::vector<float> Y = {
      1.0f, 1.25f, 1.5f, 1.75f, 2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.0f, 3.0f, 3.0f,
      2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f, 4.0f, 4.25f, 4.5f, 4.5f, 4.5f, 4.5f,
      4.0f, 4.25f, 4.5f, 4.75f, 5.0f, 5.25f, 5.5f, 5.75f, 6.0f, 6.0f, 6.0f, 6.0f,
      4.0f, 4.25f, 4.5f, 4.75f, 5.0f, 5.25f, 5.5f, 5.75f, 6.0f, 6.0f, 6.0f, 6.0f,

      7.0f, 7.25f, 7.5f, 7.75f, 8.0f, 8.25f, 8.5f, 8.75f, 9.0f, 9.0f, 9.0f, 9.0f,
      8.5f, 8.75f, 9.0f, 9.25f, 9.5f, 9.75f, 10.0f, 10.25f, 10.5f, 10.5f, 10.5f, 10.5f,
      10.0f, 10.25f, 10.5f, 10.75f, 11.0f, 11.25f, 11.5f, 11.75f, 12.0f, 12.0f, 12.0f, 12.0f,
      10.0f, 10.25f, 10.5f, 10.75f, 11.0f, 11.25f, 11.5f, 11.75f, 12.0f, 12.0f, 12.0f, 12.0f};

  test.AddOutput<float>("Y", {N, (int64_t)(H * scales[1]), (int64_t)(W * scales[2]), C}, Y);
  // CUDA: result mismatch due to not implementing NHWC support
  // TensorRT: results mismatch
  // ROCm: results mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kCudaNHWCExecutionProvider, kTensorrtExecutionProvider, kRocmExecutionProvider});
}

TEST(UpsampleOpTest, NhwcUpsampleOp4DBilinearTest) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: The difference between expected[i] and output[i] is 0.75, which exceeds threshold";
  }

  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 2.0f, 2.0f, 1.0f};
  test.AddAttribute("mode", "linear");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 2, H = 2, W = 2, C = 3;
  std::vector<float> X = {1.0f, 2.0f, 3.0f,
                          4.0f, 5.0f, 6.0f,
                          7.0f, 8.0f, 9.0f,
                          10.0f, 11.0f, 12.0f,

                          13.0f, 14.0f, 15.0f,
                          16.0f, 17.0f, 18.0f,
                          19.0f, 20.0f, 21.0f,
                          22.0f, 23.0f, 24.0f};

  test.AddInput<float>("X", {N, H, W, C}, X);

  std::vector<float> Y = {
      1.0f, 2.0f, 3.0f,
      2.5f, 3.5f, 4.5f,
      4.0f, 5.0f, 6.0f,
      4.0f, 5.0f, 6.0f,

      4.0f, 5.0f, 6.0f,
      5.5f, 6.5f, 7.5f,
      7.0f, 8.0f, 9.0f,
      7.0f, 8.0f, 9.0f,

      7.0f, 8.0f, 9.0f,
      8.5f, 9.5f, 10.5f,
      10.0f, 11.0f, 12.0f,
      10.0f, 11.0f, 12.0f,

      7.0f, 8.0f, 9.0f,
      8.5f, 9.5f, 10.5f,
      10.0f, 11.0f, 12.0f,
      10.0f, 11.0f, 12.0f,

      13.0f, 14.0f, 15.0f,
      14.5f, 15.5f, 16.5f,
      16.0f, 17.0f, 18.0f,
      16.0f, 17.0f, 18.0f,

      16.0f, 17.0f, 18.0f,
      17.5f, 18.5f, 19.5f,
      19.0f, 20.0f, 21.0f,
      19.0f, 20.0f, 21.0f,

      19.0f, 20.0f, 21.0f,
      20.5f, 21.5f, 22.5f,
      22.0f, 23.0f, 24.0f,
      22.0f, 23.0f, 24.0f,

      19.0f, 20.0f, 21.0f,
      20.5f, 21.5f, 22.5f,
      22.0f, 23.0f, 24.0f,
      22.0f, 23.0f, 24.0f};

  test.AddOutput<float>("Y", {N, (int64_t)(H * scales[1]), (int64_t)(W * scales[2]), C}, Y);
  // CUDA: result mismatch due to not implementing NHWC support
  // TensorRT: results mismatch
  // ROCm: results mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kCudaNHWCExecutionProvider, kTensorrtExecutionProvider, kRocmExecutionProvider});
}

TEST(UpsampleOpTest, UpsampleOp2DBilinearTest) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: The difference between expected[i] and output[i] is 0.5, which exceeds threshold";
  }

  OpTester test("Upsample");

  std::vector<float> scales{2.0f, 4.0f};
  test.AddAttribute("mode", "linear");
  test.AddAttribute("scales", scales);

  constexpr int64_t H = 2, W = 2;
  std::vector<float> X = {1.0f, 3.0f,
                          3.0f, 5.0f};

  test.AddInput<float>("X", {H, W}, X);

  std::vector<float> Y = {
      1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.0f, 3.0f, 3.0f,
      2.0f, 2.5f, 3.0f, 3.5f, 4.0f, 4.0f, 4.0f, 4.0f,
      3.0f, 3.5f, 4.0f, 4.5f, 5.0f, 5.0f, 5.0f, 5.0f,
      3.0f, 3.5f, 4.0f, 4.5f, 5.0f, 5.0f, 5.0f, 5.0f};

  test.AddOutput<float>("Y", {(int64_t)(H * scales[0]), (int64_t)(W * scales[1])}, Y);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});  // TensorRT/OpenVINO-EP: results mismatch
}

TEST(UpsampleOpTest, UpsampleOp4DBilinearTest_ScalesNoOp) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 1.0f, 1.0f, 1.0f};
  test.AddAttribute("mode", "linear");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 2, C = 1, H = 2, W = 2;
  std::vector<float> X = {1.0f, 3.0f,
                          3.0f, 5.0f,

                          3.0f, 5.0f,
                          7.0f, 9.0f};

  test.AddInput<float>("X", {N, C, H, W}, X);

  std::vector<float> Y = {1.0f, 3.0f,
                          3.0f, 5.0f,

                          3.0f, 5.0f,
                          7.0f, 9.0f};

  test.AddOutput<float>("Y", {N, C, H, W}, Y);
  test.Run();
}

TEST(UpsampleOpTest, UpsampleOp4DBilinearTest_int32) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 1.0f, 2.0f, 4.0f};
  test.AddAttribute("mode", "linear");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 2, C = 1, H = 2, W = 2;
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
  test.Run();
}

TEST(UpsampleOpTest, NhwcUpsampleOp4DBilinearTest_int32) {
  OpTester test("Upsample");

  std::vector<float> scales{1.0f, 2.0f, 4.0f, 1.0f};
  test.AddAttribute("mode", "linear");
  test.AddAttribute("scales", scales);

  constexpr int64_t N = 2, H = 2, W = 2, C = 1;
  std::vector<int32_t> X = {1, 3,
                            3, 5,

                            3, 5,
                            7, 9};

  test.AddInput<int32_t>("X", {N, H, W, C}, X);

  std::vector<int32_t> Y = {
      1, 1, 2, 2, 3, 3, 3, 3,
      2, 2, 3, 3, 4, 4, 4, 4,
      3, 3, 4, 4, 5, 5, 5, 5,
      3, 3, 4, 4, 5, 5, 5, 5,

      3, 3, 4, 4, 5, 5, 5, 5,
      5, 5, 6, 6, 7, 7, 7, 7,
      7, 7, 8, 8, 9, 9, 9, 9,
      7, 7, 8, 8, 9, 9, 9, 9};

  test.AddOutput<int32_t>("Y", {N, (int64_t)(H * scales[1]), (int64_t)(W * scales[2]), C}, Y);
  // CUDA: result mismatch due to not implementing NHWC support
  // TensorRT: results mismatch
  // ROCm: results mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kCudaNHWCExecutionProvider, kTensorrtExecutionProvider, kRocmExecutionProvider});
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
  test.Run();
}

TEST(UpsampleOpTest, UpsampleOpNearest2XTest_opset9) {
  OpTester test("Upsample", 9);

  std::vector<float> scales{1.0f, 1.0f, 2.0f, 2.0f};
  test.AddAttribute("mode", "nearest");

  constexpr int64_t N = 1, C = 2, H = 2, W = 2;
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

  // TRT: segmentation fault in A100
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", ExcludeTrtOnA100());
}

TEST(UpsampleOpTest, NhwcUpsampleOpNearest2XTest_opset9) {
  OpTester test("Upsample", 9);

  std::vector<float> scales{1.0f, 2.0f, 2.0f, 1.0};
  test.AddAttribute("mode", "nearest");

  constexpr int64_t N = 1, H = 2, W = 2, C = 2;
  std::vector<int32_t> X = {1, 3,
                            3, 5,

                            3, 5,
                            7, 9};

  test.AddInput<int32_t>("X", {N, H, W, C}, X);
  test.AddInput<float>("scales", {4}, scales);

  std::vector<int32_t> Y = {
      1, 3,
      1, 3,
      3, 5,
      3, 5,

      1, 3,
      1, 3,
      3, 5,
      3, 5,

      3, 5,
      3, 5,
      7, 9,
      7, 9,

      3, 5,
      3, 5,
      7, 9,
      7, 9};

  test.AddOutput<int32_t>("Y", {N, (int64_t)(H * scales[1]), (int64_t)(W * scales[2]), C}, Y);
  // CUDA: result mismatch due to not implementing NHWC support
  // TensorRT: results mismatch
  // ROCm: results mismatch
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaExecutionProvider, kTensorrtExecutionProvider, kRocmExecutionProvider});
}
}  // namespace test
}  // namespace onnxruntime
