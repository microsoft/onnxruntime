// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

TEST(MathOpTest, Clip_6) {
  OpTester test("Clip", 6);

  test.AddAttribute("min", -10.0f);
  test.AddAttribute("max", 10.0f);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("X", dims,
                       {11.0f, 4.4f, 432.3f,
                        -1.3f, 3.5f, 64.0f,
                        -5.4f, 9.3f, 82.4f});
  test.AddOutput<float>("Y", dims,
                        {10.0f, 4.4f, 10.0f,
                         -1.3f, 3.5f, 10.0f,
                         -5.4f, 9.3f, 10.0f});
#if defined(OPENVINO_CONFIG_CPU)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
#else
  test.Run();
#endif
}

TEST(MathOpTest, Clip_Default) {
  OpTester test("Clip", 12);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<float>("X", dims,
                       {11.0f, 4.4f, 432.3f,
                        -1.3f, 3.5f, 64.0f,
                        -5.4f, 9.3f, 82.4f});
  test.AddOutput<float>("Y", dims,
                        {11.0f, 4.4f, 432.3f,
                         -1.3f, 3.5f, 64.0f,
                         -5.4f, 9.3f, 82.4f});

  test.Run();
}

TEST(MathOpTest, Clip_Default_int8) {
  OpTester test("Clip", 12);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<int8_t>("X", dims,
                        {11, 4, 127,
                         -1, 3, 64,
                         -5, 9, 82});
  test.AddOutput<int8_t>("Y", dims,
                         {11, 4, 127,
                          -1, 3, 64,
                          -5, 9, 82});

  // TensorRT does not support Clip opset 12 yet.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(MathOpTest, Clip_Default_uint8) {
  OpTester test("Clip", 12);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<uint8_t>("X", dims,
                         {11, 4, 255,
                          1, 3, 64,
                          5, 9, 82});
  test.AddOutput<uint8_t>("Y", dims,
                          {11, 4, 255,
                           1, 3, 64,
                           5, 9, 82});

  // TensorRT does not support Clip opset 12 yet.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(MathOpTest, Clip_Default_int64) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: Expected equality of these values: 11 and -9223372036854775808";
  }

  OpTester test("Clip", 12);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<int64_t>("X", dims,
                         {11, 4, 432,
                          -1, 3, 64,
                          -5, 9, 82});
  test.AddOutput<int64_t>("Y", dims,
                          {11, 4, 432,
                           -1, 3, 64,
                           -5, 9, 82});

  // TensorRT does not support Clip opset 12 yet.
  // Skipping for OpenVINO because of the following error: Expected equality of these values: cur_expected[i] Which is: 11 cur_actual[i] Which is: 0
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(MathOpTest, Clip_Default_uint64) {
  OpTester test("Clip", 12);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<uint64_t>("X", dims,
                          {11, 4, 432,
                           1, 3, 64,
                           5, 9, 82});
  test.AddOutput<uint64_t>("Y", dims,
                           {11, 4, 432,
                            1, 3, 64,
                            5, 9, 82});

  // TensorRT does not support Clip opset 12 yet.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(MathOpTest, Clip_MLFloat16) {
  auto run_test = [](bool min_max_are_initializer) {
    OpTester test("Clip", 12);

    std::vector<int64_t> dims{3, 3};
    test.AddInput<MLFloat16>("X", dims,
                             {MLFloat16(-1.0f), MLFloat16(-2.0f), MLFloat16(-3.0f),
                              MLFloat16(-4.0f), MLFloat16(0.0f), MLFloat16(2.0f),
                              MLFloat16(4.0f), MLFloat16(6.0f), MLFloat16(8.0f)});
    test.AddInput<MLFloat16>("min", {}, {MLFloat16(0.0f)}, min_max_are_initializer);
    test.AddInput<MLFloat16>("max", {}, {MLFloat16(6.0f)}, min_max_are_initializer);
    test.AddOutput<MLFloat16>("Y", dims,
                              {MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(0.0f),
                               MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(2.0f),
                               MLFloat16(4.0f), MLFloat16(6.0f), MLFloat16(6.0f)});

    test.Run();
  };
  run_test(true);  // coreml requires constant max/min
  run_test(false);
}

TEST(MathOpTest, Clip_MLFloat16_NoMin_NoMax) {
  OpTester test("Clip", 12);

  std::vector<int64_t> dims{3};
  test.AddInput<MLFloat16>("X", dims,
                           {MLFloat16(-1.0f), MLFloat16(-2.0f), MLFloat16(3.0f)});
  test.AddOutput<MLFloat16>("Y", dims,
                            {MLFloat16(-1.0f), MLFloat16(-2.0f), MLFloat16(3.0f)});

  test.Run();
}

TEST(MathOpTest, Clip_MLFloat16_NoMax) {
  OpTester test("Clip", 12);

  std::vector<int64_t> dims{3};
  test.AddInput<MLFloat16>("X", dims,
                           {MLFloat16(-1.0f), MLFloat16(-2.0f), MLFloat16(3.0f)});
  test.AddInput<MLFloat16>("min", {}, {MLFloat16(0.0f)});
  test.AddOutput<MLFloat16>("Y", dims,
                            {MLFloat16(0.0f), MLFloat16(0.0f), MLFloat16(3.0f)});

  test.Run();
}

TEST(MathOpTest, Clip_MLFloat16_NoMin) {
  OpTester test("Clip", 12);

  std::vector<int64_t> dims{3};
  test.AddInput<MLFloat16>("X", dims,
                           {MLFloat16(-1.0f), MLFloat16(-2.0f), MLFloat16(3.0f)});
  test.AddOptionalInputEdge<MLFloat16>();  // no min
  test.AddInput<MLFloat16>("max", {}, {MLFloat16(0.0f)});
  test.AddOutput<MLFloat16>("Y", dims,
                            {MLFloat16(-1.0f), MLFloat16(-2.0f), MLFloat16(0.0f)});

  test.Run();
}

TEST(MathOpTest, Clip_int32) {
  OpTester test("Clip", 12);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<int32_t>("X", dims,
                         {-1, 0, 1,
                          -16, 12, -6,
                          -5, 2, 16});
  test.AddInput<int32_t>("min", {}, {-10});
  test.AddInput<int32_t>("max", {}, {10});
  test.AddOutput<int32_t>("Y", dims,
                          {-1, 0, 1,
                           -10, 10, -6,
                           -5, 2, 10});

  // TensorRT does not support Clip opset 12 yet.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(MathOpTest, Clip_uint32) {
  OpTester test("Clip", 12);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<uint32_t>("X", dims,
                          {0, 0, 1,
                           5, 12, 3,
                           2, 7, 16});
  test.AddInput<uint32_t>("min", {}, {3});
  test.AddInput<uint32_t>("max", {}, {10});
  test.AddOutput<uint32_t>("Y", dims,
                           {3, 3, 3,
                            5, 10, 3,
                            3, 7, 10});

  // TensorRT does not support Clip opset 12 yet.
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// int32 Clip, run explicitly on the WebGPU EP (the 4-byte templated Clip kernel). Uses opset 12 (the
// earliest at which integer Clip is valid) to exercise the 12-12 registration; the uint32 test below
// uses opset 13 for the other registration. Mirrors the Clip_int32 reference values above.
TEST(MathOpTest, Clip_int32_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  OpTester test("Clip", 12);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<int32_t>("X", dims,
                         {-1, 0, 1,
                          -16, 12, -6,
                          -5, 2, 16});
  test.AddInput<int32_t>("min", {}, {-10});
  test.AddInput<int32_t>("max", {}, {10});
  test.AddOutput<int32_t>("Y", dims,
                          {-1, 0, 1,
                           -10, 10, -6,
                           -5, 2, 10});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// uint32 Clip, run explicitly on the WebGPU EP. Mirrors the Clip_uint32 reference values above.
TEST(MathOpTest, Clip_uint32_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  OpTester test("Clip", 13);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<uint32_t>("X", dims,
                          {0, 0, 1,
                           5, 12, 3,
                           2, 7, 16});
  test.AddInput<uint32_t>("min", {}, {3});
  test.AddInput<uint32_t>("max", {}, {10});
  test.AddOutput<uint32_t>("Y", dims,
                           {3, 3, 3,
                            5, 10, 3,
                            3, 7, 10});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// int64 Clip, run explicitly on the WebGPU EP. WebGPU has no native 64-bit integer type; the EP
// stores int64 as vec2<u32> and the dedicated ClipInt64 kernel clamps on the truncated low 32 bits
// (interpreted as i32), then sign-extends on write. Values are kept within the int32 range -- the
// realistic index/position case and where the truncated clamp matches the reference result.
TEST(MathOpTest, Clip_int64_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  OpTester test("Clip", 13);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<int64_t>("X", dims,
                         {-1, 0, 1,
                          -16, 12, -6,
                          -5, 2, 16});
  test.AddInput<int64_t>("min", {}, {-10});
  test.AddInput<int64_t>("max", {}, {10});
  test.AddOutput<int64_t>("Y", dims,
                          {-1, 0, 1,
                           -10, 10, -6,
                           -5, 2, 10});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// int64 Clip on WebGPU with min/max OUTSIDE the int32 range. ClipInt64 saturates the bounds into
// the i32 range the shader operates on, so an out-of-i32-range min/max means "no clamp on that
// side" -- the data (all within int32 range here) must pass through unchanged. Without saturation
// the raw int64 bounds would be truncated to bogus i32 values and clamp incorrectly, so this pins
// the saturate_to_i32 behavior.
TEST(MathOpTest, Clip_int64_saturate_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  OpTester test("Clip", 13);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<int64_t>("X", dims,
                         {-2147483648, -1000, -1,
                          0, 1, 1000,
                          2147483647, 42, -42});
  // -5000000000 < INT32_MIN and 5000000000 > INT32_MAX; both saturate to the i32 limits.
  test.AddInput<int64_t>("min", {}, {-5000000000LL});
  test.AddInput<int64_t>("max", {}, {5000000000LL});
  test.AddOutput<int64_t>("Y", dims,
                          {-2147483648, -1000, -1,
                           0, 1, 1000,
                           2147483647, 42, -42});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// int64 Clip on WebGPU at opset 12 with only the optional `min` input present (max omitted). This
// exercises two paths the opset-13 tests above do not: the opset 12-12 kernel registration, and the
// clip_max_tensor == nullptr branch in ClipInt64 (a missing bound means "no clamp on that side", so
// max defaults to INT32_MAX). Only the lower bound is applied; values above min pass through.
// (Opset 12 is the earliest at which integer Clip is valid; opset 11 constrains T to float types.)
TEST(MathOpTest, Clip_int64_min_only_opset12_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  OpTester test("Clip", 12);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<int64_t>("X", dims,
                         {-1, 0, 1,
                          -16, 12, -6,
                          -5, 2, 16});
  test.AddInput<int64_t>("min", {}, {-10});
  // max omitted: no upper clamp.
  test.AddOutput<int64_t>("Y", dims,
                          {-1, 0, 1,
                           -10, 12, -6,
                           -5, 2, 16});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Symmetric to the min-only test above: int64 Clip on WebGPU at opset 12 with only the optional
// `max` input present (min omitted). This covers the clip_min_tensor == nullptr branch in ClipInt64
// (a missing lower bound means "no clamp on that side", so min defaults to INT32_MIN). Only the
// upper bound is applied; values below max pass through.
TEST(MathOpTest, Clip_int64_max_only_opset12_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  OpTester test("Clip", 12);

  std::vector<int64_t> dims{3, 3};
  test.AddInput<int64_t>("X", dims,
                         {-1, 0, 1,
                          -16, 12, -6,
                          -5, 2, 16});
  test.AddOptionalInputEdge<int64_t>();  // no min
  test.AddInput<int64_t>("max", {}, {10});
  test.AddOutput<int64_t>("Y", dims,
                          {-1, 0, 1,
                           -16, 10, -6,
                           -5, 2, 10});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MathOpTest, Clip) {
  // To test NNAPI EP, we need the min/max to be in initializers
  auto run_test = [](bool min_max_are_initializer) {
    OpTester test("Clip", 11);

    std::vector<int64_t> dims{3, 3};
    test.AddInput<float>("X", dims,
                         {-1.0f, 0.0f, 1.0f,
                          -6.0f, 0.0f, 6.0f,
                          -5.4f, 2.0f, 6.0f});
    test.AddInput<float>("min", {}, {-5}, min_max_are_initializer);
    test.AddInput<float>("max", {}, {5}, min_max_are_initializer);
    test.AddOutput<float>("Y", dims,
                          {-1.0f, 0.0f, 1.0f,
                           -5.0f, 0.0f, 5.0f,
                           -5.0f, 2.0f, 5.0f});

    // TensorRT does not support Clip opset 11 yet.
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  };

  run_test(false);
  run_test(true);
}

// Use clip between [0, 6] as Relu6 to test optimized path in some  EPs, such as NNAPI and CoreML
TEST(MathOpTest, Clip_Relu6) {
  // To test NNAPI EP, we need the min/max to be in initializers
  auto run_test = [](bool min_max_are_initializer) {
    OpTester test("Clip", 11);

    std::vector<int64_t> dims{3, 3};
    test.AddInput<float>("X", dims,
                         {-1.0f, 0.0f, 1.0f,
                          -6.0f, 3.5f, 6.0f,
                          -5.4f, 2.0f, 8.0f});
    test.AddInput<float>("min", {}, {0.0f}, min_max_are_initializer);
    test.AddInput<float>("max", {}, {6.0f}, min_max_are_initializer);
    test.AddOutput<float>("Y", dims,
                          {0.0f, 0.0f, 1.0f,
                           0.0f, 3.5f, 6.0f,
                           0.0f, 2.0f, 6.0f});

    // TensorRT does not support Clip opset 11 yet.
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  };

  run_test(false);
  run_test(true);
}

// Use clip between [0, inf] as Relu to test optimized path in some EPs, such as CoreML
TEST(MathOpTest, Clip_Relu) {
  // To test NNAPI EP, we need the min/max to be in initializers
  auto run_test = [](bool min_max_are_initializer) {
    OpTester test("Clip", 11);

    std::vector<int64_t> dims{3, 3};
    test.AddInput<float>("X", dims,
                         {-1.0f, 0.0f, 1.0f,
                          -6.0f, 3.5f, 6.0f,
                          -5.4f, 2.0f, 8.0f});
    test.AddInput<float>("min", {}, {0.0f}, min_max_are_initializer);
    test.AddOutput<float>("Y", dims,
                          {0.0f, 0.0f, 1.0f,
                           0.0f, 3.5f, 6.0f,
                           0.0f, 2.0f, 8.0f});

    // TensorRT does not support Clip opset 11 yet.
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  };

  run_test(false);
  run_test(true);
}

// Use clip between [-1, 1] as Relu1 (for some EPs, such as NNAPI)
TEST(MathOpTest, Clip_Relu1) {
  // To test NNAPI EP, we need the min/max to be in initializers
  auto run_test = [](bool min_max_are_initializer) {
    OpTester test("Clip", 11);

    std::vector<int64_t> dims{3, 3};
    test.AddInput<float>("X", dims,
                         {-1.0f, 0.0f, 1.0f,
                          -6.0f, 3.5f, 6.0f,
                          -5.4f, 2.0f, 8.0f});
    test.AddInput<float>("min", {}, {-1.0f}, min_max_are_initializer);
    test.AddInput<float>("max", {}, {1.0f}, min_max_are_initializer);
    test.AddOutput<float>("Y", dims,
                          {-1.0f, 0.0f, 1.0f,
                           -1.0f, 1.0f, 1.0f,
                           -1.0f, 1.0f, 1.0f});

    // TensorRT and Tensorrt does not support Clip opset 11 yet.
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  };

  run_test(false);
  run_test(true);
}

TEST(MathOpTest, ClipDimWithZero) {
  std::vector<int64_t> dims{3, 0};  // dim with value of zero should be handled

  OpTester test("Clip", 11);
  test.AddInput<float>("X", dims, {});
  test.AddInput<float>("min", {}, {-5});
  test.AddInput<float>("max", {}, {5});
  test.AddOutput<float>("Y", dims, {});

  // Tensorrt does not support Clip opset 11 yet.
  // CoreML EP does not support empty inputs
  // QNN can't handle zero dim
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kCoreMLExecutionProvider, kQnnExecutionProvider});

  OpTester test1("Clip");  //
  test1.AddInput<float>("X", dims, {});
  test1.AddAttribute("min", -10.0f);
  test1.AddAttribute("max", 10.0f);
  test1.AddOutput<float>("Y", dims, {});
  // TRT doesn't handle this
  // CoreML EP does not support empty inputs
  // QNN can't handle zero dim
  test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kCoreMLExecutionProvider, kQnnExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
