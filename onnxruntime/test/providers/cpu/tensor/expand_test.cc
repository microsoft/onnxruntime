// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/util/math.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#if defined(ENABLE_STRIDED_TENSORS) && (defined(USE_CUDA) || defined(USE_ROCM))
#include "test/providers/kernel_compute_test_utils.h"
#endif

namespace onnxruntime {
namespace test {

TEST(ExpandOpTest, Expand_3x3) {
  OpTester test("Expand", 8);
  test.AddInput<float>("data_0", {1}, {1.0f});
  test.AddInput<int64_t>("data_1", {2}, {3, 3});
  test.AddOutput<float>("result", {3, 3},
                        {1.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f,
                         1.0f, 1.0f, 1.0f});
  test.Run();
}

TEST(ExpandOpTest, Expand_3x1) {
  OpTester test("Expand", 8);
  test.AddInput<float>("data_0", {3}, {1.0f, 2.0f, 3.0f});
  test.AddInput<int64_t>("data_1", {2}, {3, 1});
  test.AddOutput<float>("result", {3, 3},
                        {1.0f, 2.0f, 3.0f,
                         1.0f, 2.0f, 3.0f,
                         1.0f, 2.0f, 3.0f});
  test.Run();
}

TEST(ExpandOpTest, Expand_1x3) {
  OpTester test("Expand", 8);
  test.AddInput<float>("data_0", {3, 1}, {1.0f, 2.0f, 3.0f});
  test.AddInput<int64_t>("data_1", {2}, {1, 3});
  test.AddOutput<float>("result", {3, 3},
                        {1.0f, 1.0f, 1.0f,
                         2.0f, 2.0f, 2.0f,
                         3.0f, 3.0f, 3.0f});
  test.Run();
}

TEST(ExpandOpTest, Expand_3x3_int32) {
  OpTester test("Expand", 8);
  test.AddInput<int32_t>("data_0", {1}, {1});
  test.AddInput<int64_t>("data_1", {2}, {3, 3});
  test.AddOutput<int32_t>("result", {3, 3},
                          {1, 1, 1,
                           1, 1, 1,
                           1, 1, 1});
  test.Run();
}

TEST(ExpandOpTest, Expand_3x1_int32) {
  OpTester test("Expand", 8);
  test.AddInput<int32_t>("data_0", {3}, {1, 2, 3});
  test.AddInput<int64_t>("data_1", {2}, {3, 1});
  test.AddOutput<int32_t>("result", {3, 3},
                          {1, 2, 3,
                           1, 2, 3,
                           1, 2, 3});
  test.Run();
}

TEST(ExpandOpTest, Expand_1x3_int32) {
  OpTester test("Expand", 8);
  test.AddInput<int32_t>("data_0", {3, 1}, {1, 2, 3});
  test.AddInput<int64_t>("data_1", {2}, {1, 3});
  test.AddOutput<int32_t>("result", {3, 3},
                          {1, 1, 1,
                           2, 2, 2,
                           3, 3, 3});
  test.Run();
}

TEST(ExpandOpTest, Expand_3x3_int64) {
  OpTester test("Expand", 8);
  test.AddInput<int64_t>("data_0", {1}, {1});
  test.AddInput<int64_t>("data_1", {2}, {3, 3});
  test.AddOutput<int64_t>("result", {3, 3},
                          {1, 1, 1,
                           1, 1, 1,
                           1, 1, 1});
  test.Run();
}

TEST(ExpandOpTest, Expand_3x1_int64) {
  OpTester test("Expand", 8);
  test.AddInput<int64_t>("data_0", {3}, {1, 2, 3});
  test.AddInput<int64_t>("data_1", {2}, {3, 1});
  test.AddOutput<int64_t>("result", {3, 3},
                          {1, 2, 3,
                           1, 2, 3,
                           1, 2, 3});
  test.Run();
}

TEST(ExpandOpTest, Expand_1x3_int64) {
  OpTester test("Expand", 8);
  test.AddInput<int64_t>("data_0", {3, 1}, {1, 2, 3});
  test.AddInput<int64_t>("data_1", {2}, {1, 3});
  test.AddOutput<int64_t>("result", {3, 3},
                          {1, 1, 1,
                           2, 2, 2,
                           3, 3, 3});
  test.Run();
}

TEST(ExpandOpTest, Expand_3x1x3x1_int64) {
  OpTester test("Expand", 8);
  test.AddInput<int64_t>("data_0", {1, 3, 1, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  test.AddInput<int64_t>("data_1", {4}, {3, 1, 3, 1});
  test.AddOutput<int64_t>("result", {3, 3, 3, 3},
                          {1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9,
                           1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9,
                           1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9});
  test.Run();
}

TEST(ExpandOpTest, Expand_3x3_float16) {
  OpTester test("Expand", 8);
  test.AddInput<MLFloat16>("data_0", {1}, {MLFloat16(1.0f)});
  test.AddInput<int64_t>("data_1", {2}, {3, 3});
  test.AddOutput<MLFloat16>("result", {3, 3},
                            {MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(1.0f),
                             MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(1.0f),
                             MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(1.0f)});
  test.Run();
}

TEST(ExpandOpTest, Expand_3x1_float16) {
  OpTester test("Expand", 8);
  test.AddInput<MLFloat16>("data_0", {3}, {MLFloat16(1.0f), MLFloat16(2.0f), MLFloat16(3.0f)});
  test.AddInput<int64_t>("data_1", {2}, {3, 1});
  test.AddOutput<MLFloat16>("result", {3, 3},
                            {MLFloat16(1.0f), MLFloat16(2.0f), MLFloat16(3.0f),
                             MLFloat16(1.0f), MLFloat16(2.0f), MLFloat16(3.0f),
                             MLFloat16(1.0f), MLFloat16(2.0f), MLFloat16(3.0f)});
  test.Run();
}

TEST(ExpandOpTest, Expand_1x3_float16) {
  OpTester test("Expand", 8);
  test.AddInput<MLFloat16>("data_0", {3, 1}, {MLFloat16(1.0f), MLFloat16(2.0f), MLFloat16(3.0f)});
  test.AddInput<int64_t>("data_1", {2}, {1, 3});
  test.AddOutput<MLFloat16>("result", {3, 3},
                            {MLFloat16(1.0f), MLFloat16(1.0f), MLFloat16(1.0f),
                             MLFloat16(2.0f), MLFloat16(2.0f), MLFloat16(2.0f),
                             MLFloat16(3.0f), MLFloat16(3.0f), MLFloat16(3.0f)});
  test.Run();
}

TEST(ExpandOpTest, Expand_2x2x1x2x1_float) {
  OpTester test("Expand", 8);
  test.AddInput<float>("data_0", {2, 2, 1, 2, 1}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
  test.AddInput<int64_t>("data_1", {5}, {1, 2, 2, 2, 2});
  test.AddOutput<float>("result", {2, 2, 2, 2, 2},
                        {1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f,
                         3.0f, 3.0f, 4.0f, 4.0f, 3.0f, 3.0f, 4.0f, 4.0f,
                         5.0f, 5.0f, 6.0f, 6.0f, 5.0f, 5.0f, 6.0f, 6.0f,
                         7.0f, 7.0f, 8.0f, 8.0f, 7.0f, 7.0f, 8.0f, 8.0f});
  test.Run();
}

#ifndef USE_TENSORRT
TEST(ExpandOpTest, Expand_scalar_float) {
  OpTester test("Expand", 8);
  test.AddInput<float>("data_0", {}, {3.0f});
  test.AddInput<int64_t>("data_1", {0}, {});
  test.AddOutput<float>("result", {}, {3.0f});
  test.Run();
}
#endif

TEST(ExpandOpTest, Expand_scalar_int32) {
  OpTester test("Expand", 8);
  test.AddInput<int32_t>("data_0", {}, {9});
  test.AddInput<int64_t>("data_1", {3}, {2, 3, 4});
  test.AddOutput<int32_t>("result", {2, 3, 4},
                          {9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                           9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9});
  test.Run();
}

#if defined(ENABLE_STRIDED_TENSORS) && (defined(USE_CUDA) || defined(USE_ROCM))
TEST(ExpandOpTest, Strided) {
#ifdef USE_CUDA
  const char* provider = kCudaExecutionProvider;
#else  // USE_ROCM
  const char* provider = kRocmExecutionProvider;
#endif
  // Generate contiguous output.
  {
    KernelComputeTester test("Expand", provider);
    test.AddInput<float>("input_0", {3, 1}, {1.f, 2.f, 3.f});
    test.AddInput<int64_t>("input_1", {2}, {1, 3}, {}, true);
    test.AddOutput<float>("output", {3, 3}, {1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 3.f, 3.f, 3.f});
    test.Run();
  }

  // Strided 2D.
  {
    KernelComputeTester test("Expand", provider);
    test.AddInput<float>("input_0", {3, 1}, {1.f, 2.f, 3.f});
    test.AddInput<int64_t>("input_1", {2}, {1, 3}, {}, true);
    test.AddOutput<float>("output", {3, 3}, {1.f, 2.f, 3.f}, {1, 0});
    test.Run({0});
  }

  // Strided 3D.
  {
    KernelComputeTester test("Expand", provider);
    test.AddInput<float>("input_0", {1, 3, 1}, {1.f, 2.f, 3.f});
    test.AddInput<int64_t>("input_1", {3}, {2, 1, 3}, {}, true);
    test.AddOutput<float>("output", {2, 3, 3}, {1.f, 2.f, 3.f}, {0, 1, 0});
    test.Run({0});
  }

  // Strided 4D.
  {
    KernelComputeTester test("Expand", provider);
    test.AddInput<float>("input_0", {1, 1, 3, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f});
    test.AddInput<int64_t>("input_1", {4}, {2, 3, 1, 1}, {}, true);
    test.AddOutput<float>("output", {2, 3, 3, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}, {0, 0, 3, 1});
    test.Run({0});
  }

  // Strided 2D -> 3D.
  {
    KernelComputeTester test("Expand", provider);
    test.AddInput<float>("input_0", {1, 3}, {1.f, 2.f, 3.f});
    test.AddInput<int64_t>("input_1", {3}, {2, 3, 1}, {}, true);
    test.AddOutput<float>("output", {2, 3, 3}, {1.f, 2.f, 3.f}, {0, 0, 1});
    test.Run({0});
  }

  // Strided 1Element -> 4D.
  {
    KernelComputeTester test("Expand", provider);
    test.AddInput<float>("input_0", {1}, {1.f});
    test.AddInput<int64_t>("input_1", {4}, {2, 3, 3, 3}, {}, true);
    test.AddOutput<float>("output", {2, 3, 3, 3}, {1.f}, {0, 0, 0, 0});
    test.Run({0});
  }
}
#endif

}  // namespace test
}  // namespace onnxruntime
