// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

TEST(TensorOpTest, DepthToSpaceTest_WebGPU_DefaultMode1) {
  OpTester test("DepthToSpace", 11);
  constexpr int64_t blocksize = 2;
  test.AddAttribute("blocksize", blocksize);

  constexpr int64_t N = 1, C = 8, H = 1, W = 1;
  std::vector<float> X = {0, 9, 18, 27, 36, 45, 54, 63};

  test.AddInput<float>("input", {N, C, H, W}, X);

  std::vector<float> result = {0, 18, 36, 54, 9, 27, 45, 63};

  test.AddOutput<float>("output", {1, 2, 2, 2}, result);
  test.Run();
}

TEST(TensorOpTest, DepthToSpaceTest_WebGPU_DefaultMode2) {
  OpTester test("DepthToSpace", 11);
  constexpr int64_t blocksize = 2;
  test.AddAttribute("blocksize", blocksize);

  constexpr int64_t N = 2, C = 8, H = 1, W = 2;
  std::vector<float> X = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                          29, 30, 31};

  test.AddInput<float>("input", {N, C, H, W}, X);

  std::vector<float> result = {0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15, 16, 20, 17, 21, 24, 28, 25, 29, 18, 22, 19, 23, 26,
                               30, 27, 31};

  test.AddOutput<float>("output", {2, 2, 2, 4}, result);
  test.Run();
}

TEST(TensorOpTest, DepthToSpaceTest_WebGPU_DCR1) {
  OpTester test("DepthToSpace", 11);
  constexpr int64_t blocksize = 2;
  test.AddAttribute("blocksize", blocksize);
  test.AddAttribute("mode", "DCR");

  constexpr int64_t N = 1, C = 8, H = 1, W = 1;
  std::vector<float> X = {0, 9, 18, 27, 36, 45, 54, 63};

  test.AddInput<float>("input", {N, C, H, W}, X);

  std::vector<float> result = {0, 18, 36, 54, 9, 27, 45, 63};

  test.AddOutput<float>("output", {1, 2, 2, 2}, result);
  test.Run();
}

TEST(TensorOpTest, DepthToSpaceTest_WebGPU_DCR2) {
  OpTester test("DepthToSpace", 11);
  constexpr int64_t blocksize = 2;
  test.AddAttribute("blocksize", blocksize);
  test.AddAttribute("mode", "DCR");

  constexpr int64_t N = 2, C = 8, H = 1, W = 2;
  std::vector<float> X = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                          29, 30, 31};

  test.AddInput<float>("input", {N, C, H, W}, X);

  std::vector<float> result = {0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15, 16, 20, 17, 21, 24, 28, 25, 29, 18, 22, 19, 23, 26,
                               30, 27, 31};

  test.AddOutput<float>("output", {2, 2, 2, 4}, result);
  test.Run();
}

TEST(TensorOpTest, DepthToSpaceTest_WebGPU_CRD1) {
  OpTester test("DepthToSpace", 11);
  constexpr int64_t blocksize = 2;
  test.AddAttribute("blocksize", blocksize);
  test.AddAttribute("mode", "CRD");

  constexpr int64_t N = 1, C = 8, H = 1, W = 1;
  std::vector<float> X = {0, 9, 18, 27, 36, 45, 54, 63};

  test.AddInput<float>("input", {N, C, H, W}, X);

  std::vector<float> result = {0, 9, 18, 27, 36, 45, 54, 63};

  test.AddOutput<float>("output", {1, 2, 2, 2}, result);
  test.Run();
}

TEST(TensorOpTest, DepthToSpaceTest_WebGPU_CRD2) {
  OpTester test("DepthToSpace", 11);
  constexpr int64_t blocksize = 2;
  test.AddAttribute("blocksize", blocksize);
  test.AddAttribute("mode", "CRD");

  constexpr int64_t N = 2, C = 8, H = 1, W = 2;
  std::vector<float> X = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                          29, 30, 31};

  test.AddInput<float>("input", {N, C, H, W}, X);

  std::vector<float> result = {0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15, 16, 18, 17, 19, 20, 22, 21, 23, 24, 26, 25, 27, 28,
                               30, 29, 31};

  test.AddOutput<float>("output", {2, 2, 2, 4}, result);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
