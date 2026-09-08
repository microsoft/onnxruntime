// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/common/common.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// GatherQuantized gathers rows from an FP8 block-scaled constant table (no zero point, since FP8
// quantization is symmetric) and dequantizes them: output[...] = float(data[...]) * scales[block(...)].

TEST(GatherQuantizedOpTest, BasicPerRowScale) {
  // data: [4, 4] FP8 E4M3FN. block_size = 0 -> one scale per row (quantize_axis = 1, the whole row).
  std::vector<Float8E4M3FN> data = {
      Float8E4M3FN(1.0f), Float8E4M3FN(2.0f), Float8E4M3FN(4.0f), Float8E4M3FN(8.0f),
      Float8E4M3FN(-1.0f), Float8E4M3FN(-2.0f), Float8E4M3FN(-4.0f), Float8E4M3FN(-8.0f),
      Float8E4M3FN(1.0f), Float8E4M3FN(1.0f), Float8E4M3FN(1.0f), Float8E4M3FN(1.0f),
      Float8E4M3FN(2.0f), Float8E4M3FN(2.0f), Float8E4M3FN(2.0f), Float8E4M3FN(2.0f)};
  std::vector<float> scales = {1.0f, 0.5f, 2.0f, 3.0f};  // shape [4, 1]
  std::vector<int64_t> indices = {1, 3};
  std::vector<float> expected = {
      -0.5f, -1.0f, -2.0f, -4.0f,
      6.0f, 6.0f, 6.0f, 6.0f};

  OpTester test("GatherQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float8E4M3FN>("data", {4, 4}, data);
  test.AddInput<int64_t>("indices", {2}, indices);
  test.AddInput<float>("scales", {4, 1}, scales);
  test.AddOutput<float>("output", {2, 4}, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kCudaNHWCExecutionProvider,
                                                        kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(GatherQuantizedOpTest, SubRowBlockScale) {
  // data: [1, 4] FP8 E4M3FN, block_size = 2 -> 2 blocks of 2 elements each along quantize_axis = 1.
  std::vector<Float8E4M3FN> data = {
      Float8E4M3FN(1.0f), Float8E4M3FN(2.0f), Float8E4M3FN(4.0f), Float8E4M3FN(8.0f)};
  std::vector<float> scales = {1.0f, 0.5f};  // shape [1, 2]
  std::vector<int64_t> indices = {0};
  std::vector<float> expected = {1.0f, 2.0f, 2.0f, 4.0f};

  OpTester test("GatherQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 2);
  test.AddInput<Float8E4M3FN>("data", {1, 4}, data);
  test.AddInput<int64_t>("indices", {1}, indices);
  test.AddInput<float>("scales", {1, 2}, scales);
  test.AddOutput<float>("output", {1, 4}, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kCudaNHWCExecutionProvider,
                                                        kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(GatherQuantizedOpTest, Float16Output) {
  std::vector<Float8E4M3FN> data = {
      Float8E4M3FN(1.0f), Float8E4M3FN(2.0f),
      Float8E4M3FN(4.0f), Float8E4M3FN(8.0f)};
  std::vector<MLFloat16> scales = {MLFloat16(1.0f), MLFloat16(2.0f)};  // shape [2, 1]
  std::vector<int32_t> indices = {0, 1};
  std::vector<MLFloat16> expected = {
      MLFloat16(1.0f), MLFloat16(2.0f),
      MLFloat16(8.0f), MLFloat16(16.0f)};

  OpTester test("GatherQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float8E4M3FN>("data", {2, 2}, data);
  test.AddInput<int32_t>("indices", {2}, indices);
  test.AddInput<MLFloat16>("scales", {2, 1}, scales);
  test.AddOutput<MLFloat16>("output", {2, 2}, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kCudaNHWCExecutionProvider,
                                                        kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(GatherQuantizedOpTest, InvalidBlockSizeThrows) {
  std::vector<Float8E4M3FN> data = {Float8E4M3FN(1.0f), Float8E4M3FN(2.0f)};
  std::vector<float> scales = {1.0f};
  std::vector<int64_t> indices = {0};

  OpTester test("GatherQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 8);  // not a power of 2 >= 16, and not 0
  test.AddInput<Float8E4M3FN>("data", {1, 2}, data);
  test.AddInput<int64_t>("indices", {1}, indices);
  test.AddInput<float>("scales", {1, 1}, scales);
  test.AddOutput<float>("output", {1, 2}, {1.0f, 2.0f});
  test.Run(OpTester::ExpectResult::kExpectFailure, "", {kCudaExecutionProvider, kCudaNHWCExecutionProvider,
                                                        kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime
