// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "core/common/common.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// GatherFpQuantized gathers rows from an FP8 or FP4 block-scaled constant table (no zero point, since
// FP8/FP4 quantization is symmetric) and dequantizes them: output[...] = float(data[...]) * scales[block(...)].

#if !defined(DISABLE_FLOAT8_TYPES)
TEST(GatherFpQuantizedOpTest, BasicPerRowScale) {
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

  OpTester test("GatherFpQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float8E4M3FN>("data", {4, 4}, data);
  test.AddInput<int64_t>("indices", {2}, indices);
  test.AddInput<float>("scales", {4, 1}, scales);
  test.AddOutput<float>("output", {2, 4}, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kCudaNHWCExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(GatherFpQuantizedOpTest, GlobalPerTensorScale) {
  // data: [4, 4] FP8 E4M3FN. scales has shape [1, 1]: a single global scale for the whole table,
  // broadcast along both gather_axis (0) and quantize_axis (1). This mirrors a FP8-quantized
  // embedding table that uses one scalar `weight_scale` shared by every row (e.g. HF's
  // FP8Embedding: `rows.to(weight_scale.dtype) * weight_scale`, where `weight_scale` has shape (1,)).
  std::vector<Float8E4M3FN> data = {
      Float8E4M3FN(1.0f), Float8E4M3FN(2.0f), Float8E4M3FN(4.0f), Float8E4M3FN(8.0f),
      Float8E4M3FN(-1.0f), Float8E4M3FN(-2.0f), Float8E4M3FN(-4.0f), Float8E4M3FN(-8.0f),
      Float8E4M3FN(1.0f), Float8E4M3FN(1.0f), Float8E4M3FN(1.0f), Float8E4M3FN(1.0f),
      Float8E4M3FN(2.0f), Float8E4M3FN(2.0f), Float8E4M3FN(2.0f), Float8E4M3FN(2.0f)};
  std::vector<float> scales = {0.5f};  // shape [1, 1], one value for the entire tensor
  std::vector<int64_t> indices = {1, 3};
  std::vector<float> expected = {
      -0.5f, -1.0f, -2.0f, -4.0f,
      1.0f, 1.0f, 1.0f, 1.0f};

  OpTester test("GatherFpQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float8E4M3FN>("data", {4, 4}, data);
  test.AddInput<int64_t>("indices", {2}, indices);
  test.AddInput<float>("scales", {1, 1}, scales);
  test.AddOutput<float>("output", {2, 4}, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kCudaNHWCExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(GatherFpQuantizedOpTest, SubRowBlockScale) {
  // data: [1, 32] FP8 E4M3FN, block_size = 16 -> 2 blocks of 16 elements each along quantize_axis = 1.
  // (block_size must be 0 or a power of 2 >= 16, per the operator contract.)
  std::vector<Float8E4M3FN> data(32);
  for (int i = 0; i < 16; ++i) {
    data[static_cast<size_t>(i)] = Float8E4M3FN(1.0f);
  }
  for (int i = 16; i < 32; ++i) {
    data[static_cast<size_t>(i)] = Float8E4M3FN(4.0f);
  }
  std::vector<float> scales = {1.0f, 0.5f};  // shape [1, 2]: one scale per 16-element block
  std::vector<int64_t> indices = {0};
  std::vector<float> expected(32);
  for (int i = 0; i < 16; ++i) {
    expected[static_cast<size_t>(i)] = 1.0f;  // block 0: 1.0 * 1.0
  }
  for (int i = 16; i < 32; ++i) {
    expected[static_cast<size_t>(i)] = 2.0f;  // block 1: 4.0 * 0.5
  }

  OpTester test("GatherFpQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 16);
  test.AddInput<Float8E4M3FN>("data", {1, 32}, data);
  test.AddInput<int64_t>("indices", {1}, indices);
  test.AddInput<float>("scales", {1, 2}, scales);
  test.AddOutput<float>("output", {1, 32}, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kCudaNHWCExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(GatherFpQuantizedOpTest, Float16Output) {
  std::vector<Float8E4M3FN> data = {
      Float8E4M3FN(1.0f), Float8E4M3FN(2.0f),
      Float8E4M3FN(4.0f), Float8E4M3FN(8.0f)};
  std::vector<MLFloat16> scales = {MLFloat16(1.0f), MLFloat16(2.0f)};  // shape [2, 1]
  std::vector<int32_t> indices = {0, 1};
  std::vector<MLFloat16> expected = {
      MLFloat16(1.0f), MLFloat16(2.0f),
      MLFloat16(8.0f), MLFloat16(16.0f)};

  OpTester test("GatherFpQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float8E4M3FN>("data", {2, 2}, data);
  test.AddInput<int32_t>("indices", {2}, indices);
  test.AddInput<MLFloat16>("scales", {2, 1}, scales);
  test.AddOutput<MLFloat16>("output", {2, 2}, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kCudaNHWCExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(GatherFpQuantizedOpTest, InvalidBlockSizeThrows) {
  std::vector<Float8E4M3FN> data = {Float8E4M3FN(1.0f), Float8E4M3FN(2.0f)};
  std::vector<float> scales = {1.0f};
  std::vector<int64_t> indices = {0};

  OpTester test("GatherFpQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 8);  // not a power of 2 >= 16, and not 0
  test.AddInput<Float8E4M3FN>("data", {1, 2}, data);
  test.AddInput<int64_t>("indices", {1}, indices);
  test.AddInput<float>("scales", {1, 1}, scales);
  test.AddOutput<float>("output", {1, 2}, {1.0f, 2.0f});
  test.Run(OpTester::ExpectResult::kExpectFailure, "", {kCudaExecutionProvider, kCudaNHWCExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}
#endif  // !defined(DISABLE_FLOAT8_TYPES)

#if !defined(DISABLE_FLOAT4_TYPES)
TEST(GatherFpQuantizedOpTest, Fp4BasicPerRowScale) {
  // data: [2, 4] FP4 E2M1, packed 2 logical elements per byte (logical shape is unaffected by packing,
  // same convention as the existing UInt4x2/Int4x2 sub-byte tensor types).
  // row0 = [1, 2, 4, 6], row1 = [-1, -2, -4, -6]; block_size = 0 -> one scale per row.
  std::vector<Float4E2M1x2> data = {
      Float4E2M1x2(1.0f, 2.0f), Float4E2M1x2(4.0f, 6.0f),
      Float4E2M1x2(-1.0f, -2.0f), Float4E2M1x2(-4.0f, -6.0f)};
  std::vector<float> scales = {1.0f, 0.5f};  // shape [2, 1]
  std::vector<int64_t> indices = {0, 1};
  std::vector<float> expected = {
      1.0f, 2.0f, 4.0f, 6.0f,
      -0.5f, -1.0f, -2.0f, -3.0f};

  OpTester test("GatherFpQuantized", 1, kMSDomain);
  test.AddAttribute<int64_t>("gather_axis", 0);
  test.AddAttribute<int64_t>("quantize_axis", 1);
  test.AddAttribute<int64_t>("block_size", 0);
  test.AddInput<Float4E2M1x2>("data", {2, 4}, data);
  test.AddInput<int64_t>("indices", {2}, indices);
  test.AddInput<float>("scales", {2, 1}, scales);
  test.AddOutput<float>("output", {2, 4}, expected);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kCudaNHWCExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}
#endif  // !defined(DISABLE_FLOAT4_TYPES)

}  // namespace test
}  // namespace onnxruntime
