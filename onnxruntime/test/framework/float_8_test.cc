// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_FLOAT8_TYPES)

#include <vector>

#include "core/framework/float8.h"
#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "test_utils.h"
#include "gtest/gtest.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace test {

TEST(Float8_Tests, CastE4M3FN) {
  std::vector<std::pair<float, float>> cases{
      std::pair<float, float>(0.00439453125, 0.00390625),
      std::pair<float, float>(0.005859375, 0.005859375),
      std::pair<float, float>(0.005759375, 0.005859375),
      std::pair<float, float>(0.0046875, 0.00390625),
      std::pair<float, float>(0.001953125, 0.001953125),
      std::pair<float, float>(0.0029296875, 0.00390625),
      std::pair<float, float>(0.002053125, 0.001953125),
      std::pair<float, float>(0.00234375, 0.001953125),
      std::pair<float, float>(0.0087890625, 0.0078125),
      std::pair<float, float>(0.001171875, 0.001953125),
      std::pair<float, float>(1.8131605, 1.875)};
  for (auto it : cases) {
    auto f8 = onnxruntime::Float8E4M3FN(it.first);
    auto f8_32 = f8.ToFloat();
    EXPECT_EQ(it.second, f8_32);
  }
}

union float_bits {
  uint32_t bits;
  float val;
};

TEST(Float8_Tests, NanE4M3FN) {
  EXPECT_EQ(onnxruntime::Float8E4M3FN((float_bits{0x7F800000}).val).val, static_cast<uint8_t>(0x7E));
  EXPECT_EQ(onnxruntime::Float8E4M3FN((float_bits{0xFF800000}).val).val, static_cast<uint8_t>(0xFE));
  EXPECT_EQ(onnxruntime::Float8E4M3FN((float_bits{0x7F800000}).val, false).val, static_cast<uint8_t>(0x7F));
  EXPECT_EQ(onnxruntime::Float8E4M3FN((float_bits{0xFF800000}).val, false).val, static_cast<uint8_t>(0xFF));
  EXPECT_EQ(onnxruntime::Float8E4M3FN((float_bits{0x7F800001}).val).val, static_cast<uint8_t>(0x7F));
  EXPECT_EQ(onnxruntime::Float8E4M3FN((float_bits{0xFF800001}).val).val, static_cast<uint8_t>(0xFF));
  // 0x7FC00000 is the value used by numpy.
  EXPECT_EQ(onnxruntime::Float8E4M3FN((float_bits{0x7FC00000}).val).val, static_cast<uint8_t>(0x7F));
  EXPECT_EQ(onnxruntime::Float8E4M3FN((float_bits{0xFFC00000}).val).val, static_cast<uint8_t>(0xFF));
  // small negative values should round to negative zero
  EXPECT_EQ(onnxruntime::Float8E4M3FN(-0.00000001f).ToFloat(), -0.0f);
  EXPECT_EQ(onnxruntime::Float8E4M3FN(-0.00000001f).val, static_cast<uint8_t>(0x80));
}

TEST(Float8_Tests, NanE4M3FNUZ) {
  EXPECT_EQ(onnxruntime::Float8E4M3FNUZ((float_bits{0x7F800000}).val).val, static_cast<uint8_t>(0x7F));
  EXPECT_EQ(onnxruntime::Float8E4M3FNUZ((float_bits{0xFF800000}).val).val, static_cast<uint8_t>(0xFF));
  EXPECT_EQ(onnxruntime::Float8E4M3FNUZ((float_bits{0x7F800000}).val, false).val, static_cast<uint8_t>(0x80));
  EXPECT_EQ(onnxruntime::Float8E4M3FNUZ((float_bits{0xFF800000}).val, false).val, static_cast<uint8_t>(0x80));
  EXPECT_EQ(onnxruntime::Float8E4M3FNUZ((float_bits{0x7F800001}).val).val, static_cast<uint8_t>(0x80));
  EXPECT_EQ(onnxruntime::Float8E4M3FNUZ((float_bits{0xFF800001}).val).val, static_cast<uint8_t>(0x80));
  // 0x7FC00000 is the value used by numpy.
  EXPECT_EQ(onnxruntime::Float8E4M3FNUZ((float_bits{0x7FC00000}).val).val, static_cast<uint8_t>(0x80));
  EXPECT_EQ(onnxruntime::Float8E4M3FNUZ((float_bits{0xFFC00000}).val).val, static_cast<uint8_t>(0x80));
  // small negative values should round to zero
  EXPECT_EQ(onnxruntime::Float8E4M3FNUZ(-0.00000001f).ToFloat(), 0.0f);
  EXPECT_EQ(onnxruntime::Float8E4M3FNUZ(-0.00000001f).val, static_cast<uint8_t>(0x00));
  EXPECT_EQ(onnxruntime::Float8E4M3FNUZ(-0x1.0p-11f).ToFloat(), 0.0f);
  EXPECT_EQ(onnxruntime::Float8E4M3FNUZ(-0x1.0p-11f).val, static_cast<uint8_t>(0x00));
}

TEST(Float8_Tests, NanE5M2) {
  EXPECT_EQ(onnxruntime::Float8E5M2((float_bits{0x7F800000}).val).val, static_cast<uint8_t>(0x7B));
  EXPECT_EQ(onnxruntime::Float8E5M2((float_bits{0xFF800000}).val).val, static_cast<uint8_t>(0xFB));
  EXPECT_EQ(onnxruntime::Float8E5M2((float_bits{0x7F800000}).val, false).val, static_cast<uint8_t>(0x7C));
  EXPECT_EQ(onnxruntime::Float8E5M2((float_bits{0xFF800000}).val, false).val, static_cast<uint8_t>(0xFC));
  EXPECT_EQ(onnxruntime::Float8E5M2((float_bits{0x7F800001}).val).val, static_cast<uint8_t>(0x7F));
  EXPECT_EQ(onnxruntime::Float8E5M2((float_bits{0xFF800001}).val).val, static_cast<uint8_t>(0xFF));
  // 0x7FC00000 is the value used by numpy.
  EXPECT_EQ(onnxruntime::Float8E5M2((float_bits{0x7FC00000}).val).val, static_cast<uint8_t>(0x7F));
  EXPECT_EQ(onnxruntime::Float8E5M2((float_bits{0xFFC00000}).val).val, static_cast<uint8_t>(0xFF));
  // small negative values should round to negative zero
  EXPECT_EQ(onnxruntime::Float8E5M2(-0.00000001f).ToFloat(), 0.0f);
  EXPECT_EQ(onnxruntime::Float8E5M2(-0.00000001f).val, static_cast<uint8_t>(0x80));
}

TEST(Float8_Tests, NanE5M2FNUZ) {
  EXPECT_EQ(onnxruntime::Float8E5M2FNUZ((float_bits{0x7F800000}).val).val, static_cast<uint8_t>(0x7F));
  EXPECT_EQ(onnxruntime::Float8E5M2FNUZ((float_bits{0xFF800000}).val).val, static_cast<uint8_t>(0xFF));
  EXPECT_EQ(onnxruntime::Float8E5M2FNUZ((float_bits{0x7F800000}).val, false).val, static_cast<uint8_t>(0x80));
  EXPECT_EQ(onnxruntime::Float8E5M2FNUZ((float_bits{0xFF800000}).val, false).val, static_cast<uint8_t>(0x80));
  EXPECT_EQ(onnxruntime::Float8E5M2FNUZ((float_bits{0x7F800001}).val).val, static_cast<uint8_t>(0x80));
  EXPECT_EQ(onnxruntime::Float8E5M2FNUZ((float_bits{0xFF800001}).val).val, static_cast<uint8_t>(0x80));
  // 0x7FC00000 is the value used by numpy.
  EXPECT_EQ(onnxruntime::Float8E5M2FNUZ((float_bits{0x7FC00000}).val).val, static_cast<uint8_t>(0x80));
  EXPECT_EQ(onnxruntime::Float8E5M2FNUZ((float_bits{0xFFC00000}).val).val, static_cast<uint8_t>(0x80));
  // small negative values should round to zero
  EXPECT_EQ(onnxruntime::Float8E5M2FNUZ(-0.00000001f).ToFloat(), 0.0f);
  EXPECT_EQ(onnxruntime::Float8E5M2FNUZ(-0.00000001f).val, static_cast<uint8_t>(0x00));
  EXPECT_EQ(onnxruntime::Float8E5M2FNUZ(-0x1.0p-18f).ToFloat(), 0.0f);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // DISABLE_FLOAT8_TYPES
