// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(DISABLE_FLOAT4_TYPES)

#include <vector>
#include <map>

#ifdef USE_CUDA
// Needed for CUDA_VERSION check in float4.h
#include <cuda.h>
#endif

#include "core/framework/float4.h"
#include "test/test_environment.h"
#include "test_utils.h"
#include "gtest/gtest.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {
namespace test {

TEST(Float4_Tests, BasicFloatConversion) {
  std::map<std::pair<float, float>, std::pair<float, float>> cases{
      {std::pair<float, float>(0.f, -0.f),
       std::pair<float, float>(0.f, -0.f)},
      {std::pair<float, float>(0.5f, -0.5f),
       std::pair<float, float>(0.5f, -0.5f)},
      {std::pair<float, float>(1.5323f, -1.932f),
       std::pair<float, float>(1.5f, -2.f)},
      {std::pair<float, float>(2.173f, 3.5f),
       std::pair<float, float>(2.f, 4.f)},
      {std::pair<float, float>(6.5f, -9.f),
       std::pair<float, float>(6.f, -6.f)},
      {std::pair<float, float>(-2.5f, 2.5),
       std::pair<float, float>(-2.f, 2.f)},
      {std::pair<float, float>(std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity()),
       std::pair<float, float>(6.f, -6.f)}};

  for (auto& c : cases) {
    auto f4e4m2_instance = Float4E2M1x2(c.first.first, c.first.second);
    auto f_cvt_returned = f4e4m2_instance.ToFloat2();
    EXPECT_EQ(f_cvt_returned.first, c.second.first);
    EXPECT_EQ(f_cvt_returned.second, c.second.second);
  }

  // NaNs
  auto NaNs_converted = Float4E2M1x2(std::numeric_limits<float>::quiet_NaN(),
                                     -std::numeric_limits<float>::quiet_NaN())
                            .ToFloat2();

  EXPECT_EQ(NaNs_converted.first, 6.f);
  EXPECT_EQ(NaNs_converted.second, 6.f);
}
TEST(Float4_Tests, BitRepresentationChecks) {
  // FromBits test
  std::pair<float, float> pair;
  pair = Float4E2M1x2(0x87, Float4E2M1x2::FromBits()).ToFloat2();
  EXPECT_EQ(pair.first, 6.f);
  EXPECT_EQ(pair.second, -0.f);

  pair = Float4E2M1x2(0x7F, Float4E2M1x2::FromBits()).ToFloat2();
  EXPECT_EQ(pair.first, -6.f);
  EXPECT_EQ(pair.second, 6.f);

  // Bit representation test
  uint8_t bits = Float4E2M1x2(-6.f, 6.f).ToBits();
  // First nibble is the second value and the second nibble is the first value
  EXPECT_EQ((bits & 0xF0) >> 4, 0x07);  // -6
  EXPECT_EQ((bits & 0x0F), 0x0F);       // +6
}

TEST(Float4_Tests, PackingAndUnpacking) {
  {
    // Unpack 5 FP4 (odd count) elements
    std::vector<Float4E2M1x2> packed{Float4E2M1x2(1.f, -0.5f),
                                     Float4E2M1x2(4.f, -6.f),
                                     Float4E2M1x2(3.f, 0.f)};  // padding 0
    std::vector<float> unpacked(5, -1.f);

    Float4E2M1x2::UnpackFloat4E2M1ToFloat(packed.data(), unpacked.data(), 5);
    EXPECT_EQ(unpacked[0], packed[0].ToFloat2().first);
    EXPECT_EQ(unpacked[1], packed[0].ToFloat2().second);
    EXPECT_EQ(unpacked[2], packed[1].ToFloat2().first);
    EXPECT_EQ(unpacked[3], packed[1].ToFloat2().second);
    EXPECT_EQ(unpacked[4], packed[2].ToFloat2().first);
  }

  {
    // Unpack 6 FP4 (even count) elements
    std::vector<Float4E2M1x2> packed{Float4E2M1x2(1.f, -0.5f),
                                     Float4E2M1x2(4.f, -6.f),
                                     Float4E2M1x2(3.f, -3.f)};
    std::vector<float> unpacked(6, -1.f);

    Float4E2M1x2::UnpackFloat4E2M1ToFloat(packed.data(), unpacked.data(), 6);
    EXPECT_EQ(unpacked[0], packed[0].ToFloat2().first);
    EXPECT_EQ(unpacked[1], packed[0].ToFloat2().second);
    EXPECT_EQ(unpacked[2], packed[1].ToFloat2().first);
    EXPECT_EQ(unpacked[3], packed[1].ToFloat2().second);
    EXPECT_EQ(unpacked[4], packed[2].ToFloat2().first);
    EXPECT_EQ(unpacked[5], packed[2].ToFloat2().second);
  }

  {
    // Pack 5 float (odd count) elements
    std::vector<float> unpacked{1.f, -0.5f, 4.f, -6.f, 3.f, 0.f};
    std::vector<Float4E2M1x2> packed(3);

    Float4E2M1x2::PackFloatToFloat4E2M1(unpacked.data(), packed.data(), 5);
    EXPECT_EQ(Float4E2M1x2(unpacked[0], unpacked[1]), packed[0]);
    EXPECT_EQ(Float4E2M1x2(unpacked[2], unpacked[3]), packed[1]);
    EXPECT_EQ(Float4E2M1x2(unpacked[4], 0), packed[2]);  // padding 0
  }

  {
    // Pack 6 float (even count) elements
    std::vector<float> unpacked{1.f, -0.5f, 4.f, -6.f, 3.f, 8.f};
    std::vector<Float4E2M1x2> packed(3);

    Float4E2M1x2::PackFloatToFloat4E2M1(unpacked.data(), packed.data(), 6);
    EXPECT_EQ(Float4E2M1x2(unpacked[0], unpacked[1]), packed[0]);
    EXPECT_EQ(Float4E2M1x2(unpacked[2], unpacked[3]), packed[1]);
    EXPECT_EQ(Float4E2M1x2(unpacked[4], unpacked[5]), packed[2]);
  }
}

TEST(Float4_Tests, TestLimits) {
  EXPECT_FALSE(std::numeric_limits<onnxruntime::Float4E2M1x2>::has_infinity);
  EXPECT_FALSE(std::numeric_limits<onnxruntime::Float4E2M1x2>::has_quiet_NaN);
  EXPECT_FALSE(std::numeric_limits<onnxruntime::Float4E2M1x2>::has_signaling_NaN);

  EXPECT_EQ(std::numeric_limits<onnxruntime::Float4E2M1x2>::min(),
            Float4E2M1x2(0x22, onnxruntime::Float4E2M1x2::FromBits()));
  EXPECT_EQ(std::numeric_limits<onnxruntime::Float4E2M1x2>::max(),
            Float4E2M1x2(0x77, onnxruntime::Float4E2M1x2::FromBits()));
  EXPECT_EQ(std::numeric_limits<onnxruntime::Float4E2M1x2>::lowest(),
            Float4E2M1x2(0xFF, onnxruntime::Float4E2M1x2::FromBits()));
  EXPECT_EQ(std::numeric_limits<onnxruntime::Float4E2M1x2>::denorm_min(),
            Float4E2M1x2(0x11, onnxruntime::Float4E2M1x2::FromBits()));
}

}  // namespace test
}  // namespace onnxruntime

#endif  // DISABLE_FLOAT4_TYPES
