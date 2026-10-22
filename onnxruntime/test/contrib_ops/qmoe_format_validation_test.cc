#include <array>
#include <limits>
#include <vector>

#include "gtest/gtest.h"
#include "contrib_ops/cpu/moe/moe_helper.h"

namespace onnxruntime::test {
namespace {
using namespace contrib::moe_helper;

template <typename ScaleT>
struct Int3Content {
  std::array<int64_t, 3> shape{2, 128, 128};
  int64_t block = 32;
  std::vector<uint8_t> weights = std::vector<uint8_t>(256 * 48, 0xff);
  std::vector<ScaleT> scales = std::vector<ScaleT>(256 * 4, ScaleT(1.0f));
  std::vector<uint8_t> zp = std::vector<uint8_t>(256 * 2);

  void SetBlock(int64_t value, int64_t blocks) {
    block = value;
    shape[2] = value * blocks;
    weights.assign(256 * PackedByteCountWithPadding(shape[2], 3), 0xff);
    scales.assign(256 * blocks, ScaleT(0.0f));
    zp.assign(256 * PackedByteCountWithPadding(blocks, 3), 0);
    const size_t stride = static_cast<size_t>(PackedByteCountWithPadding(blocks, 3));
    for (size_t row = 0; row < 256; ++row) {
      for (int64_t index = 0; index < blocks; ++index) {
        const size_t bit = static_cast<size_t>(index * 3 + 2);
        zp[row * stride + bit / 8] |= static_cast<uint8_t>(1u << (bit % 8));
      }
    }
  }

  Status Validate(bool present = true) {
    return ValidateInt3ProjectionContent<ScaleT>(weights, scales, zp, present, shape, block);
  }
};

TEST(QMoEFormatValidationTest, ValidHalfAndBFloat16) {
  for (int64_t block : {32, 64, 128}) {
    for (int64_t blocks : {1, 2, 3, 4, 7, 8, 9}) {
      Int3Content<MLFloat16> half;
      Int3Content<BFloat16> bfloat;
      half.SetBlock(block, blocks);
      bfloat.SetBlock(block, blocks);
      half.scales.front() = MLFloat16(1.0f);
      bfloat.scales.front() = BFloat16(1.0f);
      ASSERT_TRUE(half.Validate().IsOK());
      ASSERT_TRUE(bfloat.Validate().IsOK());
      half.zp.clear();
      bfloat.zp.clear();
      ASSERT_TRUE(half.Validate(false).IsOK());
      ASSERT_TRUE(bfloat.Validate(false).IsOK());
      EXPECT_FALSE(half.Validate(true).IsOK());
    }
  }
}

template <typename ScaleT>
void CheckInvalidScales() {
  Int3Content<ScaleT> data;
  data.SetBlock(32, 4);
  for (float invalid : {-1.0f, std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
    data.scales.back() = ScaleT(invalid);
    EXPECT_FALSE(data.Validate().IsOK());
  }
}

TEST(QMoEFormatValidationTest, InvalidScales) {
  CheckInvalidScales<MLFloat16>();
  CheckInvalidScales<BFloat16>();
}

TEST(QMoEFormatValidationTest, ZeroPointContentPaddingAndPresence) {
  Int3Content<MLFloat16> data;
  data.SetBlock(32, 1);
  EXPECT_FALSE(data.Validate(false).IsOK());
  for (uint8_t invalid : {uint8_t{0}, uint8_t{5}, uint8_t{12}, uint8_t{0x84}}) {
    data.zp.back() = invalid;
    EXPECT_FALSE(data.Validate().IsOK());
  }
}

TEST(QMoEFormatValidationTest, ExactBufferSizes) {
  Int3Content<MLFloat16> data;
  for (int delta : {-1, 1}) {
    data.SetBlock(32, 4);
    data.weights.resize(data.weights.size() + delta);
    EXPECT_FALSE(data.Validate().IsOK());
    data.SetBlock(32, 4);
    data.scales.resize(data.scales.size() + delta);
    EXPECT_FALSE(data.Validate().IsOK());
    data.SetBlock(32, 4);
    data.zp.resize(data.zp.size() + delta);
    EXPECT_FALSE(data.Validate().IsOK());
  }
}

TEST(QMoEFormatValidationTest, ShapeAlignmentAndOverflow) {
  Int3Content<MLFloat16> data;
  data.SetBlock(32, 4);
  for (int64_t block : {0, -32, 16, 256}) {
    data.block = block;
    EXPECT_FALSE(data.Validate().IsOK());
  }
  data.block = 32;
  for (size_t axis = 0; axis < 3; ++axis) {
    for (int64_t invalid : std::array<int64_t, 4>{-1, 0, 127, std::numeric_limits<int64_t>::max()}) {
      const auto original = data.shape;
      data.shape[axis] = invalid;
      EXPECT_FALSE(data.Validate().IsOK());
      data.shape = original;
    }
  }
  const std::array<int64_t, 2> rank2{128, 128};
  EXPECT_FALSE(ValidateInt3ProjectionContent<MLFloat16>({}, {}, {}, false, rank2, 32).IsOK());
  data.shape = {std::numeric_limits<int64_t>::max() / 128, 128, 128};
  EXPECT_FALSE(data.Validate().IsOK());
}

TEST(QMoEFormatValidationTest, ByteHelpers) {
  for (int64_t bits : {1, 2, 3, 4, 5, 6, 7, 8}) {
    EXPECT_EQ(PackedByteCount(32, bits), 4 * bits);
    for (int64_t count = 0; count < 33; ++count) {
      EXPECT_EQ(PackedByteCountWithPadding(count, bits), (count * bits + 7) / 8);
    }
  }
#ifndef ORT_NO_EXCEPTIONS
  EXPECT_ANY_THROW(PackedByteCount(1, 3));
  EXPECT_ANY_THROW(PackedByteCount(std::numeric_limits<int64_t>::max(), 8));
  EXPECT_ANY_THROW(PackedByteCountWithPadding(-1, 3));
  EXPECT_ANY_THROW(PackedByteCountWithPadding(1, 0));
  EXPECT_ANY_THROW(PackedByteCountWithPadding(1, -1));
  EXPECT_ANY_THROW(PackedByteCountWithPadding(1, 9));
#endif
  EXPECT_EQ(PackedByteCountWithPadding(std::numeric_limits<int64_t>::max(), 8),
            std::numeric_limits<int64_t>::max());
  EXPECT_EQ(PackedByteCountWithPadding(std::numeric_limits<int64_t>::max(), 3),
            (std::numeric_limits<int64_t>::max() / 8) * 3 + 3);
  EXPECT_EQ(WeightBitsFromPackSize(2).fc1, 4);
  EXPECT_EQ(WeightBitsFromPackSize(1).fc2, 8);
#ifndef ORT_NO_EXCEPTIONS
  EXPECT_ANY_THROW(WeightBitsFromPackSize(0));
  EXPECT_ANY_THROW(WeightBitsFromPackSize(3));
#endif
}
}  // namespace
}  // namespace onnxruntime::test