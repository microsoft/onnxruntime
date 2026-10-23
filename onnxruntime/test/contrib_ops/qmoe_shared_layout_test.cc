// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "contrib_ops/cpu/moe/qmoe_shared_layout.h"

namespace onnxruntime::test {
namespace {

namespace shared = contrib::qmoe_shared;

TEST(QMoESharedLayoutTest, StorageRangeChecksOffsetEvenWithoutExternalLength) {
  constexpr auto max_offset = std::numeric_limits<uint64_t>::max();
  size_t bytes = 123;
  EXPECT_FALSE(shared::ValidateExpertStorageRange(2, 256, max_offset - 511, 0, &bytes));
  EXPECT_EQ(bytes, 123U);
  EXPECT_FALSE(shared::ValidateExpertStorageRange(2, 256, max_offset - 511, 512));
  EXPECT_TRUE(shared::ValidateExpertStorageRange(2, 256, max_offset - 512, 0, &bytes));
  EXPECT_EQ(bytes, 512U);
  EXPECT_TRUE(shared::ValidateExpertStorageRange(2, 256, max_offset - 512, 512));
}

TEST(QMoESharedLayoutTest, HalfConversionPreservesSignedZero) {
  const std::array<uint16_t, 2> bits{0x0000, 0x8000};
  for (bool scalar : {true, false}) {
    std::array<float, 2> output{};
    shared::HalfBitsToFloatRow(bits.data(), output.data(), bits.size(), scalar);
    EXPECT_EQ(output[0], 0.0f);
    EXPECT_FALSE(std::signbit(output[0]));
    EXPECT_EQ(output[1], 0.0f);
    EXPECT_TRUE(std::signbit(output[1]));
  }
}

void CheckPartialInt4Ranges(bool scalar) {
  constexpr int64_t hidden = 128;
  constexpr int64_t inter = 64;
  constexpr float sentinel = 12345.0f;
  shared::ExpertBlockLayout layout;
  ASSERT_TRUE(shared::MakeExpertBlockLayout(hidden, inter, &layout));
  std::vector<uint8_t> block(layout.stride);
  auto* weights = block.data() + layout.fc2_weights;
  auto* scales = reinterpret_cast<uint16_t*>(block.data() + layout.fc2_scales);
  auto* bias = reinterpret_cast<uint16_t*>(block.data() + layout.fc2_bias);
  for (int64_t i = 0; i < hidden * inter / 2; ++i) weights[i] = static_cast<uint8_t>(i * 67 + i / 13);
  for (int64_t i = 0; i < hidden * inter / 32; ++i) scales[i] = shared::FloatToHalf((i % 7 + 1) / 64.0f);
  for (int64_t i = 0; i < hidden; ++i) bias[i] = shared::FloatToHalf((i % 11 - 5) / 32.0f);

  const std::array<std::pair<int64_t, int64_t>, 8> ranges{
      {{0, 0}, {0, 1}, {1, 4}, {1, 5}, {4, 7}, {124, 127}, {126, 128}, {0, 128}}};
  for (int64_t rows = 1; rows <= shared::kMaxRowsPerCall; ++rows) {
    std::vector<float> input(static_cast<size_t>(rows * inter));
    for (size_t i = 0; i < input.size(); ++i) input[i] = (static_cast<int>(i % 17) - 8) / 64.0f;
    std::vector<float> expected(static_cast<size_t>(rows * hidden));
    shared::Fc2Range(layout, block.data(), input.data(), rows, expected.data(), 0, hidden, true);
    for (const auto& [begin, end] : ranges) {
      SCOPED_TRACE(testing::Message() << "rows=" << rows << " range=[" << begin << "," << end << ")");
      std::vector<float> output(static_cast<size_t>(rows * hidden + 2), sentinel);
      shared::Fc2Range(layout, block.data(), input.data(), rows, output.data() + 1, begin, end, scalar);
      EXPECT_EQ(output.front(), sentinel);
      EXPECT_EQ(output.back(), sentinel);
      for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < hidden; ++col) {
          const size_t index = static_cast<size_t>(row * hidden + col);
          if (col >= begin && col < end) {
            EXPECT_NEAR(output[index + 1], expected[index], 1e-5f);
          } else {
            EXPECT_EQ(output[index + 1], sentinel);
          }
        }
      }
    }
  }
}

TEST(QMoESharedLayoutTest, ScalarPartialInt4RangesPreserveOtherColumns) {
  CheckPartialInt4Ranges(true);
}

TEST(QMoESharedLayoutTest, Avx512PartialInt4RangesPreserveOtherColumns) {
  if (!shared::CpuHasAvx512()) GTEST_SKIP() << "AVX-512 kernels are not available";
  CheckPartialInt4Ranges(false);
}

TEST(QMoESharedLayoutTest, Avx512BFloat16ConversionMatchesScalarIncludingNaNs) {
  if (!shared::CpuHasAvx512()) GTEST_SKIP() << "AVX-512 kernels are not available";
  const std::array<uint32_t, 18> patterns{
      0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x3F808000, 0x3F818000,
      0x00000001, 0x00008000, 0x00018000, 0x7F7FFFFF, 0x7F800000, 0xFF800000,
      0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFFFFFFF, 0x7FC00000, 0xFFC00000};
  std::array<float, 36> input{};
  for (size_t i = 0; i < input.size(); ++i) {
    const auto bits = patterns[i % patterns.size()];
    std::memcpy(&input[i], &bits, sizeof(bits));
  }
  std::array<uint16_t, 36> scalar{}, simd{};
  shared::FloatToBFloat16BitsRow(input.data(), scalar.data(), input.size(), true);
  shared::FloatToBFloat16BitsRow(input.data(), simd.data(), input.size(), false);
  EXPECT_EQ(simd, scalar);
  EXPECT_EQ(simd[12], 0x7FC1);
  EXPECT_EQ(simd[13], 0x7FC1);
}

TEST(QMoESharedLayoutTest, Avx512Fp4ScalesMatchScalarIncludingNonfiniteValues) {
  if (!shared::CpuHasAvx512()) GTEST_SKIP() << "AVX-512 kernels are not available";
  constexpr int64_t k = 32;
  constexpr int64_t columns = 9;  // Exercise vector groups and the scalar tail.
  std::array<uint8_t, columns * k / 2> weights;
  weights.fill(0x22);  // E2M1 1.0 in both nibbles.
  std::array<uint8_t, columns * k / 16> scales{};
  std::array<float, shared::kMaxRowsPerCall * k> input;
  input.fill(1.0f);
  std::array<float, shared::kMaxRowsPerCall * columns> scalar{}, simd{};
  for (bool mxfp4 : {false, true}) {
    for (bool bf16 : {false, true}) {
      for (int code = 0; code < 256; ++code) {
        scales.fill(static_cast<uint8_t>(code));
        for (int64_t rows = 1; rows <= shared::kMaxRowsPerCall; ++rows) {
          SCOPED_TRACE(testing::Message() << "mxfp4=" << mxfp4 << " bf16=" << bf16
                                          << " scale=" << code << " rows=" << rows);
          const auto gemm = mxfp4 ? shared::Mxfp4GemmRange : shared::Nvfp4GemmRange;
          gemm(weights.data(), scales.data(), 1.0f, nullptr, bf16, columns, k, input.data(), rows,
               scalar.data(), columns, 0, columns, true, bf16);
          gemm(weights.data(), scales.data(), 1.0f, nullptr, bf16, columns, k, input.data(), rows,
               simd.data(), columns, 0, columns, false, bf16);
          for (int64_t i = 0; i < rows * columns; ++i) {
            if (std::isnan(scalar[i])) {
              EXPECT_TRUE(std::isnan(simd[i])) << i;
            } else {
              EXPECT_EQ(simd[i], scalar[i]) << i;
            }
          }
        }
      }
    }
  }
}

#if defined(__linux__)
TEST(QMoESharedLayoutTest, MappedStoreHonorsPinsAndRollsBackFailedGroups) {
  constexpr size_t stride = shared::kBankAlignment;
  alignas(shared::kBankAlignment) std::array<uint8_t, 3 * stride> backing{};
  for (size_t i = 0; i < backing.size(); ++i) backing[i] = static_cast<uint8_t>(i / stride + 1);
  shared::FileExpertStore::MappedBank bank{backing.data(), stride, 0};
  shared::FileExpertStore store;
  ASSERT_TRUE(store.InitializeMapped(&bank, 1, 3, stride, 2));
  shared::ExpertBlockLease lease{};
  ASSERT_TRUE(store.AcquireReady(0, &lease));
  EXPECT_EQ(lease.data[0], 1);
  EXPECT_FALSE(store.Evict(0));
  const std::array<int32_t, 2> group{1, 3};
  std::array<shared::ExpertBlockLease, 2> group_leases{};
  EXPECT_FALSE(store.AcquireGroupReady(group.data(), group.size(), group_leases.data()));
  EXPECT_TRUE(store.Evict(1));
  store.ReleaseLease(0);
  const int32_t prefetch = 2;
  ASSERT_TRUE(store.Prefetch(&prefetch, 1, 1));
  store.WaitPrefetchIdle();
  ASSERT_TRUE(store.TryAcquireCached(2, &lease));
  EXPECT_EQ(lease.data[0], 3);
  store.ReleaseLease(2);
  auto* replacement = store.BeginReplace(2, 1);
  ASSERT_NE(replacement, nullptr);
  EXPECT_FALSE(store.TryAcquireCached(1, &lease));
  std::fill_n(replacement, stride, uint8_t{42});
  store.FinishReplace(1, true);
  ASSERT_TRUE(store.TryAcquireCached(1, &lease));
  EXPECT_EQ(lease.data[0], 42);
  store.ReleaseLease(1);
  ASSERT_NE(store.BeginReplace(1, 2), nullptr);
  store.FinishReplace(2, false);
  EXPECT_FALSE(store.TryAcquireCached(2, &lease));
  ASSERT_TRUE(store.AcquireReady(2, &lease));
  EXPECT_EQ(lease.data[0], 3);
  store.ReleaseLease(2);
}
#endif

}  // namespace
}  // namespace onnxruntime::test
