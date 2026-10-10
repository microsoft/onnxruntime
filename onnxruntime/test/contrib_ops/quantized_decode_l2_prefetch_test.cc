// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include <utility>

#include "gtest/gtest.h"
#include "contrib_ops/cuda/quantization/quantized_decode_l2_prefetch.h"

namespace onnxruntime::test {

using contrib::cuda::QuantizedDecodeL2PrefetchOffset;
using contrib::cuda::ShouldPrefetchQuantizedDecodeL2;

TEST(QuantizedDecodeL2PrefetchTest, RequiresOptInAndSm121) {
  EXPECT_FALSE(ShouldPrefetchQuantizedDecodeL2(false, 12, 1));
  EXPECT_TRUE(ShouldPrefetchQuantizedDecodeL2(true, 12, 1));
  for (const auto& [major, minor] : {std::pair{8, 0}, std::pair{9, 0}, std::pair{10, 0},
                                     std::pair{12, 0}, std::pair{12, 2}, std::pair{13, 0}}) {
    EXPECT_FALSE(ShouldPrefetchQuantizedDecodeL2(true, major, minor));
  }
}

TEST(QuantizedDecodeL2PrefetchTest, TwoIterationLookAheadAndTail) {
  EXPECT_EQ(QuantizedDecodeL2PrefetchOffset(0, 128, 384), 256);
  EXPECT_EQ(QuantizedDecodeL2PrefetchOffset(127, 128, 384), 383);
  EXPECT_EQ(QuantizedDecodeL2PrefetchOffset(128, 128, 384), -1);
  EXPECT_EQ(QuantizedDecodeL2PrefetchOffset(0, 128, 256), -1);
  EXPECT_EQ(QuantizedDecodeL2PrefetchOffset(0, 128, 257), 256);
  EXPECT_EQ(QuantizedDecodeL2PrefetchOffset(256, 128, 256), -1);
  EXPECT_EQ(QuantizedDecodeL2PrefetchOffset(-1, 128, 384), -1);
  EXPECT_EQ(QuantizedDecodeL2PrefetchOffset(0, 0, 384), -1);
  EXPECT_EQ(QuantizedDecodeL2PrefetchOffset(0, -1, 384), -1);
  EXPECT_EQ(QuantizedDecodeL2PrefetchOffset(0, 128, 0), -1);
}

TEST(QuantizedDecodeL2PrefetchTest, SplitKAndLargeOffsets) {
  EXPECT_EQ(QuantizedDecodeL2PrefetchOffset(1, 4, 10), 9);
  EXPECT_EQ(QuantizedDecodeL2PrefetchOffset(2, 4, 10), -1);
  constexpr int64_t max = std::numeric_limits<int64_t>::max();
  EXPECT_EQ(QuantizedDecodeL2PrefetchOffset(max - 3, 1, max), max - 1);
  EXPECT_EQ(QuantizedDecodeL2PrefetchOffset(max - 2, 1, max), -1);
  EXPECT_EQ(QuantizedDecodeL2PrefetchOffset(0, max, max), -1);
  EXPECT_EQ(QuantizedDecodeL2PrefetchOffset(0, max / 2, max), max - 1);
  EXPECT_EQ(QuantizedDecodeL2PrefetchOffset(0, max / 2 + 1, max), -1);
}

TEST(QuantizedDecodeL2PrefetchTest, PackedAndRaggedWeightRows) {
  for (int elements_per_byte : {1, 2, 4}) {
    for (int k : {16, 32, 64, 128, 256, 512, 544, 1024, 2048, 4096}) {
      const int extent = k / elements_per_byte;
      const int step = 256 / elements_per_byte;
      for (int lane = 0; lane < 32; ++lane) {
        for (int offset = lane * (8 / elements_per_byte); offset < extent; offset += step) {
          const int64_t next = QuantizedDecodeL2PrefetchOffset(offset, step, extent);
          if (static_cast<int64_t>(offset) + 2 * step < extent) {
            EXPECT_EQ(next, offset + 2 * step);
            EXPECT_GE(next, 0);
            EXPECT_LT(next, extent);
          } else {
            EXPECT_EQ(next, -1);
          }
        }
      }
    }
  }
}

}  // namespace onnxruntime::test
