// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <limits>

#include "gtest/gtest.h"

#include "core/common/exceptions.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace test {

using webgpu::CeilDiv;

TEST(WebGpuUtilsTest, CeilDivHandlesIntegerLimits) {
  constexpr uint32_t max_u32 = std::numeric_limits<uint32_t>::max();
  constexpr uint64_t max_u64 = std::numeric_limits<uint64_t>::max();
  constexpr int64_t max_i64 = std::numeric_limits<int64_t>::max();
  EXPECT_EQ(CeilDiv(uint32_t{0}, uint32_t{2}), 0U);
  EXPECT_EQ(CeilDiv(uint32_t{8}, uint32_t{2}), 4U);
  EXPECT_EQ(CeilDiv(uint32_t{9}, uint32_t{2}), 5U);
  EXPECT_EQ(CeilDiv(max_u32, uint32_t{1}), max_u32);
  EXPECT_EQ(CeilDiv(max_u32, uint32_t{2}), max_u32 / 2 + 1);
  EXPECT_EQ(CeilDiv(max_u64, uint64_t{2}), max_u64 / 2 + 1);
  EXPECT_EQ(CeilDiv(max_i64, int64_t{1}), max_i64);
  EXPECT_EQ(CeilDiv(max_i64, int64_t{2}), max_i64 / 2 + 1);
}

TEST(WebGpuUtilsTest, CeilDivRejectsInvalidArguments) {
  EXPECT_THROW(CeilDiv(uint32_t{1}, uint32_t{0}), OnnxRuntimeException);
  EXPECT_THROW(CeilDiv(int64_t{1}, int64_t{-1}), OnnxRuntimeException);
  EXPECT_THROW(CeilDiv(int64_t{-1}, int64_t{2}), OnnxRuntimeException);
}

}  // namespace test
}  // namespace onnxruntime
