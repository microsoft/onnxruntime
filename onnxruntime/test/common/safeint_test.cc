// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include "gtest/gtest.h"

namespace onnxruntime::test {

static_assert(is_supported_integer_v<int>);
static_assert(is_supported_integer_v<uint8_t>);
static_assert(!is_supported_integer_v<bool>);

TEST(SafeIntTest, SafeMulMultipliesOperands) {
  EXPECT_EQ(SafeMul<size_t>(size_t{2}, 3U), size_t{6});
  EXPECT_EQ(SafeMul<int>(-2, 3, 4), -24);
}

TEST(SafeIntTest, SafeMulHandlesSameVariableOperands) {
  const int value = 7;
  EXPECT_EQ(SafeMul<int>(value, value), 49);
}

#ifndef ORT_NO_EXCEPTIONS
TEST(SafeIntTest, SafeMulThrowsOnInitialCastOverflow) {
  EXPECT_THROW((void)SafeMul<uint32_t>(-1, 2), OnnxRuntimeException);
}

TEST(SafeIntTest, SafeMulThrowsOnMultiplyOverflow) {
  EXPECT_THROW((void)SafeMul<int>(std::numeric_limits<int>::max(), 2), OnnxRuntimeException);
}
#endif

}  // namespace onnxruntime::test
