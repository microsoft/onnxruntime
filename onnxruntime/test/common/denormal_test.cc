// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>

#include "core/common/denormal.h"

#include "gtest/gtest.h"

#include "test/util/include/asserts.h"

#include <array>

namespace onnxruntime {
namespace test {

TEST(DenormalTest, DenormalAsZeroTest) {
  auto test_denormal = [&](bool set_denormal_as_zero) {
    constexpr float denormal_float = 1e-38f;
    constexpr double denormal_double = 1e-308;

    volatile float input_float = denormal_float;
    volatile double input_double = denormal_double;

    // When it returns false, disabling denormal as zero isn't supported,
    // so validation will be skipped
    bool set = SetDenormalAsZero(set_denormal_as_zero);
    if (set || !set_denormal_as_zero) {
      EXPECT_EQ(input_float * 2, ((set_denormal_as_zero) ? 0.0f : denormal_float * 2));
      EXPECT_EQ(input_double * 2, ((set_denormal_as_zero) ? 0.0 : denormal_double * 2));
    }
  };

  test_denormal(true);
  test_denormal(false);
}

}  // namespace test
}  // namespace onnxruntime
