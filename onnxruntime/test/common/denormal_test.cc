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
    const float denormal_float = 1e-38f;
    const double denormal_double = 1e-308;

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

#ifdef _OPENMP
TEST(DenormalTest, OpenMPDenormalAsZeroTest) {
  auto test_denormal = [&](bool set_denormal_as_zero) {
    const float denormal_float = 1e-38f;
    const double denormal_double = 1e-308;
    const int test_size = 4;

    std::array<float, test_size> input_float;
    std::array<double, test_size> input_double;

    // When it returns false, disabling denormal as zero isn't supported,
    // so validation will be skipped
    bool set = SetDenormalAsZero(set_denormal_as_zero);
    if (set || !set_denormal_as_zero) {
      input_float.fill(denormal_float);
      input_double.fill(denormal_double);

      InitializeWithDenormalAsZero(set_denormal_as_zero);
#pragma omp parallel for
      for (auto i = 0; i < test_size; ++i) {
        input_float[i] *= 2;
        input_double[i] *= 2;
      }

      std::for_each(input_float.begin(), input_float.end(), [&](float f) {
        EXPECT_EQ(f, (set_denormal_as_zero) ? 0.0f : denormal_float * 2);
      });
      std::for_each(input_double.begin(), input_double.end(), [&](double d) {
        EXPECT_EQ(d, (set_denormal_as_zero) ? 0.0 : denormal_double * 2);
      });
    }
  };

  test_denormal(true);
  test_denormal(false);
}
#endif

}  // namespace test
}  // namespace onnxruntime
