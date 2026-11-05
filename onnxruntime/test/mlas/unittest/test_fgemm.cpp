// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_fgemm.h"
#include "test_fgemm_fixture.h"

#include <memory>
#include <limits>
#include <sstream>

TEST(FGemmPackB, SizeBounds) {
  const size_t max = (std::numeric_limits<size_t>::max)();
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG config{};
  for (bool use_kleidiai : {false, true}) {
    config.use_kleidiai = use_kleidiai;
    EXPECT_EQ(MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, max, 1, &config), size_t{0});
    EXPECT_EQ(MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, 16, max, &config), size_t{0});
    EXPECT_EQ(MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, 16, max / 16, &config), size_t{0});
  }
  config.use_kleidiai = false;
  const size_t alignment = MlasGetPreferredBufferAlignment();
  const size_t expected = (16 * 3 * sizeof(float) + alignment - 1) & ~(alignment - 1);
  EXPECT_EQ(MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, 5, 3, &config), expected);
  const size_t largest_k = (max - (alignment - 1)) / (16 * sizeof(float));
  const size_t largest_size = (16 * largest_k * sizeof(float) + alignment - 1) & ~(alignment - 1);
  EXPECT_EQ(MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, 16, largest_k, &config), largest_size);
  EXPECT_EQ(MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, 16, largest_k + 1, &config), size_t{0});
  EXPECT_EQ(MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, 0, 3, &config), size_t{0});
  EXPECT_EQ(MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, 5, 0, &config), size_t{0});
}

static size_t FGemmRegistLongExecute() {
  size_t count = 0;

  count += MlasLongExecuteTests<MlasFgemmTest<float, false, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasFgemmTest<float, true, false>>::RegisterLongExecute();

  if (GetMlasThreadPool() != nullptr) {
    count += MlasLongExecuteTests<MlasFgemmTest<float, false, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasFgemmTest<float, true, true>>::RegisterLongExecute();
  }

#ifdef MLAS_SUPPORTS_GEMM_DOUBLE

  count += MlasLongExecuteTests<MlasFgemmTest<double, false, false>>::RegisterLongExecute();
  if (GetMlasThreadPool() != nullptr) {
    count += MlasLongExecuteTests<MlasFgemmTest<double, false, true>>::RegisterLongExecute();
  }

#endif

  return count;
}

static size_t FGemmRegistShortExecute() {
  size_t count = 0;

  count += FgemmShortExecuteTest<float, false, false>::RegisterShortExecuteTests();
  count += FgemmShortExecuteTest<float, true, false>::RegisterShortExecuteTests();

  if (GetMlasThreadPool() != nullptr) {
    count += FgemmShortExecuteTest<float, false, true>::RegisterShortExecuteTests();
    count += FgemmShortExecuteTest<float, true, true>::RegisterShortExecuteTests();
  }

#ifdef MLAS_SUPPORTS_GEMM_DOUBLE

  count += FgemmShortExecuteTest<double, false, false>::RegisterShortExecuteTests();
  if (GetMlasThreadPool() != nullptr) {
    count += FgemmShortExecuteTest<double, false, true>::RegisterShortExecuteTests();
  }

#endif

  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? FGemmRegistShortExecute() : FGemmRegistLongExecute();
});
