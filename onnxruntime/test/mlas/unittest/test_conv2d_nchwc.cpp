// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_conv2d_nchwc.h"
#include "test_conv2d_fixture.h"

static size_t Conv2dNchwcRegistLongExecute() {
  size_t count = 0;

  if (MlasNchwcGetBlockSize() > 1) {
    count += MlasLongExecuteTests<MlasNchwcConv2DTest<false>>::RegisterLongExecute();
    if (GetMlasThreadPool() != nullptr) {
      count += MlasLongExecuteTests<MlasNchwcConv2DTest<true>>::RegisterLongExecute();
    }
#if defined(__aarch64__) && defined(__linux__)
    if (MlasBf16AccelerationSupported()) {
      count += MlasLongExecuteTests<MlasNchwcConv2DBf16Test<false>>::RegisterLongExecute();
      if (GetMlasThreadPool() != nullptr) {
        count += MlasLongExecuteTests<MlasNchwcConv2DBf16Test<true>>::RegisterLongExecute();
      }
    }
#endif
  }

  return count;
}

static size_t Conv2dNchwcRegistShortExecute() {
  size_t count = 0;

  if (MlasNchwcGetBlockSize() > 1) {
    count += Conv2dShortExecuteTest<MlasNchwcConv2DTest<false>>::RegisterShortExecuteTests();
    if (GetMlasThreadPool() != nullptr) {
      count += Conv2dShortExecuteTest<MlasNchwcConv2DTest<true>>::RegisterShortExecuteTests();
    }
#if defined(__aarch64__) && defined(__linux__)
    if (MlasBf16AccelerationSupported()) {
      count += Conv2dShortExecuteTest<MlasNchwcConv2DBf16Test<false>>::RegisterShortExecuteTests();
      if (GetMlasThreadPool() != nullptr) {
        count += Conv2dShortExecuteTest<MlasNchwcConv2DBf16Test<true>>::RegisterShortExecuteTests();
      }
    }
#endif
  }

  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? Conv2dNchwcRegistShortExecute() : Conv2dNchwcRegistLongExecute();
});
