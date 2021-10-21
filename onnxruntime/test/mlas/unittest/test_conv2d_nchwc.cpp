// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_conv2d_nchwc.h"
#include "test_conv2d_fixture.h"

template <> MlasNchwcConv2DTest<false>* MlasTestFixture<MlasNchwcConv2DTest<false>>::mlas_tester(nullptr);
template <> MlasNchwcConv2DTest<true>* MlasTestFixture<MlasNchwcConv2DTest<true>>::mlas_tester(nullptr);

static size_t Conv2dNchwcRegistLongExecute() {
    size_t count = 0;

    if (MlasNchwcGetBlockSize() > 1) {
        count += MlasLongExecuteTests<MlasNchwcConv2DTest<false>>::RegisterLongExecute();
        if (GetMlasThreadPool() != nullptr) {
            count += MlasLongExecuteTests<MlasNchwcConv2DTest<true>>::RegisterLongExecute();
        }
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
    }

    return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? Conv2dNchwcRegistShortExecute() : Conv2dNchwcRegistLongExecute();
});
