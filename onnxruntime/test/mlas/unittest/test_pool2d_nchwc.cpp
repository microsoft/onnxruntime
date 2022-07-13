// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_pool2d_nchwc.h"
#include "test_pool2d_fixture.h"

template <> MlasNchwcPool2DTest<MlasMaximumPooling, false>* MlasTestFixture<MlasNchwcPool2DTest<MlasMaximumPooling, false>>::mlas_tester(nullptr);
template <> MlasNchwcPool2DTest<MlasAveragePoolingExcludePad, false>* MlasTestFixture<MlasNchwcPool2DTest<MlasAveragePoolingExcludePad, false>>::mlas_tester(nullptr);
template <> MlasNchwcPool2DTest<MlasAveragePoolingIncludePad, false>* MlasTestFixture<MlasNchwcPool2DTest<MlasAveragePoolingIncludePad, false>>::mlas_tester(nullptr);

template <> MlasNchwcPool2DTest<MlasMaximumPooling, true>* MlasTestFixture<MlasNchwcPool2DTest<MlasMaximumPooling, true>>::mlas_tester(nullptr);
template <> MlasNchwcPool2DTest<MlasAveragePoolingExcludePad, true>* MlasTestFixture<MlasNchwcPool2DTest<MlasAveragePoolingExcludePad, true>>::mlas_tester(nullptr);
template <> MlasNchwcPool2DTest<MlasAveragePoolingIncludePad, true>* MlasTestFixture<MlasNchwcPool2DTest<MlasAveragePoolingIncludePad, true>>::mlas_tester(nullptr);

static size_t Pool2dNchwcRegistLongExecute() {
  size_t count = 0;
  if (MlasNchwcGetBlockSize() > 1) {
    count += MlasLongExecuteTests<MlasNchwcPool2DTest<MlasMaximumPooling, false>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasNchwcPool2DTest<MlasAveragePoolingExcludePad, false>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasNchwcPool2DTest<MlasAveragePoolingIncludePad, false>>::RegisterLongExecute();
    if (GetMlasThreadPool() != nullptr) {
      count += MlasLongExecuteTests<MlasNchwcPool2DTest<MlasMaximumPooling, true>>::RegisterLongExecute();
      count += MlasLongExecuteTests<MlasNchwcPool2DTest<MlasAveragePoolingExcludePad, true>>::RegisterLongExecute();
      count += MlasLongExecuteTests<MlasNchwcPool2DTest<MlasAveragePoolingIncludePad, true>>::RegisterLongExecute();
    }
  }
  return count;
}

static size_t Pool2dNchwcRegistShortExecute() {
  size_t count = 0;
  if (MlasNchwcGetBlockSize() > 1) {
    count += Pooling2dShortExecuteTest<MlasNchwcPool2DTest<MlasMaximumPooling, false>>::RegisterShortExecuteTests();
    count += Pooling2dShortExecuteTest<MlasNchwcPool2DTest<MlasAveragePoolingExcludePad, false>>::RegisterShortExecuteTests();
    count += Pooling2dShortExecuteTest<MlasNchwcPool2DTest<MlasAveragePoolingIncludePad, false>>::RegisterShortExecuteTests();
    if (GetMlasThreadPool() != nullptr) {
      count += Pooling2dShortExecuteTest<MlasNchwcPool2DTest<MlasMaximumPooling, true>>::RegisterShortExecuteTests();
      count += Pooling2dShortExecuteTest<MlasNchwcPool2DTest<MlasAveragePoolingExcludePad, true>>::RegisterShortExecuteTests();
      count += Pooling2dShortExecuteTest<MlasNchwcPool2DTest<MlasAveragePoolingIncludePad, true>>::RegisterShortExecuteTests();
    }
  }
  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? Pool2dNchwcRegistShortExecute() : Pool2dNchwcRegistLongExecute();
});
