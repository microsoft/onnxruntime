// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_pool2d.h"
#include "test_pool2d_fixture.h"

static size_t Pool2dRegistLongExecute() {
  size_t count = 0;
  count += MlasLongExecuteTests<MlasPool2DTest<MlasMaximumPooling, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasPool2DTest<MlasAveragePoolingExcludePad, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasPool2DTest<MlasAveragePoolingIncludePad, false>>::RegisterLongExecute();
  if (GetMlasThreadPool() != nullptr) {
    count += MlasLongExecuteTests<MlasPool2DTest<MlasMaximumPooling, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasPool2DTest<MlasAveragePoolingExcludePad, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasPool2DTest<MlasAveragePoolingIncludePad, true>>::RegisterLongExecute();
  }
  return count;
}

static size_t Pool2dRegistShortExecute() {
  size_t count = 0;
  count += Pooling2dShortExecuteTest<MlasPool2DTest<MlasMaximumPooling, false>>::RegisterShortExecuteTests();
  count += Pooling2dShortExecuteTest<MlasPool2DTest<MlasAveragePoolingExcludePad, false>>::RegisterShortExecuteTests();
  count += Pooling2dShortExecuteTest<MlasPool2DTest<MlasAveragePoolingIncludePad, false>>::RegisterShortExecuteTests();
  if (GetMlasThreadPool() != nullptr) {
    count += Pooling2dShortExecuteTest<MlasPool2DTest<MlasMaximumPooling, true>>::RegisterShortExecuteTests();
    count += Pooling2dShortExecuteTest<MlasPool2DTest<MlasAveragePoolingExcludePad, true>>::RegisterShortExecuteTests();
    count += Pooling2dShortExecuteTest<MlasPool2DTest<MlasAveragePoolingIncludePad, true>>::RegisterShortExecuteTests();
  }
  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? Pool2dRegistShortExecute() : Pool2dRegistLongExecute();
});
