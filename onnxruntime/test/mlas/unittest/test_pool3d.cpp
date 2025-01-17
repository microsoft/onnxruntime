// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_pool3d.h"
#include "test_pool3d_fixture.h"

static size_t Pool3dRegistLongExecute() {
  size_t count = 0;
  count += MlasLongExecuteTests<MlasPool3DTest<MlasMaximumPooling, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasPool3DTest<MlasAveragePoolingExcludePad, false>>::RegisterLongExecute();
  count += MlasLongExecuteTests<MlasPool3DTest<MlasAveragePoolingIncludePad, false>>::RegisterLongExecute();
  if (GetMlasThreadPool() != nullptr) {
    count += MlasLongExecuteTests<MlasPool3DTest<MlasMaximumPooling, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasPool3DTest<MlasAveragePoolingExcludePad, true>>::RegisterLongExecute();
    count += MlasLongExecuteTests<MlasPool3DTest<MlasAveragePoolingIncludePad, true>>::RegisterLongExecute();
  }
  return count;
}

static size_t Pool3dRegistShortExecute() {
  size_t count = 0;
  count += Pooling3dShortExecuteTest<MlasMaximumPooling, false>::RegisterShortExecuteTests();
  count += Pooling3dShortExecuteTest<MlasAveragePoolingExcludePad, false>::RegisterShortExecuteTests();
  count += Pooling3dShortExecuteTest<MlasAveragePoolingIncludePad, false>::RegisterShortExecuteTests();
  if (GetMlasThreadPool() != nullptr) {
    count += Pooling3dShortExecuteTest<MlasMaximumPooling, true>::RegisterShortExecuteTests();
    count += Pooling3dShortExecuteTest<MlasAveragePoolingExcludePad, true>::RegisterShortExecuteTests();
    count += Pooling3dShortExecuteTest<MlasAveragePoolingIncludePad, true>::RegisterShortExecuteTests();
  }
  return count;
}

static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
  return is_short_execute ? Pool3dRegistShortExecute() : Pool3dRegistLongExecute();
});
