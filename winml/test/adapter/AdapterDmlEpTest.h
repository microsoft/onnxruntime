// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test.h"
struct AdapterDmlEpTestApi {
  SetupTest AdapterDmlEpTestSetup;
  TeardownClass AdapterDmlEpTestTeardown;
  VoidTest DmlExecutionProviderFlushContext;
  VoidTest DmlExecutionProviderReleaseCompletedReferences;
  VoidTest DmlCreateGPUAllocationFromD3DResource;
  VoidTest DmlCreateAndFreeGPUAllocationFromD3DResource;
  VoidTest GetTensorMemoryInfo;
  VoidTest ExecutionProviderSync;
  VoidTest DmlCopyTensor;
  VoidTest CreateCustomRegistry;
  VoidTest ValueGetDeviceId;
  VoidTest SessionGetInputRequiredDeviceId;
};
const AdapterDmlEpTestApi& getapi();

WINML_TEST_CLASS_BEGIN(AdapterDmlEpTest)
WINML_TEST_CLASS_SETUP_METHOD(AdapterDmlEpTestSetup)
WINML_TEST_CLASS_TEARDOWN_METHOD(AdapterDmlEpTestTeardown)
WINML_TEST_CLASS_BEGIN_TESTS
WINML_TEST(AdapterDmlEpTest, DmlExecutionProviderFlushContext)
WINML_TEST(AdapterDmlEpTest, DmlExecutionProviderReleaseCompletedReferences)
WINML_TEST(AdapterDmlEpTest, DmlCreateGPUAllocationFromD3DResource)
WINML_TEST(AdapterDmlEpTest, DmlCreateAndFreeGPUAllocationFromD3DResource)
WINML_TEST(AdapterDmlEpTest, GetTensorMemoryInfo)
WINML_TEST(AdapterDmlEpTest, ExecutionProviderSync)
WINML_TEST(AdapterDmlEpTest, DmlCopyTensor)
WINML_TEST(AdapterDmlEpTest, CreateCustomRegistry)
WINML_TEST(AdapterDmlEpTest, ValueGetDeviceId)
WINML_TEST(AdapterDmlEpTest, SessionGetInputRequiredDeviceId)
WINML_TEST_CLASS_END()
