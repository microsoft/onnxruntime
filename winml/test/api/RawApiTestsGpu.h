// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test.h"

struct RawApiTestsGpuApi {
  SetupClass RawApiTestsGpuApiTestsClassSetup;
  VoidTest CreateDirectXDevice;
  VoidTest CreateD3D11DeviceDevice;
  VoidTest CreateD3D12CommandQueueDevice;
  VoidTest CreateDirectXHighPerformanceDevice;
  VoidTest CreateDirectXMinPowerDevice;
  VoidTest Evaluate;
  VoidTest EvaluateNoInputCopy;
  VoidTest EvaluateManyBuffers;
};

const RawApiTestsGpuApi& getapi();

WINML_TEST_CLASS_BEGIN(RawApiTestsGpu)
WINML_TEST_CLASS_SETUP_CLASS(RawApiTestsGpuApiTestsClassSetup)
WINML_TEST_CLASS_BEGIN_TESTS
WINML_TEST(RawApiTestsGpu, CreateDirectXDevice)
WINML_TEST(RawApiTestsGpu, CreateD3D11DeviceDevice)
WINML_TEST(RawApiTestsGpu, CreateD3D12CommandQueueDevice)
WINML_TEST(RawApiTestsGpu, CreateDirectXHighPerformanceDevice)
WINML_TEST(RawApiTestsGpu, CreateDirectXMinPowerDevice)
WINML_TEST(RawApiTestsGpu, Evaluate)
WINML_TEST(RawApiTestsGpu, EvaluateNoInputCopy)
WINML_TEST(RawApiTestsGpu, EvaluateManyBuffers)
WINML_TEST_CLASS_END()
