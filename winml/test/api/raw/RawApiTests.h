// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test.h"

struct RawApiTestsApi
{
  VoidTest CreateModelFromFilePath;
};

struct RawApiTestsGpuApi
{
  SetupTest MethodSetup;
};

template <typename TApi> const TApi& getapi();
template <> const RawApiTestsApi& getapi<RawApiTestsApi>();
template <> const RawApiTestsGpuApi& getapi<RawApiTestsGpuApi>();

WINML_TEST_CLASS_BEGIN(RawApiTests)
WINML_TEST_CLASS_BEGIN_TESTS
WINML_TEST_EX(RawApiTests, CreateModelFromFilePath, RawApiTestsApi)
WINML_TEST_CLASS_END()

WINML_TEST_CLASS_BEGIN(RawApiTestsGpu)
WINML_TEST_CLASS_SETUP_METHOD_EX(MethodSetup, RawApiTestsGpuApi)
WINML_TEST_CLASS_BEGIN_TESTS
WINML_TEST_CLASS_END()
