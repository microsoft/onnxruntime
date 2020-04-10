// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test.h"
struct AdapterSessionTestAPi
{
  SetupTest AdapterSessionTestSetup;
  VoidTest AppendExecutionProvider_CPU;
  VoidTest AppendExecutionProvider_DML;
  VoidTest CreateWithoutModel;
  VoidTest GetExecutionProvider_CPU;
  VoidTest GetExecutionProvider_DML;
  VoidTest Initialize;
  VoidTest RegisterGraphTransformers;
  VoidTest RegisterGraphTransformers_DML;
  VoidTest RegisterCustomRegistry;
  VoidTest RegisterCustomRegistry_DML;
  VoidTest LoadAndPurloinModel;
  VoidTest Profiling;
  VoidTest CopyInputAcrossDevices;
};
const AdapterSessionTestAPi& getapi();

WINML_TEST_CLASS_BEGIN_WITH_SETUP(AdapterSessionTest, AdapterSessionTestSetup)
WINML_TEST(AdapterSessionTest, AppendExecutionProvider_CPU)
WINML_TEST(AdapterSessionTest, AppendExecutionProvider_DML)
WINML_TEST(AdapterSessionTest, CreateWithoutModel)
WINML_TEST(AdapterSessionTest, GetExecutionProvider_CPU)
WINML_TEST(AdapterSessionTest, GetExecutionProvider_DML)
WINML_TEST(AdapterSessionTest, Initialize)
WINML_TEST(AdapterSessionTest, RegisterGraphTransformers)
WINML_TEST(AdapterSessionTest, RegisterGraphTransformers_DML)
WINML_TEST(AdapterSessionTest, RegisterCustomRegistry)
WINML_TEST(AdapterSessionTest, RegisterCustomRegistry_DML)
WINML_TEST(AdapterSessionTest, LoadAndPurloinModel)
WINML_TEST(AdapterSessionTest, Profiling)
WINML_TEST(AdapterSessionTest, CopyInputAcrossDevices)
WINML_TEST_CLASS_END()
