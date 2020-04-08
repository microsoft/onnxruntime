// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test.h"
struct AdapterSessionTestAPi
{
  SetupTest AdapterSessionTestSetup;
  VoidTest AppendExecutionProvider_CPU;
  VoidTest AppendExecutionProvider_DML;
  VoidTest CreateWithoutModel;
  VoidTest GetExecutionProvider;
  VoidTest Initialize;
  VoidTest RegisterGraphTransformers;
  VoidTest RegisterCustomRegistry;
  VoidTest LoadAndPurloinModel;
  VoidTest StartProfiling;
  VoidTest EndProfiling;
  VoidTest CopyInputAcrossDevices;
};
const AdapterSessionTestAPi& getapi();

WINML_TEST_CLASS_BEGIN_WITH_SETUP(AdapterSessionTest, AdapterSessionTestSetup)
WINML_TEST(AdapterSessionTest, AppendExecutionProvider_CPU)
WINML_TEST(AdapterSessionTest, AppendExecutionProvider_DML)
WINML_TEST(AdapterSessionTest, CreateWithoutModel)
WINML_TEST(AdapterSessionTest, GetExecutionProvider)
WINML_TEST(AdapterSessionTest, Initialize)
WINML_TEST(AdapterSessionTest, RegisterGraphTransformers)
WINML_TEST(AdapterSessionTest, RegisterCustomRegistry)
WINML_TEST(AdapterSessionTest, LoadAndPurloinModel)
WINML_TEST(AdapterSessionTest, StartProfiling)
WINML_TEST(AdapterSessionTest, EndProfiling)
WINML_TEST(AdapterSessionTest, CopyInputAcrossDevices)
WINML_TEST_CLASS_END()
