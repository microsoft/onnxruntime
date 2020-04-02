// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test.h"
struct AdapterModelTestApi
{
  SetupTest AdapterModelTestSetup;
  VoidTest CreateModelFromPath;
  VoidTest CreateModelFromData;
  VoidTest CloneModel;
  VoidTest ModelGetAuthor;
  VoidTest ModelGetName;
  VoidTest ModelGetDomain;
  VoidTest ModelGetDescription;
  VoidTest ModelGetVersion;
  VoidTest ModelGetInputCount;
  VoidTest ModelGetOutputCount;
  VoidTest ModelGetInputName;
  VoidTest ModelGetOutputName;
  VoidTest ModelGetInputDescription;
  VoidTest ModelGetOutputDescription;
  VoidTest ModelGetInputTypeInfo;
  VoidTest ModelGetOutputTypeInfo;
  VoidTest ModelGetMetadataCount;
  VoidTest ModelGetMetadata;
  VoidTest ModelEnsureNoFloat16;
};
const AdapterModelTestApi& getapi();
const WinmlAdapterApi *winmlAdapter;
OrtModel* ortModel = nullptr;

WINML_TEST_CLASS_BEGIN_WITH_SETUP(AdapterModelTest, AdapterModelTestSetup)
WINML_TEST(AdapterModelTest, CreateModelFromPath)
WINML_TEST(AdapterModelTest, CreateModelFromData)
WINML_TEST(AdapterModelTest, CloneModel)
WINML_TEST(AdapterModelTest, ModelGetAuthor)
WINML_TEST(AdapterModelTest, ModelGetName)
WINML_TEST(AdapterModelTest, ModelGetDomain)
WINML_TEST(AdapterModelTest, ModelGetDescription)
WINML_TEST(AdapterModelTest, ModelGetVersion)
WINML_TEST(AdapterModelTest, ModelGetInputCount)
WINML_TEST(AdapterModelTest, ModelGetOutputCount)
WINML_TEST(AdapterModelTest, ModelGetInputName)
WINML_TEST(AdapterModelTest, ModelGetOutputName)
WINML_TEST(AdapterModelTest, ModelGetInputDescription)
WINML_TEST(AdapterModelTest, ModelGetOutputDescription)
WINML_TEST(AdapterModelTest, ModelGetInputTypeInfo)
WINML_TEST(AdapterModelTest, ModelGetOutputTypeInfo)
WINML_TEST(AdapterModelTest, ModelGetMetadataCount)
WINML_TEST(AdapterModelTest, ModelGetMetadata)
WINML_TEST(AdapterModelTest, ModelEnsureNoFloat16)
WINML_TEST_CLASS_END()
