// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test.h"
#include "onnxruntime_c_api.h"
#include "../../winml/adapter/winml_adapter_c_api.h"

struct AdapterTestApi
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
  VoidTest EnvConfigureCustomLoggerAndProfiler;
};
const AdapterTestApi& getapi();
const WinmlAdapterApi* winmlAdapter;
const OrtApi* ortApi;
OrtModel* squeezenetModel = nullptr;
OrtModel* metadataModel = nullptr;
OrtModel* float16Model = nullptr;
static bool loggingFunctionCalled = false;
static bool profilingFunctionCalled = false;

WINML_TEST_CLASS_BEGIN_WITH_SETUP(AdapterTest, AdapterModelTestSetup)
WINML_TEST(AdapterTest, CreateModelFromPath)
WINML_TEST(AdapterTest, CreateModelFromData)
WINML_TEST(AdapterTest, CloneModel)
WINML_TEST(AdapterTest, ModelGetAuthor)
WINML_TEST(AdapterTest, ModelGetName)
WINML_TEST(AdapterTest, ModelGetDomain)
WINML_TEST(AdapterTest, ModelGetDescription)
WINML_TEST(AdapterTest, ModelGetVersion)
WINML_TEST(AdapterTest, ModelGetInputCount)
WINML_TEST(AdapterTest, ModelGetOutputCount)
WINML_TEST(AdapterTest, ModelGetInputName)
WINML_TEST(AdapterTest, ModelGetOutputName)
WINML_TEST(AdapterTest, ModelGetInputDescription)
WINML_TEST(AdapterTest, ModelGetOutputDescription)
WINML_TEST(AdapterTest, ModelGetInputTypeInfo)
WINML_TEST(AdapterTest, ModelGetOutputTypeInfo)
WINML_TEST(AdapterTest, ModelGetMetadataCount)
WINML_TEST(AdapterTest, ModelGetMetadata)
WINML_TEST(AdapterTest, ModelEnsureNoFloat16)
WINML_TEST(AdapterTest, EnvConfigureCustomLoggerAndProfiler)
WINML_TEST_CLASS_END()
