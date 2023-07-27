// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test.h"
#include "core/providers/winml/winml_provider_factory.h"
#include "winml_adapter_c_api.h"

struct AdapterTestApi {
  SetupClass AdapterTestSetup;
  TeardownClass AdapterTestTeardown;
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
const WinmlAdapterApi* winml_adapter_api;
const OrtApi* ort_api;
OrtModel* squeezenet_model = nullptr;
OrtModel* metadata_model = nullptr;
OrtModel* float16_Model = nullptr;
static bool logging_function_called = false;
static bool profiling_function_called = false;

WINML_TEST_CLASS_BEGIN(AdapterTest)
WINML_TEST_CLASS_SETUP_CLASS(AdapterTestSetup)
WINML_TEST_CLASS_TEARDOWN_CLASS(AdapterTestTeardown)
WINML_TEST_CLASS_BEGIN_TESTS
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
