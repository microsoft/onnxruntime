// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test.h"
struct LearningModelApiTestsApi
{
  SetupClass LearningModelAPITestsClassSetup;
  SetupTest LearningModelAPITestsGpuMethodSetup;
  VoidTest CreateModelFromFilePath;
  VoidTest CreateModelFileNotFound;
  VoidTest CreateModelFromIStorage;
  VoidTest CreateModelFromIStorageOutsideCwd;
  VoidTest CreateModelFromIStream;
  VoidTest ModelGetAuthor;
  VoidTest ModelGetName;
  VoidTest ModelGetDomain;
  VoidTest ModelGetDescription;
  VoidTest ModelGetVersion;
  VoidTest EnumerateInputs;
  VoidTest EnumerateOutputs;
  VoidTest CloseModelCheckMetadata;
  VoidTest CloseModelCheckEval;
  VoidTest CloseModelNoNewSessions;
};
const LearningModelApiTestsApi& getapi();

WINML_TEST_CLASS_BEGIN(LearningModelAPITests)
WINML_TEST_CLASS_SETUP_CLASS(LearningModelAPITestsClassSetup)
WINML_TEST_CLASS_BEGIN_TESTS
WINML_TEST(LearningModelAPITests, CreateModelFromFilePath)
WINML_TEST(LearningModelAPITests, CreateModelFileNotFound)
WINML_TEST(LearningModelAPITests, CreateModelFromIStorage)
WINML_TEST(LearningModelAPITests, CreateModelFromIStorageOutsideCwd)
WINML_TEST(LearningModelAPITests, CreateModelFromIStream)
WINML_TEST(LearningModelAPITests, ModelGetAuthor)
WINML_TEST(LearningModelAPITests, ModelGetName)
WINML_TEST(LearningModelAPITests, ModelGetDomain)
WINML_TEST(LearningModelAPITests, ModelGetDescription)
WINML_TEST(LearningModelAPITests, ModelGetVersion)
WINML_TEST(LearningModelAPITests, EnumerateInputs)
WINML_TEST(LearningModelAPITests, EnumerateOutputs)
WINML_TEST(LearningModelAPITests, CloseModelCheckMetadata)
WINML_TEST(LearningModelAPITests, CloseModelNoNewSessions)
WINML_TEST_CLASS_END()

WINML_TEST_CLASS_BEGIN(LearningModelAPITestsGpu)
WINML_TEST_CLASS_SETUP_CLASS(LearningModelAPITestsClassSetup)
WINML_TEST_CLASS_SETUP_METHOD(LearningModelAPITestsGpuMethodSetup)
WINML_TEST_CLASS_BEGIN_TESTS
WINML_TEST(LearningModelAPITestsGpu, CloseModelCheckEval)
WINML_TEST_CLASS_END()