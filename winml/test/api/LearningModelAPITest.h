// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test.h"
struct LearningModelApiTestApi
{
  SetupTest LearningModelAPITestSetup;
  SetupTest LearningModelAPITestGpuSetup;
  VoidTest CreateModelFromFilePath;
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
const LearningModelApiTestApi& getapi();

WINML_TEST_CLASS_BEGIN_WITH_SETUP(LearningModelAPITest, LearningModelAPITestSetup)
WINML_TEST(LearningModelAPITest, CreateModelFromFilePath)
WINML_TEST(LearningModelAPITest, CreateModelFromIStorage)
WINML_TEST(LearningModelAPITest, CreateModelFromIStorageOutsideCwd)
WINML_TEST(LearningModelAPITest, CreateModelFromIStream)
WINML_TEST(LearningModelAPITest, ModelGetAuthor)
WINML_TEST(LearningModelAPITest, ModelGetName)
WINML_TEST(LearningModelAPITest, ModelGetDomain)
WINML_TEST(LearningModelAPITest, ModelGetDescription)
WINML_TEST(LearningModelAPITest, ModelGetVersion)
WINML_TEST(LearningModelAPITest, EnumerateInputs)
WINML_TEST(LearningModelAPITest, EnumerateOutputs)
WINML_TEST(LearningModelAPITest, CloseModelCheckMetadata)
WINML_TEST(LearningModelAPITest, CloseModelNoNewSessions)
WINML_TEST_CLASS_END()

WINML_TEST_CLASS_BEGIN_WITH_SETUP(LearningModelAPITestGpu, LearningModelAPITestGpuSetup)
WINML_TEST(LearningModelAPITestGpu, CloseModelCheckEval)
WINML_TEST_CLASS_END()