// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test.h"
struct LearningModelApiTestsApi
{
  SetupClass LearningModelAPITestsClassSetup;
  VoidTest CreateModelFromFilePath;
  VoidTest CreateModelFromUnicodeFilePath;
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
  VoidTest CheckLearningModelPixelRange;
  VoidTest CloseModelCheckEval;
  VoidTest CloseModelNoNewSessions;
  VoidTest CheckMetadataCaseInsensitive;
  VoidTest CreateCorruptModel;
};
const LearningModelApiTestsApi& getapi();

WINML_TEST_CLASS_BEGIN(LearningModelAPITests)
WINML_TEST_CLASS_SETUP_CLASS(LearningModelAPITestsClassSetup)
WINML_TEST_CLASS_BEGIN_TESTS
WINML_TEST(LearningModelAPITests, CreateModelFromFilePath)
WINML_TEST(LearningModelAPITests, CreateModelFromUnicodeFilePath)
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
WINML_TEST(LearningModelAPITests, CheckLearningModelPixelRange)
WINML_TEST(LearningModelAPITests, CloseModelNoNewSessions)
WINML_TEST(LearningModelAPITests, CloseModelCheckEval)
WINML_TEST(LearningModelAPITests, CheckMetadataCaseInsensitive)
WINML_TEST(LearningModelAPITests, CreateCorruptModel)
WINML_TEST_CLASS_END()