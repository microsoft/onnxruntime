// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test.h"

struct LearningModelBindingAPITestsApi {
  SetupClass LearningModelBindingAPITestsClassSetup;
  VoidTest CpuSqueezeNet;
  VoidTest CpuSqueezeNetEmptyOutputs;
  VoidTest CpuSqueezeNetUnboundOutputs;
  VoidTest CpuSqueezeNetBindInputTensorAsInspectable;
  VoidTest CastMapInt64;
  VoidTest DictionaryVectorizerMapInt64;
  VoidTest DictionaryVectorizerMapString;
  VoidTest ZipMapInt64;
  VoidTest ZipMapInt64Unbound;
  VoidTest ZipMapString;
  VoidTest GpuSqueezeNet;
  VoidTest GpuSqueezeNetEmptyOutputs;
  VoidTest GpuSqueezeNetUnboundOutputs;
  VoidTest ImageBindingDimensions;
  VoidTest VerifyInvalidBindExceptions;
  VoidTest BindInvalidInputName;
  VoidTest VerifyOutputAfterEvaluateAsyncCalledTwice;
  VoidTest VerifyOutputAfterImageBindCalledTwice;
  VoidTest SequenceLengthTensorFloat;
  VoidTest SequenceConstructTensorString;
};
const LearningModelBindingAPITestsApi& getapi();

WINML_TEST_CLASS_BEGIN(LearningModelBindingAPITests)
WINML_TEST_CLASS_SETUP_CLASS(LearningModelBindingAPITestsClassSetup)
WINML_TEST_CLASS_BEGIN_TESTS
WINML_TEST(LearningModelBindingAPITests, CpuSqueezeNet)
WINML_TEST(LearningModelBindingAPITests, CpuSqueezeNetEmptyOutputs)
WINML_TEST(LearningModelBindingAPITests, CpuSqueezeNetUnboundOutputs)
WINML_TEST(LearningModelBindingAPITests, CpuSqueezeNetBindInputTensorAsInspectable)
WINML_TEST(LearningModelBindingAPITests, CastMapInt64)
WINML_TEST(LearningModelBindingAPITests, DictionaryVectorizerMapInt64)
WINML_TEST(LearningModelBindingAPITests, DictionaryVectorizerMapString)
WINML_TEST(LearningModelBindingAPITests, ZipMapInt64)
WINML_TEST(LearningModelBindingAPITests, ZipMapInt64Unbound)
WINML_TEST(LearningModelBindingAPITests, ZipMapString)
WINML_TEST(LearningModelBindingAPITests, VerifyOutputAfterEvaluateAsyncCalledTwice)
WINML_TEST(LearningModelBindingAPITests, VerifyOutputAfterImageBindCalledTwice)
WINML_TEST(LearningModelBindingAPITests, SequenceLengthTensorFloat)
WINML_TEST(LearningModelBindingAPITests, SequenceConstructTensorString)
WINML_TEST(LearningModelBindingAPITests, GpuSqueezeNet)
WINML_TEST(LearningModelBindingAPITests, GpuSqueezeNetEmptyOutputs)
WINML_TEST(LearningModelBindingAPITests, GpuSqueezeNetUnboundOutputs)
WINML_TEST(LearningModelBindingAPITests, ImageBindingDimensions)
WINML_TEST(LearningModelBindingAPITests, VerifyInvalidBindExceptions)
WINML_TEST(LearningModelBindingAPITests, BindInvalidInputName)
WINML_TEST_CLASS_END()
