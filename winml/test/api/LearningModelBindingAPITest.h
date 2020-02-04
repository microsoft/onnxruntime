// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test.h"

struct LearningModelBindingAPITestApi {
  SetupTest LearningModelBindingAPITestSetup;
  SetupTest LearningModelBindingAPITestGpuSetup;
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
};
const LearningModelBindingAPITestApi& getapi();

WINML_TEST_CLASS_BEGIN_WITH_SETUP(LearningModelBindingAPITest, LearningModelBindingAPITestSetup)
WINML_TEST(LearningModelBindingAPITest, CpuSqueezeNet)
WINML_TEST(LearningModelBindingAPITest, CpuSqueezeNetEmptyOutputs)
WINML_TEST(LearningModelBindingAPITest, CpuSqueezeNetUnboundOutputs)
WINML_TEST(LearningModelBindingAPITest, CpuSqueezeNetBindInputTensorAsInspectable)
WINML_TEST(LearningModelBindingAPITest, CastMapInt64)
WINML_TEST(LearningModelBindingAPITest, DictionaryVectorizerMapInt64)
WINML_TEST(LearningModelBindingAPITest, DictionaryVectorizerMapString)
WINML_TEST(LearningModelBindingAPITest, ZipMapInt64)
WINML_TEST(LearningModelBindingAPITest, ZipMapInt64Unbound)
WINML_TEST(LearningModelBindingAPITest, ZipMapString)
WINML_TEST(LearningModelBindingAPITest, VerifyOutputAfterEvaluateAsyncCalledTwice)
WINML_TEST(LearningModelBindingAPITest, VerifyOutputAfterImageBindCalledTwice)
WINML_TEST_CLASS_END()

WINML_TEST_CLASS_BEGIN_WITH_SETUP(LearningModelBindingAPITestGpu, LearningModelBindingAPITestGpuSetup)
WINML_TEST(LearningModelBindingAPITestGpu, GpuSqueezeNet)
WINML_TEST(LearningModelBindingAPITestGpu, GpuSqueezeNetEmptyOutputs)
WINML_TEST(LearningModelBindingAPITestGpu, GpuSqueezeNetUnboundOutputs)
WINML_TEST(LearningModelBindingAPITestGpu, ImageBindingDimensions)
WINML_TEST(LearningModelBindingAPITestGpu, VerifyInvalidBindExceptions)
WINML_TEST(LearningModelBindingAPITestGpu, BindInvalidInputName)
WINML_TEST_CLASS_END()