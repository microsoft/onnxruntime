// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "test.h"

struct ConcurrencyTestsApi
{
  SetupTest ConcurrencyTestsApiSetup;
  VoidTest LoadBindEvalSqueezenetRealDataWithValidationConcurrently;
  VoidTest MultiThreadLoadModel;
  VoidTest MultiThreadSession;
  VoidTest EvalAsyncDifferentModels;
  VoidTest EvalAsyncDifferentSessions;
  VoidTest EvalAsyncDifferentBindings;
};
const ConcurrencyTestsApi& getapi();

WINML_TEST_CLASS_BEGIN_WITH_SETUP(ConcurrencyTests, ConcurrencyTestsApiSetup)
WINML_TEST(ConcurrencyTests, LoadBindEvalSqueezenetRealDataWithValidationConcurrently)
WINML_TEST(ConcurrencyTests, MultiThreadLoadModel)
WINML_TEST(ConcurrencyTests, MultiThreadSession)
WINML_TEST(ConcurrencyTests, EvalAsyncDifferentModels)
WINML_TEST(ConcurrencyTests, EvalAsyncDifferentSessions)
WINML_TEST(ConcurrencyTests, EvalAsyncDifferentBindings)
WINML_TEST_CLASS_END()

// indices for imagenet label
static constexpr uint32_t s_IndexTabbyCat = 281;
static constexpr uint32_t s_IndexTench = 0;
// maximum number of threads to run concurrent jobs
static constexpr uint32_t s_max_threads = 30;
