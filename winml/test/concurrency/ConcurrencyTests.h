// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "test.h"

struct ConcurrencyTestsApi
{
  SetupClass ConcurrencyTestsClassSetup;
  VoidTest LoadBindEvalSqueezenetRealDataWithValidationConcurrently;
  VoidTest MultiThreadLoadModel;
  VoidTest MultiThreadMultiSession;
  VoidTest MultiThreadMultiSessionGpu;
  VoidTest MultiThreadSingleSession;
  VoidTest MultiThreadSingleSessionGpu;
  VoidTest EvalAsyncDifferentModels;
  VoidTest EvalAsyncDifferentSessions;
  VoidTest EvalAsyncDifferentBindings;
};
const ConcurrencyTestsApi& getapi();

WINML_TEST_CLASS_BEGIN(ConcurrencyTests)
WINML_TEST_CLASS_SETUP_CLASS(ConcurrencyTestsClassSetup)
WINML_TEST_CLASS_BEGIN_TESTS
WINML_TEST(ConcurrencyTests, LoadBindEvalSqueezenetRealDataWithValidationConcurrently)
WINML_TEST(ConcurrencyTests, MultiThreadLoadModel)
WINML_TEST(ConcurrencyTests, MultiThreadMultiSession)
WINML_TEST(ConcurrencyTests, MultiThreadSingleSession)
WINML_TEST(ConcurrencyTests, EvalAsyncDifferentModels)
WINML_TEST(ConcurrencyTests, EvalAsyncDifferentSessions)
WINML_TEST(ConcurrencyTests, EvalAsyncDifferentBindings)
WINML_TEST(ConcurrencyTests, MultiThreadMultiSessionGpu)
WINML_TEST(ConcurrencyTests, MultiThreadSingleSessionGpu)
WINML_TEST_CLASS_END()

// indices for imagenet label
static constexpr uint32_t TABBY_CAT_INDEX = 281;
static constexpr uint32_t TENCH_INDEX = 0;
// concurrency bugs are often race conditions and hard to catch deterministically.

// there are several approach's to find them, from consistent testing to random
// style stress/fuzz testing

// the testing strategy for *this* test is:
// - use a consistent and reasonable number of threads (10 vs. 1000)
// - run them for a consistent and long enough period of time (60 seconds) .
// - the smaller number of threads is also to make sure memory pressure is not an issue
// - on pre checkin CI test machines
static constexpr uint32_t NUM_THREADS = 10;
static constexpr uint32_t NUM_SECONDS = 10;
