//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

// stl
#include <algorithm>
#include <codecvt>
#include <fcntl.h>
#include <future>
#include <io.h>
#include <locale>
#include <numeric>
#include <random>
#include <string_view>
#include <utility>
#include <vector>


// IUnknown must be declared before winrt/base.h is included to light up support for native COM
// interfaces with C++/WinRT types (e.g. winrt::com_ptr<ITensorNative>).
#include <Unknwn.h>
#include "winrt/base.h"
#include "winrt/Windows.Foundation.Collections.h"
#include "comp_generated/winrt/windows.ai.machinelearning.h"

// WinML
#include "Windows.AI.MachineLearning.Native.h"

#include "runtimeParameters.h"

#define EXPECT_THROW_SPECIFIC(statement, exception, condition)  \
    EXPECT_THROW(                                               \
        try {                                                   \
            statement;                                          \
        } catch (const exception& e) {                          \
            EXPECT_TRUE(condition(e));                          \
            throw;                                              \
        }                                                       \
    , exception);

// For old versions of gtest without GTEST_SKIP, stream the message and return success instead
#ifndef GTEST_SKIP
#define GTEST_SKIP_(message) \
    return GTEST_MESSAGE_(message, ::testing::TestPartResult::kSuccess)
#define GTEST_SKIP GTEST_SKIP_("")
#endif

#ifndef INSTANTIATE_TEST_SUITE_P
// Use the old name, removed in newer versions of googletest
#define INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_CASE_P
#endif

#define GPUTEST \
    if (auto noGpuTests = RuntimeParameters::Parameters.find("noGPUtests");             \
        noGpuTests != RuntimeParameters::Parameters.end() && noGpuTests->second != "0") \
    {                                                                                   \
        GTEST_SKIP << "GPU tests disabled";                                             \
    }

#define SKIP_EDGECORE \
    if (auto isEdgeCore = RuntimeParameters::Parameters.find("EdgeCore");               \
        isEdgeCore != RuntimeParameters::Parameters.end() && isEdgeCore->second != "0") \
    {                                                                                   \
        GTEST_SKIP << "Test can't be run in EdgeCore";                                  \
    }
