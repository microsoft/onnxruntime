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

#include "test.h"

// IUnknown must be declared before winrt/base.h is included to light up support for native COM
// interfaces with C++/WinRT types (e.g. winrt::com_ptr<ITensorNative>).
#include <Unknwn.h>
#include "winrt/base.h"
#include "winrt/Windows.Foundation.Collections.h"
#include "comp_generated/winrt/windows.ai.machinelearning.h"

// WinML
#include "Windows.AI.MachineLearning.Native.h"

#define EXPECT_THROW_SPECIFIC(statement, exception, condition)  \
    EXPECT_THROW(                                               \
        try {                                                   \
            statement;                                          \
        } catch (const exception& e) {                          \
            EXPECT_TRUE(condition(e));                          \
            throw;                                              \
        }                                                       \
    , exception);

#ifndef INSTANTIATE_TEST_SUITE_P
// Use the old name, removed in newer versions of googletest
#define INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_CASE_P
#endif