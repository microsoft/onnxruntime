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

#define GPUTEST                                                                                            \
{                                                                                                          \
    bool noGPUTests;                                                                                       \
    if (SUCCEEDED(RuntimeParameters::TryGetValue(L"noGPUtests", noGPUTests)) && noGPUTests)                \
    {                                                                                                      \
        Log::Result(TestResults::Skipped, L"This test is disabled by the noGPUTests runtime parameter.");  \
        return;                                                                                            \
    }                                                                                                      \
}

#define SKIP_EDGECORE                                                                                      \
{                                                                                                          \
    bool edgeCoreRun;                                                                                      \
    if (SUCCEEDED(RuntimeParameters::TryGetValue(L"EdgeCore", edgeCoreRun)) && edgeCoreRun)                \
    {                                                                                                      \
        Log::Result(TestResults::Skipped, L"This test is disabled by the EdgeCore runtime parameter.");    \
        return;                                                                                            \
    }                                                                                                      \
}
