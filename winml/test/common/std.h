// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
#include <wil/cppwinrt.h>
#include "winrt/base.h"
#include "winrt/Windows.Foundation.Collections.h"
#include "comp_generated/winrt/windows.ai.machinelearning.h"

// WinML
#include "Windows.AI.MachineLearning.Native.h"
