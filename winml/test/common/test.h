// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

using VoidTest = void (*)();
using SetupTest = VoidTest;

constexpr bool alwaysTrue() {
    return true;
}
#define WINML_SUPRESS_UNREACHABLE_BELOW(statement)    \
    if (alwaysTrue()) { statement; }

#ifdef BUILD_GOOGLE_TEST
#include "googleTestMacros.h"
#else
#ifdef BUILD_TAEF_TEST
#include "taefTestMacros.h"
#endif
#endif
