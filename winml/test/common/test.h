// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

using VoidTest = void (*)();
using SetupTest = VoidTest;

constexpr bool alwaysTrue() {
    return true;
}
constexpr bool alwaysFalse() {
    return false;
}
#define WINML_SUPRESS_UNREACHABLE_BELOW(statement)    \
    if (alwaysTrue()) { statement; }

#if !defined(BUILD_GOOGLE_TEST) && !defined(BUILD_TAEF_TEST)
#define BUILD_GOOGLE_TEST
#endif

#ifdef BUILD_GOOGLE_TEST
#include "googleTestMacros.h"
#else
#ifdef BUILD_TAEF_TEST
#include "taefTestMacros.h"
#endif
#endif
