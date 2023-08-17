// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

using VoidTest = void (*)();
using SetupClass = VoidTest;
using TeardownClass = VoidTest;
using SetupTest = VoidTest;
using TeardownTest = VoidTest;

#pragma warning(push)
#pragma warning(disable : 4505)  // unreferenced local function has been removed

constexpr bool alwaysTrue() {
  return true;
}
constexpr bool alwaysFalse() {
  return false;
}
#define WINML_SUPRESS_UNREACHABLE_BELOW(statement) \
  if (alwaysTrue()) {                              \
    statement;                                     \
  }

#ifdef BUILD_TAEF_TEST
#include "taefTestMacros.h"
#else
#include "googleTestMacros.h"
#endif

static void SkipTest() {
  WINML_SKIP_TEST("");
}

#pragma warning(pop)
