// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "TimerHelper.h"

enum WinMLRuntimePerf {
  kLoadModel = 0,
  kEvaluateModel,
  kCount
};

extern Profiler<WinMLRuntimePerf> profiler;