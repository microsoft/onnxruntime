// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <napi.h>

namespace Ort {
struct RunOptions;
}

// parse a Javascript run options object and fill the native RunOptions object.
void ParseRunOptions(const Napi::Object options, Ort::RunOptions &runOptions);
