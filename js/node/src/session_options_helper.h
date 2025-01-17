// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <napi.h>

namespace Ort {
struct SessionOptions;
}

// parse a Javascript session options object and fill the native SessionOptions object.
void ParseSessionOptions(const Napi::Object options, Ort::SessionOptions& sessionOptions);

// parse a Javascript session options object and prepare the preferred output locations.
void ParsePreferredOutputLocations(const Napi::Object options, const std::vector<std::string>& outputNames, std::vector<int>& preferredOutputLocations);