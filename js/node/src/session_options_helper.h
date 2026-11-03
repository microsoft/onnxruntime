// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <napi.h>

#include <vector>

namespace Ort {
struct SessionOptions;
}

// parse a Javascript session options object and fill the native SessionOptions object.
// 'requiresDeviceSerialization' is set if a requested execution provider has device state shared across sessions,
// so that all work on the session has to hold the DeviceLock.
void ParseSessionOptions(const Napi::Object options, Ort::SessionOptions& sessionOptions,
                         bool& requiresDeviceSerialization,
                         std::vector<std::vector<char>>* externalDataBuffers = nullptr);

// parse a Javascript session options object and prepare the preferred output locations.
void ParsePreferredOutputLocations(const Napi::Object options, const std::vector<std::string>& outputNames, std::vector<int>& preferredOutputLocations);