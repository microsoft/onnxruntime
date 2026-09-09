// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <napi.h>

namespace Ort {
struct SessionOptions;
}

// parse a Javascript session options object and fill the native SessionOptions object.
// 'requires_device_serialization' reports whether any requested execution provider keeps device
// state that cannot be used concurrently, so runs on the session have to be serialized process-wide.
void ParseSessionOptions(const Napi::Object options, Ort::SessionOptions& sessionOptions,
                         bool* requires_device_serialization = nullptr);

// parse a Javascript session options object and prepare the preferred output locations.
void ParsePreferredOutputLocations(const Napi::Object options, const std::vector<std::string>& outputNames, std::vector<int>& preferredOutputLocations);