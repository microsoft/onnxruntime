// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <string_view>

#include "core/platform/telemetry.h"

namespace Microsoft::Applications::Events {
class EventProperties;
}

namespace onnxruntime::telemetry_internal {

::Microsoft::Applications::Events::EventProperties BuildExecutionProviderEvent(const LUID& adapter_luid);
::Microsoft::Applications::Events::EventProperties BuildDriverInfoEvent(
    std::string_view device_class, std::wstring_view driver_names, std::wstring_view driver_versions);
::Microsoft::Applications::Events::EventProperties BuildProviderOptionsEvent(
    const std::string& provider_id, const std::string& provider_options, bool capture_state);

}  // namespace onnxruntime::telemetry_internal
