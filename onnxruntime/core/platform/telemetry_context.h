// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <string>
#include <string_view>

#include "core/platform/telemetry_sha256.h"

namespace onnxruntime::telemetry_internal {

// These PAL-populated fields are not needed by ORT. ext.os.locale is not populated by this SDK path.
inline constexpr std::array<const char*, 4> kSuppressedCommonContextFields{
    "AppInfo.Language",
    "UserInfo.Language",
    "UserInfo.TimeZone",
    "M365aInfo.EnrolledTenantId",
};

inline constexpr std::array<const char*, 3> kProcessInfoOnlyNetworkContextFields{
    "DeviceInfo.NetworkCost",
    "DeviceInfo.NetworkProvider",
    "DeviceInfo.NetworkType",
};

inline std::string FormatDeviceIdForTelemetry(std::string_view raw_device_id) {
  return raw_device_id.empty() ? std::string{} : "c:" + Sha256::HashStringHex(raw_device_id);
}

// ext.sdk.* is populated by a separate decorator. Keep it intact because epoch/sequence are
// per-event SDK ordering metadata and the public SDK has no field-level suppression control.

template <typename SemanticContext>
void SuppressUnneededCommonContext(SemanticContext& context) {
  for (const char* field : kSuppressedCommonContextFields) {
    context.SetCommonField(field, std::string{});
  }
}

template <typename SemanticContext>
void SetApplicationNameFromProcessName(SemanticContext& context, const std::string& process_name) {
  if (!process_name.empty()) {
    context.SetCommonField("AppInfo.Name", process_name);
  }
}

template <typename SemanticContext>
void SuppressNetworkContext(SemanticContext& context) {
  for (const char* field : kProcessInfoOnlyNetworkContextFields) {
    context.SetCommonField(field, std::string{});
  }
}

}  // namespace onnxruntime::telemetry_internal
