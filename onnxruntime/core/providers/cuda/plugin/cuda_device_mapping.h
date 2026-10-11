// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "gsl/gsl"

namespace onnxruntime::cuda_plugin {

inline std::string NormalizePciBusId(std::string_view pci_bus_id) {
  std::string normalized{pci_bus_id};
  std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return normalized;
}

inline std::optional<int> FindCudaOrdinalForHardwareDeviceIdentity(
    std::string_view hardware_device_identity,
    gsl::span<const std::string> cuda_device_identities,
    gsl::span<const uint8_t> assigned_cuda_ordinals) {
  if (hardware_device_identity.empty()) {
    return std::nullopt;
  }

  for (size_t i = 0; i < cuda_device_identities.size(); ++i) {
    if (assigned_cuda_ordinals[i] == 0 &&
        cuda_device_identities[i] == hardware_device_identity) {
      return static_cast<int>(i);
    }
  }

  return std::nullopt;
}

inline std::optional<int> FindCudaOrdinalWithoutIdentity(
    gsl::span<const std::string> cuda_device_identities,
    gsl::span<const uint8_t> assigned_cuda_ordinals) {
  for (size_t i = 0; i < cuda_device_identities.size(); ++i) {
    if (assigned_cuda_ordinals[i] == 0 && cuda_device_identities[i].empty()) {
      return static_cast<int>(i);
    }
  }

  return std::nullopt;
}

}  // namespace onnxruntime::cuda_plugin
