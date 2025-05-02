// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>

#include "core/common/hash_combine.h"
#include "core/session/abi_key_value_pairs.h"
#include "core/session/onnxruntime_c_api.h"

struct OrtHardwareDevice {
  OrtHardwareDeviceType type;
  uint32_t vendor_id;
  uint32_t device_id;  // identifies the hardware type when combined with vendor id. not a unique id.
  std::string vendor;
  OrtKeyValuePairs metadata;

  static size_t Hash(const OrtHardwareDevice& hd) {
    auto h = std::hash<int>()(hd.device_id);  // start with a field that always has a non-trivial value
    onnxruntime::HashCombine(hd.vendor_id, h);
    onnxruntime::HashCombine(hd.vendor, h);
    onnxruntime::HashCombine(hd.type, h);
    for (const auto& [key, value] : hd.metadata.entries) {
      onnxruntime::HashCombine(key, h);
      onnxruntime::HashCombine(value, h);
    }

    return h;
  }
};

// This is to make OrtHardwareDevice a valid key in hash tables
namespace std {
template <>
struct hash<OrtHardwareDevice> {
  size_t operator()(const OrtHardwareDevice& hd) const {
    return OrtHardwareDevice::Hash(hd);
  }
};

template <>
struct equal_to<OrtHardwareDevice> {
  bool operator()(const OrtHardwareDevice& lhs, const OrtHardwareDevice& rhs) const noexcept {
    return lhs.type == rhs.type &&
           lhs.vendor_id == rhs.vendor_id &&
           lhs.device_id == rhs.device_id &&
           lhs.vendor == rhs.vendor &&
           lhs.metadata.keys == rhs.metadata.keys &&
           lhs.metadata.values == rhs.metadata.values;
  }
};
}  // namespace std

struct OrtEpDevice {
  std::string ep_name;
  std::string ep_vendor;
  const OrtHardwareDevice* device;

  OrtKeyValuePairs ep_metadata;
  OrtKeyValuePairs ep_options;

  OrtEpFactory* ep_factory;
};
