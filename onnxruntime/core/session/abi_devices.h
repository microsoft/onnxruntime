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
  int32_t vendor_id;   // GPU has id
  std::string vendor;  // CPU uses string
  int32_t bus_id;
  OrtKeyValuePairs metadata;

  static size_t Hash(const OrtHardwareDevice& hd) {
    auto h = std::hash<int>()(hd.bus_id);  // start with a field that always has a non-trivial value
    onnxruntime::HashCombine(hd.vendor_id, h);
    onnxruntime::HashCombine(hd.vendor, h);
    onnxruntime::HashCombine(hd.type, h);
    // skip the metadata for now

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
           lhs.vendor == rhs.vendor &&
           lhs.bus_id == rhs.bus_id &&
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
