// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <sstream>
#include "core/common/hash_combine.h"

// Struct to represent a physical device.
struct OrtDevice {
  using DeviceType = int8_t;
  using MemoryType = int8_t;
  using DeviceId = int16_t;
  using Alignment = size_t;

  // Pre-defined device types.
  static const DeviceType CPU = 0;
  static const DeviceType GPU = 1;  // Nvidia or AMD
  static const DeviceType FPGA = 2;
  static const DeviceType NPU = 3;  // Ascend
  static const DeviceType DML = 4;

  struct MemType {
    // Pre-defined memory types.
    static const MemoryType DEFAULT = 0;
    static const MemoryType CUDA_PINNED = 1;
    static const MemoryType HIP_PINNED = 2;
    static const MemoryType CANN_PINNED = 3;
    static const MemoryType QNN_HTP_SHARED = 4;
  };

  constexpr OrtDevice(DeviceType device_type_, MemoryType memory_type_, DeviceId device_id_, Alignment alignment) noexcept
      : device_type(device_type_),
        memory_type(memory_type_),
        device_id(device_id_),
        alignment(alignment) {}

  constexpr OrtDevice(DeviceType device_type_, MemoryType memory_type_, DeviceId device_id_) noexcept
      : OrtDevice(device_type_, memory_type_, device_id_, 0) {}

  constexpr OrtDevice() noexcept : OrtDevice(CPU, MemType::DEFAULT, 0) {}

  DeviceType Type() const noexcept {
    return device_type;
  }

  MemoryType MemType() const noexcept {
    return memory_type;
  }

  DeviceId Id() const noexcept {
    return device_id;
  }

  Alignment GetAlignment() const noexcept {
    return alignment;
  }

  std::string ToString() const {
    std::ostringstream ostr;
    ostr << "Device:["
         << "DeviceType:" << static_cast<int>(device_type)
         << " MemoryType:" << static_cast<int>(memory_type)
         << " DeviceId:" << device_id
         << " Alignment:" << alignment
         << "]";
    return ostr.str();
  }

  // This is to make OrtDevice a valid key in hash tables
  size_t Hash() const {
    auto h = std::hash<int>()(device_type);
    onnxruntime::HashCombine(memory_type, h);
    onnxruntime::HashCombine(device_id, h);
    onnxruntime::HashCombine(alignment, h);
    return h;
  }

  // To make OrtDevice become a valid key in std map
  bool operator<(const OrtDevice& other) const {
    if (device_type != other.device_type)
      return device_type < other.device_type;
    if (memory_type != other.memory_type)
      return memory_type < other.memory_type;
    if (device_id != other.device_id)
      return device_id < other.device_id;

    return alignment < other.alignment;
  }

 private:
  // Device type.
  int32_t device_type : 8;

  // Memory type.
  int32_t memory_type : 8;

  // Device index.
  int32_t device_id : 16;

  // Required alignment
  Alignment alignment;
};

inline bool operator==(const OrtDevice& left, const OrtDevice& other) {
  return left.Id() == other.Id() && left.MemType() == other.MemType() && left.Type() == other.Type() && left.GetAlignment() == other.GetAlignment();
}

inline bool operator!=(const OrtDevice& left, const OrtDevice& other) {
  return !(left == other);
}

namespace std {
template <>
struct hash<OrtDevice> {
  size_t operator()(const OrtDevice& i) const {
    return i.Hash();
  }
};
}  // namespace std
