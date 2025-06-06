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
  using VendorId = uint32_t;
  using Alignment = size_t;

  // Pre-defined device types.
  static const DeviceType CPU = 0;
  static const DeviceType GPU = 1;
  static const DeviceType FPGA = 2;
  static const DeviceType NPU = 3;
  static const DeviceType DML = 4;

  struct MemType {
    // Pre-defined memory types.
    static const MemoryType DEFAULT = 0;
    static const MemoryType HOST_ACCESSIBLE = 5;  // Device memory that is accessible from host and device.

    // these are deprecated. use vendor_id instead
    static const MemoryType CUDA_PINNED = 1;
    static const MemoryType HIP_PINNED = 2;
    static const MemoryType CANN_PINNED = 3;
    static const MemoryType QNN_HTP_SHARED = 4;
  };

  // PCI vendor ids we explicitly use to map legacy values.
  struct VendorIds {
    static const VendorId NONE = 0x0000;  // No vendor ID. Valid for DeviceType::CPU with MemType::DEFAULT or for generic allocators.
    static const VendorId AMD = 0x1002;   // AMD/ATI
    static const VendorId NVIDIA = 0x10DE;
    static const VendorId ARM = 0x13B5;
    static const VendorId MICROSOFT = 0x1414;  // DML EP
    static const VendorId HUAWEI = 0x19E5;     // CANN EP
    static const VendorId QUALCOMM = 0x5143;
    static const VendorId INTEL = 0x8086;
  };

  constexpr OrtDevice(DeviceType device_type_, MemoryType memory_type_, VendorId vendor_id_, DeviceId device_id_,
                      Alignment alignment) noexcept
      : device_type(device_type_),
        memory_type(memory_type_),
        device_id(device_id_),
        vendor_id(vendor_id_),
        alignment(alignment) {
    // infer vendor id for old values
    // TODO: Can we remove them completely?
    if (memory_type != MemType::DEFAULT && memory_type != MemType::HOST_ACCESSIBLE) {
      switch (memory_type_) {
        case MemType::CUDA_PINNED:
          vendor_id = VendorIds::NVIDIA;
          memory_type = MemType::HOST_ACCESSIBLE;
          break;
        case MemType::HIP_PINNED:
          vendor_id = VendorIds::AMD;
          memory_type = MemType::HOST_ACCESSIBLE;
          break;
        case MemType::CANN_PINNED:
          vendor_id = VendorIds::HUAWEI;
          memory_type = MemType::HOST_ACCESSIBLE;
          break;
        case MemType::QNN_HTP_SHARED:
          vendor_id = VendorIds::QUALCOMM;
          memory_type = MemType::HOST_ACCESSIBLE;
          break;
      };
    }
  }

  constexpr OrtDevice(DeviceType device_type_, MemoryType memory_type_, VendorId vendor_id_,
                      DeviceId device_id_) noexcept
      : OrtDevice(device_type_, memory_type_, vendor_id_, device_id_, 0) {}

  constexpr OrtDevice() noexcept : OrtDevice(CPU, MemType::DEFAULT, VendorIds::NONE, 0) {}

  DeviceType Type() const noexcept {
    return device_type;
  }

  MemoryType MemType() const noexcept {
    return memory_type;
  }

  VendorId Vendor() const noexcept {
    return vendor_id;
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
         << " VendorId:" << vendor_id
         << " DeviceId:" << device_id
         << " Alignment:" << alignment
         << "]";
    return ostr.str();
  }

  // This is to make OrtDevice a valid key in hash tables
  size_t Hash() const {
    auto h = std::hash<int>()(device_type);
    onnxruntime::HashCombine(memory_type, h);
    onnxruntime::HashCombine(vendor_id, h);
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
    if (vendor_id != other.vendor_id)
      return vendor_id < other.vendor_id;
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

  uint32_t vendor_id;

  // Required alignment
  Alignment alignment;
};

inline bool operator==(const OrtDevice& left, const OrtDevice& other) {
  return left.Type() == other.Type() &&
         left.MemType() == other.MemType() &&
         left.Vendor() == other.Vendor() &&
         left.Id() == other.Id() &&
         left.GetAlignment() == other.GetAlignment();
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
