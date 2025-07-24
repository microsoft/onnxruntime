// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <sstream>
#include "core/common/common.h"
#include "core/common/hash_combine.h"

// fix clash with INTEL that is defined in
// MacOSX14.2.sdk/System/Library/Frameworks/Security.framework/Headers/oidsbase.h
#if defined(__APPLE__)
#undef INTEL
#endif

// Struct to represent a combination of physical device and memory type.
// A memory allocation and allocator have a specific OrtDevice associated with them, and this information is used
// to determine when data transfer is required.
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
  // this is used in the python API so we need to keep it for backward compatibility
  // it is only used in the OrtDevice ctor, and is mapped to GPU + VendorIds::MICROSOFT
  static const DeviceType DML = 4;

  struct MemType {
    static const MemoryType DEFAULT = 0;

    // deprecated values. MemType + VendorId is used to identify the memory type.
    enum Deprecated : MemoryType {
      CUDA_PINNED = 1,
      HIP_PINNED = 2,
      CANN_PINNED = 3,
      QNN_HTP_SHARED = 4,
    };

    // HOST_ACCESSIBLE memory is treated as CPU memory.
    // When creating an OrtDevice with MemType::HOST_ACCESSIBLE:
    //   - For memory that is only accessible by a specific device and CPU, use the specific device type and id.
    //   - When creating an OrtDevice for an EP allocator, you would typically use the same device type and id
    //     that the EP is registered with (i.e. the OrtDevice passed to the base IExecutionProvider constructor).
    //   - Otherwise use OrtDevice::CPU.
    static const MemoryType HOST_ACCESSIBLE = 5;
  };

  // PCI vendor ids
  enum VendorIds : VendorId {
    // No vendor ID. Valid for DeviceType::CPU + MemType::DEFAULT or for generic allocators like WebGPU.
    NONE = 0x0000,
    AMD = 0x1002,        // ROCm, MIGraphX EPs
    NVIDIA = 0x10DE,     // CUDA/TensorRT
    ARM = 0x13B5,        // ARM GPU EP
    MICROSOFT = 0x1414,  // DML EP
    HUAWEI = 0x19E5,     // CANN EP
    QUALCOMM = 0x5143,   // QNN DP
    INTEL = 0x8086,      // OpenVINO
  };

  constexpr OrtDevice(DeviceType device_type_, MemoryType memory_type_, VendorId vendor_id_, DeviceId device_id_,
                      Alignment alignment) /*noexcept*/
      : device_type(device_type_),
        memory_type(memory_type_),
        device_id(device_id_),
        vendor_id(vendor_id_),
        alignment(alignment) {
    // temporary to make sure we haven't missed any places where the deprecated values were used
    // ctor can go back to noexcept once everything is validated and this is removed`1
    ORT_ENFORCE(memory_type == MemType::DEFAULT || memory_type == MemType::HOST_ACCESSIBLE,
                "Invalid memory type: ", static_cast<int>(memory_type));

    if (device_type == DML) {
      device_type = GPU;
      vendor_id = VendorIds::MICROSOFT;
    }
  }

  constexpr OrtDevice(DeviceType device_type_, MemoryType memory_type_, VendorId vendor_id_,
                      DeviceId device_id_) noexcept
      : OrtDevice(device_type_, memory_type_, vendor_id_, device_id_, /*alignment*/ 0) {}

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

  // CPU or HOST_ACCESSIBLE memory.
  bool UsesCpuMemory() const noexcept {
    return device_type == CPU || memory_type == MemType::HOST_ACCESSIBLE;
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

  bool EqualIgnoringAlignment(const OrtDevice& other) const {
    return device_type == other.device_type &&
           memory_type == other.memory_type &&
           vendor_id == other.vendor_id &&
           device_id == other.device_id;
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
