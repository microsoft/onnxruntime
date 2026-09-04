// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

// Catalog of PCI-SIG vendor ID constants for ONNX Runtime
// (https://pcisig.com/membership/member-companies).
// A vendor may have more than one assignment: kAmdAti (0x1002) is the former ATI assignment used
// by AMD GPUs, while kAmd (0x1022) is the AMD assignment used by CPUs and Ryzen AI NPUs.
// Qualcomm has two assignments: kQualcommInc (0x5143) and kQualcommTechnologies (0x17CB).
// ORT represents the ACPI vendor identifier 'QCOM' (0x4D4F4351) for Qualcomm Snapdragon CPUs.
// This is not a PCI vendor ID (see cpuid_info_vendor.cc and the QNN EP factory).
namespace onnxruntime {
namespace pci_vendor_ids {

inline constexpr uint32_t kAmdAti = 0x1002;
inline constexpr uint32_t kIbm = 0x1014;
inline constexpr uint32_t kAmd = 0x1022;
inline constexpr uint32_t kApple = 0x106B;
inline constexpr uint32_t kNvidia = 0x10DE;
inline constexpr uint32_t kArm = 0x13B5;
inline constexpr uint32_t kMicrosoft = 0x1414;
inline constexpr uint32_t kQualcommTechnologies = 0x17CB;
inline constexpr uint32_t kHuawei = 0x19E5;
inline constexpr uint32_t kQualcommInc = 0x5143;
inline constexpr uint32_t kIntel = 0x8086;

}  // namespace pci_vendor_ids
}  // namespace onnxruntime
