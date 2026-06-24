// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/common/cpuid_uarch.h"

#if defined(CPUIDINFO_ARCH_ARM)

namespace onnxruntime {
namespace test {

// Helper to construct a MIDR value from implementer and part fields.
// MIDR layout: [31:24] implementer, [23:20] variant, [19:16] arch, [15:4] part, [3:0] revision
static uint32_t MakeMIDR(uint32_t implementer, uint32_t part,
                         uint32_t variant = 0, uint32_t revision = 0) {
  return (implementer << 24) | (variant << 20) | (0xF << 16) | (part << 4) | revision;
}

// ARM64-only part numbers (guarded by #if in cpuid_uarch.cc)
#if defined(_M_ARM64) || defined(__aarch64__)

TEST(CpuidUarch, QualcommOryon) {
  // Qualcomm Oryon: implementer = 'Q' (0x51), part = 0x001
  uint32_t uarch = cpuinfo_uarch_unknown;
  decodeMIDR(MakeMIDR('Q', 0x001), &uarch);
  EXPECT_EQ(uarch, static_cast<uint32_t>(cpuinfo_uarch_oryon));
}

TEST(CpuidUarch, QualcommOryonV3) {
  // Qualcomm Oryon V3: implementer = 'Q' (0x51), part = 0x002
  uint32_t uarch = cpuinfo_uarch_unknown;
  decodeMIDR(MakeMIDR('Q', 0x002), &uarch);
  EXPECT_EQ(uarch, static_cast<uint32_t>(cpuinfo_uarch_oryon_v3));
}

TEST(CpuidUarch, QualcommFalkor) {
  uint32_t uarch = cpuinfo_uarch_unknown;
  decodeMIDR(MakeMIDR('Q', 0xC00), &uarch);
  EXPECT_EQ(uarch, static_cast<uint32_t>(cpuinfo_uarch_falkor));
}

TEST(CpuidUarch, QualcommSaphira) {
  uint32_t uarch = cpuinfo_uarch_unknown;
  decodeMIDR(MakeMIDR('Q', 0xC01), &uarch);
  EXPECT_EQ(uarch, static_cast<uint32_t>(cpuinfo_uarch_saphira));
}

#endif  // ARM64

TEST(CpuidUarch, ArmCortexA76) {
  uint32_t uarch = cpuinfo_uarch_unknown;
  decodeMIDR(MakeMIDR('A', 0xD0B), &uarch);
  EXPECT_EQ(uarch, static_cast<uint32_t>(cpuinfo_uarch_cortex_a76));
}

TEST(CpuidUarch, ArmCortexA78CMapsToA78) {
  uint32_t uarch = cpuinfo_uarch_unknown;
  decodeMIDR(MakeMIDR('A', 0xD4B), &uarch);
  EXPECT_EQ(uarch, static_cast<uint32_t>(cpuinfo_uarch_cortex_a78));
}

TEST(CpuidUarch, UnknownImplementerReturnsUnknown) {
  // Implementer 'Z' (0x5A) is not in the table
  uint32_t uarch = cpuinfo_uarch_unknown;
  decodeMIDR(MakeMIDR('Z', 0x001), &uarch);
  EXPECT_EQ(uarch, static_cast<uint32_t>(cpuinfo_uarch_unknown));
}

}  // namespace test
}  // namespace onnxruntime

#endif  // defined(CPUIDINFO_ARCH_ARM)
