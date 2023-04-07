/*++

Copyright (c) Microsoft Corporation.  All rights reserved.

Licensed under the MIT License.

Module Name:

    cpuid_uarch.h

Abstract:
    ARM Processor microarchitecture descriptions and utilities

    Processors with different microarchitectures often have different
    instruction performance characteristics, and may have dramatically
    different pipeline organization.

    We leverage Pytorch CPUinfo (https://github.com/pytorch/cpuinfo) for
    hardware feature detection. Unfortunately CPUinfo package only works
    on a few platforms but we need to support more. This ugly hack is
    here to bridge the gap.

--*/

#pragma once

#include <cstdint>

#include "core/common/cpuid_arch_definition.h"

namespace onnxruntime {

enum CPUIDINFOuarch {
  /** Microarchitecture is unknown, or the library failed to get information about the microarchitecture from OS */
  cpuinfo_uarch_unknown = 0,

  /** Intel/Marvell XScale series. */
  cpuinfo_uarch_xscale = 0x00100600,

  /** ARM7 series. */
  cpuinfo_uarch_arm7 = 0x00300100,
  /** ARM9 series. */
  cpuinfo_uarch_arm9 = 0x00300101,
  /** ARM 1136, ARM 1156, ARM 1176, or ARM 11MPCore. */
  cpuinfo_uarch_arm11 = 0x00300102,

  /** ARM Cortex-A5. */
  cpuinfo_uarch_cortex_a5 = 0x00300205,
  /** ARM Cortex-A7. */
  cpuinfo_uarch_cortex_a7 = 0x00300207,
  /** ARM Cortex-A8. */
  cpuinfo_uarch_cortex_a8 = 0x00300208,
  /** ARM Cortex-A9. */
  cpuinfo_uarch_cortex_a9 = 0x00300209,
  /** ARM Cortex-A12. */
  cpuinfo_uarch_cortex_a12 = 0x00300212,
  /** ARM Cortex-A15. */
  cpuinfo_uarch_cortex_a15 = 0x00300215,
  /** ARM Cortex-A17. */
  cpuinfo_uarch_cortex_a17 = 0x00300217,

  /** ARM Cortex-A32. */
  cpuinfo_uarch_cortex_a32 = 0x00300332,
  /** ARM Cortex-A35. */
  cpuinfo_uarch_cortex_a35 = 0x00300335,
  /** ARM Cortex-A53. */
  cpuinfo_uarch_cortex_a53 = 0x00300353,
  /** ARM Cortex-A55 revision 0 (restricted dual-issue capabilities compared to revision 1+). */
  cpuinfo_uarch_cortex_a55r0 = 0x00300354,
  /** ARM Cortex-A55. */
  cpuinfo_uarch_cortex_a55 = 0x00300355,
  /** ARM Cortex-A57. */
  cpuinfo_uarch_cortex_a57 = 0x00300357,
  /** ARM Cortex-A65. */
  cpuinfo_uarch_cortex_a65 = 0x00300365,
  /** ARM Cortex-A72. */
  cpuinfo_uarch_cortex_a72 = 0x00300372,
  /** ARM Cortex-A73. */
  cpuinfo_uarch_cortex_a73 = 0x00300373,
  /** ARM Cortex-A75. */
  cpuinfo_uarch_cortex_a75 = 0x00300375,
  /** ARM Cortex-A76. */
  cpuinfo_uarch_cortex_a76 = 0x00300376,
  /** ARM Cortex-A77. */
  cpuinfo_uarch_cortex_a77 = 0x00300377,
  /** ARM Cortex-A78. */
  cpuinfo_uarch_cortex_a78 = 0x00300378,

  /** ARM Neoverse N1. */
  cpuinfo_uarch_neoverse_n1 = 0x00300400,
  /** ARM Neoverse E1. */
  cpuinfo_uarch_neoverse_e1 = 0x00300401,

  /** ARM Cortex-X1. */
  cpuinfo_uarch_cortex_x1 = 0x00300500,

  /** Qualcomm Scorpion. */
  cpuinfo_uarch_scorpion = 0x00400100,
  /** Qualcomm Krait. */
  cpuinfo_uarch_krait = 0x00400101,
  /** Qualcomm Kryo. */
  cpuinfo_uarch_kryo = 0x00400102,
  /** Qualcomm Falkor. */
  cpuinfo_uarch_falkor = 0x00400103,
  /** Qualcomm Saphira. */
  cpuinfo_uarch_saphira = 0x00400104,

  /** Nvidia Denver. */
  cpuinfo_uarch_denver = 0x00500100,
  /** Nvidia Denver 2. */
  cpuinfo_uarch_denver2 = 0x00500101,
  /** Nvidia Carmel. */
  cpuinfo_uarch_carmel = 0x00500102,

  /** Samsung Exynos M1 (Exynos 8890 big cores). */
  cpuinfo_uarch_exynos_m1 = 0x00600100,
  /** Samsung Exynos M2 (Exynos 8895 big cores). */
  cpuinfo_uarch_exynos_m2 = 0x00600101,
  /** Samsung Exynos M3 (Exynos 9810 big cores). */
  cpuinfo_uarch_exynos_m3 = 0x00600102,
  /** Samsung Exynos M4 (Exynos 9820 big cores). */
  cpuinfo_uarch_exynos_m4 = 0x00600103,
  /** Samsung Exynos M5 (Exynos 9830 big cores). */
  cpuinfo_uarch_exynos_m5 = 0x00600104,

  /* Deprecated synonym for Cortex-A76 */
  cpuinfo_uarch_cortex_a76ae = 0x00300376,
  /* Deprecated names for Exynos. */
  cpuinfo_uarch_mongoose_m1 = 0x00600100,
  cpuinfo_uarch_mongoose_m2 = 0x00600101,
  cpuinfo_uarch_meerkat_m3 = 0x00600102,
  cpuinfo_uarch_meerkat_m4 = 0x00600103,

  /** Apple A6 and A6X processors. */
  cpuinfo_uarch_swift = 0x00700100,
  /** Apple A7 processor. */
  cpuinfo_uarch_cyclone = 0x00700101,
  /** Apple A8 and A8X processor. */
  cpuinfo_uarch_typhoon = 0x00700102,
  /** Apple A9 and A9X processor. */
  cpuinfo_uarch_twister = 0x00700103,
  /** Apple A10 and A10X processor. */
  cpuinfo_uarch_hurricane = 0x00700104,
  /** Apple A11 processor (big cores). */
  cpuinfo_uarch_monsoon = 0x00700105,
  /** Apple A11 processor (little cores). */
  cpuinfo_uarch_mistral = 0x00700106,
  /** Apple A12 processor (big cores). */
  cpuinfo_uarch_vortex = 0x00700107,
  /** Apple A12 processor (little cores). */
  cpuinfo_uarch_tempest = 0x00700108,
  /** Apple A13 processor (big cores). */
  cpuinfo_uarch_lightning = 0x00700109,
  /** Apple A13 processor (little cores). */
  cpuinfo_uarch_thunder = 0x0070010A,
  /** Apple M1 processor (big cores). */
  cpuinfo_uarch_firestorm = 0x0070010B,
  /** Apple M1 processor (little cores). */
  cpuinfo_uarch_icestorm = 0x0070010C,

  /** Cavium ThunderX. */
  cpuinfo_uarch_thunderx = 0x00800100,
  /** Cavium ThunderX2 (originally Broadcom Vulkan). */
  cpuinfo_uarch_thunderx2 = 0x00800200,

  /** Marvell PJ4. */
  cpuinfo_uarch_pj4 = 0x00900100,

  /** Broadcom Brahma B15. */
  cpuinfo_uarch_brahma_b15 = 0x00A00100,
  /** Broadcom Brahma B53. */
  cpuinfo_uarch_brahma_b53 = 0x00A00101,

  /** Applied Micro X-Gene. */
  cpuinfo_uarch_xgene = 0x00B00100,

  /* Hygon Dhyana (a modification of AMD Zen for Chinese market). */
  cpuinfo_uarch_dhyana = 0x01000100,

  /** HiSilicon TaiShan v110 (Huawei Kunpeng 920 series processors). */
  cpuinfo_uarch_taishan_v110 = 0x00C00100,
};

#if defined(CPUIDINFO_ARCH_ARM)

void decodeMIDR(uint32_t midr, uint32_t uarch[1]);

#endif  // arm or arm64

}  // namespace onnxruntime
