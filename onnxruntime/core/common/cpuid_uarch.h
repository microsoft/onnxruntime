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

#define CPUINFO_ARM_MIDR_IMPLEMENTER_MASK UINT32_C(0xFF000000)
#define CPUINFO_ARM_MIDR_VARIANT_MASK UINT32_C(0x00F00000)
#define CPUINFO_ARM_MIDR_ARCHITECTURE_MASK UINT32_C(0x000F0000)
#define CPUINFO_ARM_MIDR_PART_MASK UINT32_C(0x0000FFF0)
#define CPUINFO_ARM_MIDR_REVISION_MASK UINT32_C(0x0000000F)

#define CPUINFO_ARM_MIDR_IMPLEMENTER_OFFSET 24
#define CPUINFO_ARM_MIDR_VARIANT_OFFSET 20
#define CPUINFO_ARM_MIDR_ARCHITECTURE_OFFSET 16
#define CPUINFO_ARM_MIDR_PART_OFFSET 4
#define CPUINFO_ARM_MIDR_REVISION_OFFSET 0

inline static uint32_t midr_get_implementer(uint32_t midr) {
  return (midr & CPUINFO_ARM_MIDR_IMPLEMENTER_MASK) >> CPUINFO_ARM_MIDR_IMPLEMENTER_OFFSET;
}

inline static uint32_t midr_get_part(uint32_t midr) {
  return (midr & CPUINFO_ARM_MIDR_PART_MASK) >> CPUINFO_ARM_MIDR_PART_OFFSET;
}

inline static uint32_t midr_get_variant(uint32_t midr) {
  return (midr & CPUINFO_ARM_MIDR_VARIANT_MASK) >> CPUINFO_ARM_MIDR_VARIANT_OFFSET;
}

static void decodeMIDR(
    uint32_t midr,
    uint32_t uarch[1]) {
  switch (midr_get_implementer(midr)) {
    case 'A':
      switch (midr_get_part(midr)) {
          //#if defined(_M_ARM) || defined(__arm__)
        case 0xC05:
          *uarch = cpuinfo_uarch_cortex_a5;
          break;
        case 0xC07:
          *uarch = cpuinfo_uarch_cortex_a7;
          break;
        case 0xC08:
          *uarch = cpuinfo_uarch_cortex_a8;
          break;
        case 0xC09:
          *uarch = cpuinfo_uarch_cortex_a9;
          break;
        case 0xC0C:
          *uarch = cpuinfo_uarch_cortex_a12;
          break;
        case 0xC0E:
          *uarch = cpuinfo_uarch_cortex_a17;
          break;
        case 0xC0D:
          /*
					 * Rockchip RK3288 only.
					 * Core information is ambiguous: some sources specify Cortex-A12, others - Cortex-A17.
					 * Assume it is Cortex-A12.
					 */
          *uarch = cpuinfo_uarch_cortex_a12;
          break;
        case 0xC0F:
          *uarch = cpuinfo_uarch_cortex_a15;
          break;
          //#endif /* ARM */
        case 0xD01:
          *uarch = cpuinfo_uarch_cortex_a32;
          break;
        case 0xD03:
          *uarch = cpuinfo_uarch_cortex_a53;
          break;
        case 0xD04:
          *uarch = cpuinfo_uarch_cortex_a35;
          break;
        case 0xD05:
          // Note: use Variant, not Revision, field
          *uarch = (midr & CPUINFO_ARM_MIDR_VARIANT_MASK) == 0 ? cpuinfo_uarch_cortex_a55r0 : cpuinfo_uarch_cortex_a55;
          break;
        case 0xD06:
          *uarch = cpuinfo_uarch_cortex_a65;
          break;
        case 0xD07:
          *uarch = cpuinfo_uarch_cortex_a57;
          break;
        case 0xD08:
          *uarch = cpuinfo_uarch_cortex_a72;
          break;
        case 0xD09:
          *uarch = cpuinfo_uarch_cortex_a73;
          break;
        case 0xD0A:
          *uarch = cpuinfo_uarch_cortex_a75;
          break;
        case 0xD0B:
          *uarch = cpuinfo_uarch_cortex_a76;
          break;
          //#if (defined(_M_ARM64) || defined(__aarch64__)) && !defined(__ANDROID__)
        case 0xD0C:
          *uarch = cpuinfo_uarch_neoverse_n1;
          break;
          //#endif /* ARM64 && !defined(__ANDROID__) */
        case 0xD0D:
          *uarch = cpuinfo_uarch_cortex_a77;
          break;
        case 0xD0E: /* Cortex-A76AE */
          *uarch = cpuinfo_uarch_cortex_a76;
          break;
        case 0xD41: /* Cortex-A78 */
          *uarch = cpuinfo_uarch_cortex_a78;
          break;
        case 0xD44: /* Cortex-X1 */
          *uarch = cpuinfo_uarch_cortex_x1;
          break;
          //#if (defined(_M_ARM64) || defined(__aarch64__)) && !defined(__ANDROID__)
        case 0xD4A:
          *uarch = cpuinfo_uarch_neoverse_e1;
          break;
          //#endif /* ARM64 && !defined(__ANDROID__) */
        default:
          switch (midr_get_part(midr) >> 8) {
              //#if defined(_M_ARM) || defined(__arm__)
            case 7:
              *uarch = cpuinfo_uarch_arm7;
              break;
            case 9:
              *uarch = cpuinfo_uarch_arm9;
              break;
            case 11:
              *uarch = cpuinfo_uarch_arm11;
              break;
              //#endif /* ARM */
            default:
              LOGS_DEFAULT(WARNING) << "unknown ARM CPU part 0x" << std::hex << midr_get_part(midr) << " ignored";
          }
      }
      break;
    case 'B':
      switch (midr_get_part(midr)) {
        case 0x00F:
          *uarch = cpuinfo_uarch_brahma_b15;
          break;
        case 0x100:
          *uarch = cpuinfo_uarch_brahma_b53;
          break;
          //#if (defined(_M_ARM64) || defined(__aarch64__)) && !defined(__ANDROID__)
        case 0x516:
          /* Broadcom Vulkan was sold to Cavium before it reached the market, so we identify it as Cavium ThunderX2 */
          *uarch = cpuinfo_uarch_thunderx2;
          break;
          //#endif
        default:
          LOGS_DEFAULT(WARNING) << "unknown Broadcom CPU part 0x" << std::hex << midr_get_part(midr) << " ignored";
      }
      break;
      //#if (defined(_M_ARM64) || defined(__aarch64__)) && !defined(__ANDROID__)
    case 'C':
      switch (midr_get_part(midr)) {
        case 0x0A0: /* ThunderX */
        case 0x0A1: /* ThunderX 88XX */
        case 0x0A2: /* ThunderX 81XX */
        case 0x0A3: /* ThunderX 83XX */
          *uarch = cpuinfo_uarch_thunderx;
          break;
        case 0x0AF: /* ThunderX2 99XX */
          *uarch = cpuinfo_uarch_thunderx2;
          break;
        default:
          LOGS_DEFAULT(WARNING) << "unknown Cavium CPU part 0x" << std::hex << midr_get_part(midr) << " ignored";
      }
      break;
      //#endif
    case 'H':
      switch (midr_get_part(midr)) {
          //#if (defined(_M_ARM64) || defined(__aarch64__)) && !defined(__ANDROID__)
        case 0xD01: /* Kunpeng 920 series */
          *uarch = cpuinfo_uarch_taishan_v110;
          break;
          //#endif
        case 0xD40: /* Kirin 980 Big/Medium cores -> Cortex-A76 */
          *uarch = cpuinfo_uarch_cortex_a76;
          break;
        default:
          LOGS_DEFAULT(WARNING) << "unknown Huawei CPU part 0x" << std::hex << midr_get_part(midr) << " ignored";
      }
      break;
      //#if defined(_M_ARM) || defined(__arm__)
    case 'i':
      switch (midr_get_part(midr) >> 8) {
        case 2: /* PXA 210/25X/26X */
        case 4: /* PXA 27X */
        case 6: /* PXA 3XX */
          *uarch = cpuinfo_uarch_xscale;
          break;
        default:
          LOGS_DEFAULT(WARNING) << "unknown Intel CPU part 0x" << std::hex << midr_get_part(midr) << " ignored";
      }
      break;
      //#endif /* ARM */
    case 'N':
      switch (midr_get_part(midr)) {
        case 0x000:
          *uarch = cpuinfo_uarch_denver;
          break;
        case 0x003:
          *uarch = cpuinfo_uarch_denver2;
          break;
        case 0x004:
          *uarch = cpuinfo_uarch_carmel;
          break;
        default:
          LOGS_DEFAULT(WARNING) << "unknown Nvidia CPU part 0x" << std::hex << midr_get_part(midr) << " ignored";
      }
      break;
#if !defined(__ANDROID__)
    case 'P':
      switch (midr_get_part(midr)) {
        case 0x000:
          *uarch = cpuinfo_uarch_xgene;
          break;
        default:
          LOGS_DEFAULT(WARNING) << "unknown Applied Micro CPU part 0x" << std::hex << midr_get_part(midr) << " ignored";
      }
      break;
#endif
    case 'Q':
      switch (midr_get_part(midr)) {
          // #if defined(_M_ARM) || defined(__arm__)
        case 0x00F:
          /* Mostly Scorpions, but some Cortex A5 may report this value as well */
          //if (has_vfpv4) {
          //  /* Unlike Scorpion, Cortex-A5 comes with VFPv4 */
          //  *vendor = cpuinfo_vendor_arm;
          //  *uarch = cpuinfo_uarch_cortex_a5;
          //} else {
          *uarch = cpuinfo_uarch_scorpion;
          //          }
          break;
        case 0x02D: /* Dual-core Scorpions */
          *uarch = cpuinfo_uarch_scorpion;
          break;
        case 0x04D:
          /*
					 * Dual-core Krait:
					 * - r1p0 -> Krait 200
					 * - r1p4 -> Krait 200
					 * - r2p0 -> Krait 300
					 */
        case 0x06F:
          /*
					 * Quad-core Krait:
					 * - r0p1 -> Krait 200
					 * - r0p2 -> Krait 200
					 * - r1p0 -> Krait 300
					 * - r2p0 -> Krait 400 (Snapdragon 800 MSMxxxx)
					 * - r2p1 -> Krait 400 (Snapdragon 801 MSMxxxxPRO)
					 * - r3p1 -> Krait 450
					 */
          *uarch = cpuinfo_uarch_krait;
          break;
          //#endif              /* ARM */
        case 0x201: /* Qualcomm Snapdragon 821: Low-power Kryo "Silver" */
        case 0x205: /* Qualcomm Snapdragon 820 & 821: High-performance Kryo "Gold" */
        case 0x211: /* Qualcomm Snapdragon 820: Low-power Kryo "Silver" */
          *uarch = cpuinfo_uarch_kryo;
          break;
        case 0x800: /* High-performance Kryo 260 (r10p2) / Kryo 280 (r10p1) "Gold" -> Cortex-A73 */
          *uarch = cpuinfo_uarch_cortex_a73;
          break;
        case 0x801: /* Low-power Kryo 260 / 280 "Silver" -> Cortex-A53 */
          *uarch = cpuinfo_uarch_cortex_a53;
          break;
        case 0x802: /* High-performance Kryo 385 "Gold" -> Cortex-A75 */
          *uarch = cpuinfo_uarch_cortex_a75;
          break;
        case 0x803: /* Low-power Kryo 385 "Silver" -> Cortex-A55r0 */
          *uarch = cpuinfo_uarch_cortex_a55r0;
          break;
        case 0x804: /* High-performance Kryo 485 "Gold" / "Gold Prime" -> Cortex-A76 */
          *uarch = cpuinfo_uarch_cortex_a76;
          break;
        case 0x805: /* Low-performance Kryo 485 "Silver" -> Cortex-A55 */
          *uarch = cpuinfo_uarch_cortex_a55;
          break;
          //#if (defined(_M_ARM64) || defined(__aarch64__)) && !defined(__ANDROID__)
        case 0xC00:
          *uarch = cpuinfo_uarch_falkor;
          break;
        case 0xC01:
          *uarch = cpuinfo_uarch_saphira;
          break;
          //#endif /* ARM64 && !defined(__ANDROID__) */
        default:
          LOGS_DEFAULT(WARNING) << "unknown Qualcomm CPU part 0x" << std::hex << midr_get_part(midr) << " ignored";
      }
      break;
    case 'S':
      switch (midr & (CPUINFO_ARM_MIDR_VARIANT_MASK | CPUINFO_ARM_MIDR_PART_MASK)) {
        case 0x00100010:
          /*
					 * Exynos 8890 MIDR = 0x531F0011, assume Exynos M1 has:
					 * - CPU variant 0x1
					 * - CPU part 0x001
					 */
          *uarch = cpuinfo_uarch_exynos_m1;
          break;
        case 0x00400010:
          /*
					 * Exynos 8895 MIDR = 0x534F0010, assume Exynos M2 has:
					 * - CPU variant 0x4
					 * - CPU part 0x001
					 */
          *uarch = cpuinfo_uarch_exynos_m2;
          break;
        case 0x00100020:
          /*
					 * Exynos 9810 MIDR = 0x531F0020, assume Exynos M3 has:
					 * - CPU variant 0x1
					 * - CPU part 0x002
					 */
          *uarch = cpuinfo_uarch_exynos_m3;
          break;
        case 0x00100030:
          /*
					 * Exynos 9820 MIDR = 0x531F0030, assume Exynos M4 has:
					 * - CPU variant 0x1
					 * - CPU part 0x003
					 */
          *uarch = cpuinfo_uarch_exynos_m4;
          break;
        case 0x00100040:
          /*
					 * Exynos 9820 MIDR = 0x531F0040, assume Exynos M5 has:
					 * - CPU variant 0x1
					 * - CPU part 0x004
					 */
          *uarch = cpuinfo_uarch_exynos_m5;
          break;
        default:
          LOGS_DEFAULT(WARNING) << "unknown Samsung CPU variant 0x"
                                << std::hex << midr_get_variant(midr) << " part 0x" << std::hex << midr_get_part(midr) << " ignored";
      }
      break;
      //#if defined(_M_ARM) || defined(__arm__)
    case 'V':
      switch (midr_get_part(midr)) {
        case 0x581: /* PJ4 / PJ4B */
        case 0x584: /* PJ4B-MP / PJ4C */
          *uarch = cpuinfo_uarch_pj4;
          break;
        default:
          LOGS_DEFAULT(WARNING) << "unknown Marvell CPU part 0x" << std::hex << midr_get_part(midr) << " ignored";
      }
      break;
      //#endif /* ARM */
    default:
      LOGS_DEFAULT(WARNING) << "unknown CPU uarch from MIDR value: 0x" << std::hex << midr;
  }
}

#endif  // arm or arm64
