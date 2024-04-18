//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#pragma once
#include <map>
#include <thread>
#include <vector>
#include "bestla.h"
#include "xbyak/xbyak_util.h"
#include "bestla_utils.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <sched.h>
#endif

#define FIXED_CACHE_SIZE ((1 << 20) - (128 << 10))
#define FIXED_CACHE 1

namespace bestla {

namespace device {

struct X64_ISA {
  int64_t MMX : 1;                  // 0
  int64_t SSE : 1;                  // 1
  int64_t SSE2 : 1;                 // 2
  int64_t SSE3 : 1;                 // 3
  int64_t SSSE3 : 1;                // 4
  int64_t SSE41 : 1;                // 5
  int64_t SSE42 : 1;                // 6
  int64_t AVX : 1;                  // 7
  int64_t F16C : 1;                 // 8
  int64_t FMA : 1;                  // 9
  int64_t AVX2 : 1;                 // 10
  int64_t AVX_VNNI : 1;             // 11
  int64_t AVX_VNNI_INT8 : 1;        // 12
  int64_t AVX_NE_CONVERT : 1;       // 13
  int64_t AVX_IFMA : 1;             // 14
  int64_t AVX512F : 1;              // 15
  int64_t AVX512BW : 1;             // 16
  int64_t AVX512CD : 1;             // 17
  int64_t AVX512DQ : 1;             // 18
  int64_t AVX512ER : 1;             // 19
  int64_t AVX512IFMA52 : 1;         // 20
  int64_t AVX512PF : 1;             // 21
  int64_t AVX512VL : 1;             // 22
  int64_t AVX512VPOPCNTDQ : 1;      // 23
  int64_t AVX512_4FMAPS : 1;        // 24
  int64_t AVX512_4VNNIW : 1;        // 25
  int64_t AVX512_BF16 : 1;          // 26
  int64_t AVX512_BITALG : 1;        // 27
  int64_t AVX512_VBMI : 1;          // 28
  int64_t AVX512_VBMI2 : 1;         // 29
  int64_t AVX512_VNNI : 1;          // 30
  int64_t AVX512_VP2INTERSECT : 1;  // 31
  int64_t AVX512_FP16 : 1;          // 32
  int64_t AMX_TILE : 1;             // 33
  int64_t AMX_BF16 : 1;             // 34
  int64_t AMX_INT8 : 1;             // 35
  int64_t AMX_FP16 : 1;             // 36
  int64_t AMX_COMPLEX : 1;          // 37
  int64_t reserved : (64 - 38);
};

class AVX2_Default {
 public:
  static constexpr bool MMX = 1;
  static constexpr bool SSE = 1;
  static constexpr bool SSE2 = 1;
  static constexpr bool SSE3 = 1;
  static constexpr bool SSSE3 = 1;
  static constexpr bool SSE41 = 1;
  static constexpr bool SSE42 = 1;
  static constexpr bool AVX = 1;
  static constexpr bool F16C = 1;
  static constexpr bool FMA = 1;
  static constexpr bool AVX2 = 1;
  static constexpr bool AVX_VNNI = 0;
  static constexpr bool AVX_VNNI_INT8 = 0;
  static constexpr bool AVX_NE_CONVERT = 0;
  static constexpr bool AVX_IFMA = 0;
  static constexpr bool AVX512F = 0;
  static constexpr bool AVX512BW = 0;
  static constexpr bool AVX512CD = 0;
  static constexpr bool AVX512DQ = 0;
  static constexpr bool AVX512ER = 0;
  static constexpr bool AVX512IFMA52 = 0;
  static constexpr bool AVX512PF = 0;
  static constexpr bool AVX512VL = 0;
  static constexpr bool AVX512VPOPCNTDQ = 0;
  static constexpr bool AVX512_4FMAPS = 0;
  static constexpr bool AVX512_4VNNIW = 0;
  static constexpr bool AVX512_BF16 = 0;
  static constexpr bool AVX512_BITALG = 0;
  static constexpr bool AVX512_VBMI = 0;
  static constexpr bool AVX512_VBMI2 = 0;
  static constexpr bool AVX512_VNNI = 0;
  static constexpr bool AVX512_VP2INTERSECT = 0;
  static constexpr bool AVX512_FP16 = 0;
  static constexpr bool AMX_TILE = 0;
  static constexpr bool AMX_BF16 = 0;
  static constexpr bool AMX_INT8 = 0;
  static constexpr bool AMX_FP16 = 0;
  static constexpr bool AMX_COMPLEX = 0;
};

class AVX512_VNNI_Default {
 public:
  static constexpr bool MMX = 1;
  static constexpr bool SSE = 1;
  static constexpr bool SSE2 = 1;
  static constexpr bool SSE3 = 1;
  static constexpr bool SSSE3 = 1;
  static constexpr bool SSE41 = 1;
  static constexpr bool SSE42 = 1;
  static constexpr bool AVX = 1;
  static constexpr bool F16C = 1;
  static constexpr bool FMA = 1;
  static constexpr bool AVX2 = 1;
  static constexpr bool AVX_VNNI = 0;
  static constexpr bool AVX_VNNI_INT8 = 0;
  static constexpr bool AVX_NE_CONVERT = 0;
  static constexpr bool AVX_IFMA = 0;
  static constexpr bool AVX512F = 1;
  static constexpr bool AVX512BW = 1;
  static constexpr bool AVX512CD = 1;
  static constexpr bool AVX512DQ = 1;
  static constexpr bool AVX512ER = 0;
  static constexpr bool AVX512IFMA52 = 0;
  static constexpr bool AVX512PF = 0;
  static constexpr bool AVX512VL = 1;
  static constexpr bool AVX512VPOPCNTDQ = 0;
  static constexpr bool AVX512_4FMAPS = 0;
  static constexpr bool AVX512_4VNNIW = 0;
  static constexpr bool AVX512_BF16 = 0;
  static constexpr bool AVX512_BITALG = 0;
  static constexpr bool AVX512_VBMI = 0;
  static constexpr bool AVX512_VBMI2 = 0;
  static constexpr bool AVX512_VNNI = 1;
  static constexpr bool AVX512_VP2INTERSECT = 0;
  static constexpr bool AVX512_FP16 = 0;
  static constexpr bool AMX_TILE = 0;
  static constexpr bool AMX_BF16 = 0;
  static constexpr bool AMX_INT8 = 0;
  static constexpr bool AMX_FP16 = 0;
  static constexpr bool AMX_COMPLEX = 0;
};

class SapphireRapids {
 public:
  static constexpr bool MMX = 1;
  static constexpr bool SSE = 1;
  static constexpr bool SSE2 = 1;
  static constexpr bool SSE3 = 1;
  static constexpr bool SSSE3 = 1;
  static constexpr bool SSE41 = 1;
  static constexpr bool SSE42 = 1;
  static constexpr bool AVX = 1;
  static constexpr bool F16C = 1;
  static constexpr bool FMA = 1;
  static constexpr bool AVX2 = 1;
  static constexpr bool AVX_VNNI = 0;
  static constexpr bool AVX_VNNI_INT8 = 0;
  static constexpr bool AVX_NE_CONVERT = 0;
  static constexpr bool AVX_IFMA = 0;
  static constexpr bool AVX512F = 1;
  static constexpr bool AVX512BW = 1;
  static constexpr bool AVX512CD = 1;
  static constexpr bool AVX512DQ = 1;
  static constexpr bool AVX512ER = 0;
  static constexpr bool AVX512IFMA52 = 0;
  static constexpr bool AVX512PF = 0;
  static constexpr bool AVX512VL = 1;
  static constexpr bool AVX512VPOPCNTDQ = 0;
  static constexpr bool AVX512_4FMAPS = 0;
  static constexpr bool AVX512_4VNNIW = 0;
  static constexpr bool AVX512_BF16 = 0;
  static constexpr bool AVX512_BITALG = 0;
  static constexpr bool AVX512_VBMI = 0;
  static constexpr bool AVX512_VBMI2 = 0;
  static constexpr bool AVX512_VNNI = 1;
  static constexpr bool AVX512_VP2INTERSECT = 0;
  static constexpr bool AVX512_FP16 = 0;
  static constexpr bool AMX_TILE = 1;
  static constexpr bool AMX_BF16 = 1;
  static constexpr bool AMX_INT8 = 1;
  static constexpr bool AMX_FP16 = 0;
  static constexpr bool AMX_COMPLEX = 0;
};

template <BTLA_ISA ISA_T>
class isa_base {
 public:
  static bool constexpr avx = ISA_T >= BTLA_ISA::AVX;
  static bool constexpr avx2 = ISA_T >= BTLA_ISA::AVX2;
  static bool constexpr avx512f = ISA_T >= BTLA_ISA::AVX512F;
  static bool constexpr avx512_vnni = ISA_T >= BTLA_ISA::AVX512_VNNI;
  static bool constexpr avx512_fp16 = ISA_T >= BTLA_ISA::AVX512_FP16;
  static bool constexpr amx_bf16 = ISA_T >= BTLA_ISA::AMX_BF16;
  static bool constexpr amx_int8 = ISA_T >= BTLA_ISA::AMX_INT8;
};

class CpuDevice {
 public:
  inline int getThreads() { return numthreads; }
  inline int getCores() { return numcores; }
  inline uint32_t getL2CacheSize() { return L2Cache; }
  inline uint32_t getL1CacheSize() { return L1Cache; }
  inline uint32_t getL2CacheSize_E() { return E_L2Cache; }
  inline uint32_t getL1CacheSize_E() { return E_L1Cache; }
  inline bool AVX() { return mHasAVX; }
  inline bool AVX2() { return mHasAVX2; }
  inline bool AVX_VNNI() { return mHasAVX_VNNI; }
  inline bool AVX512F() { return mHasAVX512F; }
  inline bool AVX512_VNNI() { return mHasAVX512_VNNI; }
  inline bool AMX_INT8() { return mHasAMX_INT8; }
  inline bool AMX_BF16() { return mHasAMX_BF16; }
  inline bool AVX512_BF16() { return mHasAVX512_BF16; }
  inline bool AVX512_FP16() { return mHasAVX512_FP16; }
  inline float getPE() { return (P_core.size() * P_power) / (E_core.size() * E_power); }
  inline size_t getPcoreNum() { return P_core.size(); }
  inline size_t getEcoreNum() { return E_core.size(); }
  inline size_t getSMTcoreNum() { return SMT_core.size(); }
  inline int* getPCores() { return P_core.data(); }
  inline int* getECores() { return E_core.data(); }
  inline int* getSMTCores() { return SMT_core.data(); }
#define ADD_FLAG(isa) mHas##isa = _cpu.has(_cpu.t##isa)
  CpuDevice() {
    static Xbyak::util::Cpu _cpu;
    ADD_FLAG(AVX);
    ADD_FLAG(AVX2);
    ADD_FLAG(AVX512F);
    ADD_FLAG(AVX512_VNNI);
    ADD_FLAG(AVX_VNNI);
    ADD_FLAG(AMX_BF16);
    ADD_FLAG(AMX_INT8);
    ADD_FLAG(AVX512_BF16);
    ADD_FLAG(AVX512_FP16);
    numcores = _cpu.getNumCores(Xbyak::util::IntelCpuTopologyLevel::CoreLevel);
    if (mHasAMX_BF16 || mHasAMX_INT8) {
      utils::request_perm_xtile_data();
    }
    static bool p = false;
    {
      uint32_t tmp[4];
      _cpu.getCpuid(7, tmp);
      if (tmp[3] & (1U << 15)) mHybrid = true;
      if (p) printf("!!!Hybrid:%d\t%x\t%x\t%x\t%x!!!\n", mHybrid, tmp[0], tmp[1], tmp[2], tmp[3]);
    }
    if (mHybrid) {
      int total_cores = numcores * _cpu.getNumCores(Xbyak::util::IntelCpuTopologyLevel::SmtLevel);
      std::vector<int> core_type(total_cores), core_id(total_cores), L1(total_cores), L2(total_cores);
      std::map<int, int> core_id_count;

      {
        // classify E-core / LPE-core and  P-core / smt
        std::vector<std::thread> thdset(total_cores);
        for (size_t i = 0; i < total_cores; i++) {
          thdset[i] = std::thread(
              [&](int tidx) {
                core_bond(tidx);
                Xbyak::util::Cpu cpu;
                L1[tidx] = cpu.getDataCacheSize(0);
                L2[tidx] = cpu.getDataCacheSize(1);
                if (isEcore(cpu))
                  core_type[tidx] = 1;
                else
                  core_type[tidx] = 2;
                core_id[tidx] = getCoreId(cpu);
              },
              int(i));
        }
        for (size_t i = 0; i < total_cores; i++) {
          thdset[i].join();
          core_id_count[core_id[i]] = core_id_count[core_id[i]] + 1;
        }
        if (p) {
          for (int i = 0; i < total_cores; i++) printf("%d %d\n", core_type[i], core_id[i]);
          for (auto& kv : core_id_count) printf("%d,%d\n", kv.first, kv.second);
        }
        for (int i = 0; i < total_cores; i++) {
          if (core_type[i] == 2) {
            if (core_id_count[core_id[i]] > 0) {
              P_core.push_back(i);
              core_id_count[core_id[i]] = 0;
            } else {
              SMT_core.push_back(i);
            }
          } else {
            if (core_id_count[core_id[i]] == 4) E_core.push_back(i);
          }
        }
        if (p) {
          printf("Pcore:");
          for (auto& i : P_core) printf("%d,", i);
          printf("\nEcore:");
          for (auto& i : E_core) printf("%d,", i);
          printf("\nsmt:");
          for (auto& i : SMT_core) printf("%d,", i);
          printf("\n");
        }
        E_L1Cache = L1[E_core[0]];
        E_L2Cache = L2[E_core[0]] / 4;
        L1Cache = E_L1Cache > L1[P_core[0]] / 2 ? L1[P_core[0]] / 2 : E_L1Cache;
        L2Cache = E_L2Cache > L2[P_core[0]] / 2 ? L2[P_core[0]] / 2 : E_L2Cache;
      }
      numcores = P_core.size() + E_core.size();
      numthreads = P_core.size() * 2 + E_core.size();
    } else {
      L1Cache = _cpu.getDataCacheSize(0);
      L2Cache = _cpu.getDataCacheSize(1);
      numthreads = numcores;
    }
#if FIXED_CACHE
    L2Cache = L2Cache >= FIXED_CACHE_SIZE ? FIXED_CACHE_SIZE : L2Cache;
    E_L2Cache = E_L2Cache >= FIXED_CACHE_SIZE ? FIXED_CACHE_SIZE : E_L2Cache;
#endif
  }

  static CpuDevice* getInstance() {
    static CpuDevice instance;
    return &instance;
  }

  void print() {
    printf(
        "AVX:%d AVX2:%d AVX512F:%d AVX_VNNI:%d AVX512_VNNI:%d AMX_INT8:%d AMX_BF16:%d AVX512_BF16:%d AVX512_FP16:%d\n",
        mHasAVX, mHasAVX2, mHasAVX512F, mHasAVX_VNNI, mHasAVX512_VNNI, mHasAMX_INT8, mHasAMX_BF16, mHasAVX512_BF16,
        mHasAVX512_FP16);
  }
#undef ADD_FLAG

  static bool isEcore() {
    Xbyak::util::Cpu cpu;
    uint32_t tmp[4];
    cpu.getCpuid(0x1A, tmp);
    int core_type = (tmp[0] >> 24) & ((1u << 7) - 1);  // cpu.extractBit(a[0], 24, 31);
    switch (core_type) {
      case 32:
        // printf("Atom\n");
        return true;  // E-core or LPE-core
        break;
      case 64:
        // printf("Core\n");
        return false;  // P-core
        break;
      default:
        // printf("No hyper\n");
        return false;
        break;
    }
    return false;
  }

  int getCoreId(Xbyak::util::Cpu& cpu) {
    uint32_t tmp[4];
    cpu.getCpuidEx(0x1F, 1, tmp);  // sub-leaf 1 is core domain
    // printf("!!!%x\t%x\t%x\t%x!!!\n", tmp[0], tmp[1], tmp[2], tmp[3]);
    if (tmp[0] != 0 && tmp[1] != 0)
      return tmp[3] >> 3;  // tmp[3] is APIC
    else
      return tmp[3];
  }

  bool isEcore(Xbyak::util::Cpu& cpu) {
    uint32_t tmp[4];
    cpu.getCpuid(0x1A, tmp);
    int core_type = (tmp[0] >> 24) & ((1u << 7) - 1);  // cpu.extractBit(a[0], 24, 31);
    switch (core_type) {
      case 32:
        // printf("Atom\n");
        return true;  // E-core or LPE-core
        break;
      case 64:
        // printf("Core\n");
        return false;  // P-core
        break;
      default:
        // printf("No hyper\n");
        return false;
        break;
    }
    return false;
  }
  static void core_bond(int core) {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 1 << core);
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    int s = sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    if (s != 0) printf("ERROR\n");
#endif
  }

  static void core_bond(std::thread& thread, int core) {
#ifdef _WIN32
    HANDLE handle = thread.native_handle();
    SetThreadAffinityMask(handle, 1 << core);
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    pthread_t pt = thread.native_handle();
    int s = pthread_setaffinity_np(pt, sizeof(cpuset), &cpuset);
    if (s != 0) printf("ERROR\n");
#endif
  }

  bool isHybrid() { return mHybrid; }

 protected:
  uint32_t L2Cache, L1Cache;
  bool mHybrid = false;
  bool mHasAVX2, mHasAVX_VNNI, mHasAVX, mHasAVX512_VNNI, mHasAMX_INT8, mHasAMX_BF16, mHasAVX512F, mHasAVX512_BF16,
      mHasAVX512_FP16;
  int numcores;
  int numthreads;
  std::vector<int> P_core, E_core, SMT_core;
  uint32_t E_L2Cache, E_L1Cache;
  float P_power = 4.8, E_power = 2.3;
};

#define GetCPUDevice() auto _cd = bestla::device::CpuDevice::getInstance();

class CpuBase {
 public:
  CpuBase() {
    GetCPUDevice();
    mL2Cache = _cd->getL2CacheSize();
    mL1Cache = _cd->getL1CacheSize();
    mNumThreads = _cd->getThreads();
  }
  size_t mL2Cache, mL1Cache;
  int mNumThreads;
};
}  // namespace device
}  // namespace bestla
