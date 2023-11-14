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
#include "jit_blas.h"
#include "xbyak/xbyak_util.h"

namespace jblas {

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

template <JBLAS_ISA ISA_T>
class isa_base {
 public:
  static bool constexpr avx = ISA_T >= JblasAVX;
  static bool constexpr avx2 = ISA_T >= JblasAVX2;
  static bool constexpr avx512f = ISA_T >= JblasAVX512F;
  static bool constexpr avx512_vnni = ISA_T >= JblasAVX512_VNNI;
  static bool constexpr avx512_fp16 = ISA_T >= JblasAVX512_FP16;
  static bool constexpr amx_bf16 = ISA_T >= JblasAMX_BF16;
  static bool constexpr amx_int8 = ISA_T >= JblasAMX_INT8;
};

class CpuDevice {
 public:
  inline void setThreads(int _nth) {
    if (_nth <= 0) {
      numthreads = numcores;
    } else {
      numthreads = std::min(numcores, _nth);
    }
  }
  inline int getThreads() { return numthreads; }
  inline int getCores() { return numcores; }
  inline uint32_t getL2CacheSize() { return L2Cache; }
  inline uint32_t getL1CacheSize() { return L1Cache; }
  inline bool AVX() { return mHasAVX; }
  inline bool AVX2() { return mHasAVX2; }
  inline bool AVX_VNNI() { return mHasAVX_VNNI; }
  inline bool AVX512F() { return mHasAVX512F; }
  inline bool AVX512_VNNI() { return mHasAVX512_VNNI; }
  inline bool AMX_INT8() { return mHasAMX_INT8; }
  inline bool AMX_BF16() { return mHasAMX_BF16; }
  inline bool AVX512_BF16() { return mHasAVX512_BF16; }
  inline bool AVX512_FP16() { return mHasAVX512_FP16; }
#define ADD_FLAG(isa) mHas##isa = _cpu.has(_cpu.t##isa)
  CpuDevice() {
    static Xbyak::util::Cpu _cpu;
    L1Cache = _cpu.getDataCacheSize(0);
    L2Cache = _cpu.getDataCacheSize(1);
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
    numthreads = numcores;
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

 protected:
  uint32_t L2Cache, L1Cache;
  bool mHasAVX2, mHasAVX_VNNI, mHasAVX, mHasAVX512_VNNI, mHasAMX_INT8, mHasAMX_BF16, mHasAVX512F, mHasAVX512_BF16,
      mHasAVX512_FP16;
  int numcores;
  int numthreads;
};

#define GetCPUDevice() auto _cd = jblas::device::CpuDevice::getInstance();

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

}  // namespace jblas
