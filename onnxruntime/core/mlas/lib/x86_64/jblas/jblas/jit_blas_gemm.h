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
#include <array>

#include "jit_blas_utils.h"
#include "jit_base.h"

namespace jblas {
namespace gemm {
enum class CompType : uint32_t {
  COMP_FP32 = 0,
  COMP_BF16_FP32 = 1,
  COMP_FP16_FP16 = 2,
  COMP_INT_START = 3,
  COMP_INT8_US_INT32 = COMP_INT_START,
  COMP_INT8_UU_INT32 = 4,
  COMP_INT8_SS_INT32 = 5,
  COMP_INT8_SU_INT32 = 6,
  COMP_INT16_SS_INT32 = 7,
  COMP_INT8_US_FP32 = 8,
  COMP_INT8_UU_FP32 = 9,
  COMP_INT8_SS_FP32 = 10,
  COMP_INT8_SU_FP32 = 11,
};

class CoreAttr {
 public:
  // INT32=LSB|**8bits:NTile**||**8bits:PackRow**||**8bits:CompType**||**8bits:Reserve**|
  static uint32_t constexpr NTILE_MASK = 0xff, NTILE_SHIFT = 0, PACKROW_MASK = 0xff00, PACKROW_SHIFT = 8,
                            COMP_MASK = 0xff0000, COMP_SHIFT = 16, ISA_MASK = 0xff000000, ISA_SHIFT = 24;

  static inline uint32_t get_mask_val(uint32_t raw, uint32_t mask, uint32_t shift) { return (raw & mask) >> shift; }
  static constexpr uint32_t make_core_id(uint32_t NTile, uint32_t PackRow, uint32_t CompType, uint32_t ISA) {
    return (NTile << NTILE_SHIFT) | (PackRow << PACKROW_SHIFT) | (CompType << COMP_SHIFT) | (ISA << ISA_SHIFT);
  }

  static void parse_id(uint32_t id, uint32_t* vals) {
    vals[0] = get_mask_val(id, NTILE_MASK, NTILE_SHIFT);
    vals[1] = get_mask_val(id, PACKROW_MASK, PACKROW_SHIFT);
    vals[2] = get_mask_val(id, COMP_MASK, COMP_SHIFT);
    vals[3] = get_mask_val(id, ISA_MASK, ISA_SHIFT);
  }

  static const char* to_str(uint32_t id) {
    static char tmp[128];
    uint32_t vals[4];
    parse_id(id, vals);
    sprintf(tmp, "N%d_PACK%d_COMP%d_ISA%d", vals[0], vals[1], vals[2], vals[3]);
    return tmp;
  }

  static inline size_t get_bsize(uint32_t id) {
    auto packrow = get_mask_val(id, PACKROW_MASK, PACKROW_SHIFT);
    return size_t(4 / packrow);
  }
};

namespace code {

template <int _NTILE, int _MTILE = 0>
class Avx2N8P1 : protected jblas::xbyak::JitAvx2 {
 public:
  static int constexpr RegLen = 8, PackRow = 1;
  static_assert(_NTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr MRegs = _MTILE == 0 ? (RegCount - 1) / NRegs : _MTILE;
  static_assert(NRegs * MRegs <= RegCount - 1);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs, KTILE = 1;
  static int constexpr KUNROLL = 2;
  static uint32_t constexpr ISA = (uint32_t)JBLAS_ISA::JblasAVX2;
  static uint32_t constexpr COMPUTE = (uint32_t)CompType::COMP_FP32;
  typedef float AType;
  typedef float BType;
  typedef float CType;

  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* matC;
    int cstride;
    int k;
    int n;
    int init;
  };
  typedef long long (*func_t)(params*);

  int CRegCount = 0, BRegCount = 0, ARegCount = 0, TmpRegCount = 0;
  int CReg = 0, BReg = 0, AReg = 0, TmpReg = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 protected:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_ret = rax;
  Xbyak::Opmask msk_wr = k1;

  void assign_regs() {
    CRegCount = MRegs * NRegs;
    ARegCount = 1;
    BRegCount = RegCount - ARegCount - CRegCount;
    if (BRegCount < NRegs) {
      BRegCount = 0;
      ARegCount = BRegCount + 1;
    }
    if (BRegCount > NRegs) {
      BRegCount = NRegs;
    }
    CReg = 0;
    BReg = CReg + CRegCount;
    AReg = BReg + BRegCount;
    TmpReg = AReg + ARegCount;
    assert(TmpReg <= RegCount);
    TmpRegCount = RegCount - TmpReg;
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();  // use local label for multiple instance
    Xbyak::util::StackFrame st(this, 1, 10, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }

  void generate_fma(int _mtile, int _ktile) {
    for (int kk = 0; kk < _ktile; kk++) {
      lea(reg_tmp1, ptr[reg_matAptr + kk * AKStepSize]);
      if (BRegCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          vmovups(vreg_t(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vbroadcastss(vreg_t(AReg), ptr[reg_tmp1]);
          add(reg_tmp1, reg_astride);
          for (int i = 0; i < NRegs; i++) {
            vfmadd231ps(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), vreg_t(BReg + i));
          }
        }
      } else if (BRegCount == 0) {
        for (int mm = 0; mm < _mtile; mm += ARegCount) {
          int mm_re = utils::remainsize(mm, _mtile, ARegCount);
          for (int imm = 0; imm < mm_re; imm++) {
            vbroadcastss(vreg_t(AReg + imm), ptr[reg_tmp1]);
            add(reg_tmp1, reg_astride);
            for (int i = 0; i < NRegs; i++) {
              vfmadd231ps(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg + imm),
                          ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
            }
          }
        }
      } else {
        assert(0);
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vxor(vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j));
      }
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
      }
      add(reg_matCptr, reg_cstride);
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(CReg + i * NRegs + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    outLocalLabel();
  }
};

template <int _NTILE, int _MTILE = 0>
class Avx512fN16P1 : protected jblas::xbyak::JitAvx512f {
 public:
  static int constexpr RegLen = 16, PackRow = 1;
  static_assert(_NTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr MRegs = _MTILE == 0 ? (RegCount - 1) / NRegs : _MTILE;
  static_assert(NRegs * MRegs <= RegCount - 1);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs, KTILE = 1;
  static int constexpr KUNROLL = 2;
  static uint32_t constexpr ISA = (uint32_t)JBLAS_ISA::JblasAVX512F;
  static uint32_t constexpr COMPUTE = (uint32_t)CompType::COMP_FP32;
  typedef float AType;
  typedef float BType;
  typedef float CType;

  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* matC;
    int cstride;
    int k;
    int n;
    int init;
  };
  typedef long long (*func_t)(params*);

  int CRegCount = 0, BRegCount = 0, ARegCount = 0, TmpRegCount = 0;
  int CReg = 0, BReg = 0, AReg = 0, TmpReg = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 protected:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_ret = rax;
  Xbyak::Opmask msk_wr = k1;

  void assign_regs() {
    CRegCount = MRegs * NRegs;
    ARegCount = 1;
    BRegCount = RegCount - ARegCount - CRegCount;
    if (BRegCount < NRegs) {
      BRegCount = 0;
      ARegCount = BRegCount + 1;
    }
    if (BRegCount > NRegs) {
      BRegCount = NRegs;
    }
    CReg = 0;
    BReg = CReg + CRegCount;
    AReg = BReg + BRegCount;
    TmpReg = AReg + ARegCount;
    assert(TmpReg <= RegCount);
    TmpRegCount = RegCount - TmpReg;
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();  // use local label for multiple instance
    Xbyak::util::StackFrame st(this, 1, 10, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }

  void generate_fma(int _mtile, int _ktile) {
    for (int kk = 0; kk < _ktile; kk++) {
      lea(reg_tmp1, ptr[reg_matAptr + kk * AKStepSize]);
      if (BRegCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          vmovups(vreg_t(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vbroadcastss(vreg_t(AReg), ptr[reg_tmp1]);
          add(reg_tmp1, reg_astride);
          for (int i = 0; i < NRegs; i++) {
            vfmadd231ps(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), vreg_t(BReg + i));
          }
        }
      } else if (BRegCount == 0) {
        for (int mm = 0; mm < _mtile; mm += ARegCount) {
          int mm_re = utils::remainsize(mm, _mtile, ARegCount);
          for (int imm = 0; imm < mm_re; imm++) {
            vbroadcastss(vreg_t(AReg + imm), ptr[reg_tmp1]);
            add(reg_tmp1, reg_astride);
            for (int i = 0; i < NRegs; i++) {
              vfmadd231ps(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg + imm),
                          ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
            }
          }
        }
      } else {
        assert(0);
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vxor(vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j));
      }
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
      }
      add(reg_matCptr, reg_cstride);
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(CReg + i * NRegs + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    outLocalLabel();
  }
};

template <int _NTILE, int _MTILE = 0>
class Avx512fp16N32P1 : protected jblas::xbyak::JitAvx512_fp16 {
 public:
  static int constexpr RegLen = 32, PackRow = 1;
  static_assert(_NTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr MRegs = _MTILE == 0 ? (RegCount - 1) / NRegs : _MTILE;
  static_assert(NRegs * MRegs <= RegCount - 1);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs, KTILE = 1;
  static int constexpr KUNROLL = 2;
  static uint32_t constexpr ISA = (uint32_t)JBLAS_ISA::JblasAVX512_FP16;
  static uint32_t constexpr COMPUTE = (uint32_t)CompType::COMP_FP16_FP16;
  typedef utils::fp16 AType;
  typedef utils::fp16 BType;
  typedef utils::fp16 CType;

  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* matC;
    int cstride;
    int k;
    int n;
    int init;
  };
  typedef long long (*func_t)(params*);

  int CRegCount = 0, BRegCount = 0, ARegCount = 0, TmpRegCount = 0;
  int CReg = 0, BReg = 0, AReg = 0, TmpReg = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 protected:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_ret = rax;
  Xbyak::Opmask msk_wr = k1;

  void assign_regs() {
    CRegCount = MRegs * NRegs;
    ARegCount = 1;
    BRegCount = RegCount - ARegCount - CRegCount;
    if (BRegCount < NRegs) {
      BRegCount = 0;
      ARegCount = BRegCount + 1;
    }
    if (BRegCount > NRegs) {
      BRegCount = NRegs;
    }
    CReg = 0;
    BReg = CReg + CRegCount;
    AReg = BReg + BRegCount;
    TmpReg = AReg + ARegCount;
    assert(TmpReg <= RegCount);
    TmpRegCount = RegCount - TmpReg;
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();  // use local label for multiple instance
    Xbyak::util::StackFrame st(this, 1, 10, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }

  void generate_fma(int _mtile, int _ktile) {
    for (int kk = 0; kk < _ktile; kk++) {
      lea(reg_tmp1, ptr[reg_matAptr + kk * AKStepSize]);
      if (BRegCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          vmovups(vreg_t(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastw(vreg_t(AReg), ptr[reg_tmp1]);
          add(reg_tmp1, reg_astride);
          for (int i = 0; i < NRegs; i++) {
            vfmadd231ph(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), vreg_t(BReg + i));
          }
        }
      } else if (BRegCount == 0) {
        for (int mm = 0; mm < _mtile; mm += ARegCount) {
          int mm_re = utils::remainsize(mm, _mtile, ARegCount);
          for (int imm = 0; imm < mm_re; imm++) {
            vpbroadcastw(vreg_t(AReg + imm), ptr[reg_tmp1]);
            add(reg_tmp1, reg_astride);
            for (int i = 0; i < NRegs; i++) {
              vfmadd231ph(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg + imm),
                          ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
            }
          }
        }
      } else {
        assert(0);
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vxor(vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j));
      }
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
      }
      add(reg_matCptr, reg_cstride);
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(CReg + i * NRegs + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    outLocalLabel();
  }
};

template <int _NTILE, int _MTILE = 0>
class Avx512bf16N16P2 : protected jblas::xbyak::JitAvx512_bf16 {
 public:
  static int constexpr RegLen = 16, PackRow = 2;
  static_assert(_NTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr MRegs = _MTILE == 0 ? (RegCount - 1) / NRegs : _MTILE;
  static_assert(NRegs * MRegs <= RegCount - 1);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs, KTILE = 2;
  static int constexpr KUNROLL = 2;
  static uint32_t constexpr ISA = (uint32_t)JBLAS_ISA::JblasAVX512_BF16;
  static uint32_t constexpr COMPUTE = (uint32_t)CompType::COMP_BF16_FP32;
  typedef utils::bf16 AType;
  typedef utils::bf16 BType;
  typedef float CType;

  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* matC;
    int cstride;
    int k;
    int n;
    int init;
  };
  typedef long long (*func_t)(params*);

  int CRegCount = 0, BRegCount = 0, ARegCount = 0, TmpRegCount = 0;
  int CReg = 0, BReg = 0, AReg = 0, TmpReg = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 protected:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_ret = rax;
  Xbyak::Opmask msk_wr = k1;

  void assign_regs() {
    CRegCount = MRegs * NRegs;
    ARegCount = 1;
    BRegCount = RegCount - ARegCount - CRegCount;
    if (BRegCount < NRegs) {
      BRegCount = 0;
      ARegCount = BRegCount + 1;
    }
    if (BRegCount > NRegs) {
      BRegCount = NRegs;
    }
    CReg = 0;
    BReg = CReg + CRegCount;
    AReg = BReg + BRegCount;
    TmpReg = AReg + ARegCount;
    assert(TmpReg <= RegCount);
    TmpRegCount = RegCount - TmpReg;
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();  // use local label for multiple instance
    Xbyak::util::StackFrame st(this, 1, 10, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }

  void generate_fma(int _mtile, int _ktile) {
    for (int kk = 0; kk < _ktile; kk++) {
      lea(reg_tmp1, ptr[reg_matAptr + kk * AKStepSize]);
      if (BRegCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          vmovups(vreg_t(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vbroadcastss(vreg_t(AReg), ptr[reg_tmp1]);
          add(reg_tmp1, reg_astride);
          for (int i = 0; i < NRegs; i++) {
            vdpbf16ps(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), vreg_t(BReg + i));
          }
        }
      } else if (BRegCount == 0) {
        for (int mm = 0; mm < _mtile; mm += ARegCount) {
          int mm_re = utils::remainsize(mm, _mtile, ARegCount);
          for (int imm = 0; imm < mm_re; imm++) {
            vbroadcastss(vreg_t(AReg + imm), ptr[reg_tmp1]);
            add(reg_tmp1, reg_astride);
            for (int i = 0; i < NRegs; i++) {
              vdpbf16ps(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg + imm),
                        ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
            }
          }
        }
      } else {
        assert(0);
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vxor(vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j));
      }
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
      }
      add(reg_matCptr, reg_cstride);
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(CReg + i * NRegs + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    outLocalLabel();
  }
};

template <int _NTILE, int _MTILE = 0>
class Avx512vnniN16P4 : protected jblas::xbyak::JitAvx512vnni {
 public:
  static int constexpr RegLen = 16, PackRow = 4;
  static_assert(_NTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr MRegs = _MTILE == 0 ? (RegCount - 1) / NRegs : _MTILE;
  static_assert(NRegs * MRegs <= RegCount - 1);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs, KTILE = 4;
  static int constexpr KUNROLL = 2;
  static uint32_t constexpr ISA = (uint32_t)JBLAS_ISA::JblasAVX512_VNNI;
  static uint32_t constexpr COMPUTE = (uint32_t)CompType::COMP_INT8_US_INT32;
  typedef uint8_t AType;
  typedef int8_t BType;
  typedef int32_t CType;
  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* matC;
    int cstride;
    int k;
    int n;
    int init;
  };
  typedef long long (*func_t)(params*);

  int CRegCount = 0, BRegCount = 0, ARegCount = 0, TmpRegCount = 0;
  int CReg = 0, BReg = 0, AReg = 0, TmpReg = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 private:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_ret = rax;

 protected:
  void assign_regs() {
    CRegCount = MRegs * NRegs;
    ARegCount = 1;
    BRegCount = RegCount - ARegCount - CRegCount;
    if (BRegCount < NRegs) {
      BRegCount = 0;
      ARegCount = BRegCount + 1;
    }
    if (BRegCount > NRegs) {
      BRegCount = NRegs;
    }
    CReg = 0;
    BReg = CReg + CRegCount;
    AReg = BReg + BRegCount;
    TmpReg = AReg + ARegCount;
    assert(TmpReg <= RegCount);
    TmpRegCount = RegCount - TmpReg;
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();
    Xbyak::util::StackFrame st(this, 1, 10, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }

  void generate_fma(int _mtile, int _kunroll) {
    for (int kk = 0; kk < _kunroll; kk++) {
      lea(reg_tmp1, ptr[reg_matAptr + kk * AKStepSize]);
      if (BRegCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          vmovups(vreg_t(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastd(vreg_t(AReg), ptr[reg_tmp1]);
          add(reg_tmp1, reg_astride);
          for (int i = 0; i < NRegs; i++) {
            vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), vreg_t(BReg + i));
          }
        }
      } else if (BRegCount == 0) {
        for (int mm = 0; mm < _mtile; mm += ARegCount) {
          int mm_re = utils::remainsize(mm, _mtile, ARegCount);
          for (int imm = 0; imm < mm_re; imm++) {
            vpbroadcastd(vreg_t(AReg + imm), ptr[reg_tmp1]);
            add(reg_tmp1, reg_astride);
            for (int i = 0; i < NRegs; i++) {
              vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg + imm),
                         ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
            }
          }
        }
      } else {
        assert(0);
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vxor(vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j));
      }
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
      }
      add(reg_matCptr, reg_cstride);
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(CReg + i * NRegs + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    outLocalLabel();
  }
};

template <int _NTILE, int _MTILE = 0>
class AvxvnniN8P4 : protected jblas::xbyak::JitAvxvnni {
 public:
  static int constexpr RegLen = 8, PackRow = 4;
  static_assert(_NTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr MRegs = _MTILE == 0 ? (RegCount - 1) / NRegs : _MTILE;
  static_assert(NRegs * MRegs <= RegCount - 1);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs, KTILE = 4;
  static int constexpr KUNROLL = 2;
  static uint32_t constexpr ISA = (uint32_t)JBLAS_ISA::JblasAVX_VNNI;
  static uint32_t constexpr COMPUTE = (uint32_t)CompType::COMP_INT8_US_INT32;
  typedef uint8_t AType;
  typedef int8_t BType;
  typedef int32_t CType;
  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* matC;
    int cstride;
    int k;
    int n;
    int init;
  };
  typedef long long (*func_t)(params*);

  int CRegCount = 0, BRegCount = 0, ARegCount = 0, TmpRegCount = 0;
  int CReg = 0, BReg = 0, AReg = 0, TmpReg = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 private:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_ret = rax;
  Xbyak::Opmask msk_wr = k1;

 protected:
  void assign_regs() {
    CRegCount = MRegs * NRegs;
    ARegCount = 1;
    BRegCount = RegCount - ARegCount - CRegCount;
    if (BRegCount < NRegs) {
      BRegCount = 0;
      ARegCount = BRegCount + 1;
    }
    if (BRegCount > NRegs) {
      BRegCount = NRegs;
    }
    CReg = 0;
    BReg = CReg + CRegCount;
    AReg = BReg + BRegCount;
    TmpReg = AReg + ARegCount;
    assert(TmpReg <= RegCount);
    TmpRegCount = RegCount - TmpReg;
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();
    Xbyak::util::StackFrame st(this, 1, 10, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }

  void generate_fma(int _mtile, int _kunroll) {
    for (int kk = 0; kk < _kunroll; kk++) {
      lea(reg_tmp1, ptr[reg_matAptr + kk * AKStepSize]);
      if (BRegCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          vmovups(vreg_t(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastd(vreg_t(AReg), ptr[reg_tmp1]);
          add(reg_tmp1, reg_astride);
          for (int i = 0; i < NRegs; i++) {
            vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), vreg_t(BReg + i));
          }
        }
      } else if (BRegCount == 0) {
        for (int mm = 0; mm < _mtile; mm += ARegCount) {
          int mm_re = utils::remainsize(mm, _mtile, ARegCount);
          for (int imm = 0; imm < mm_re; imm++) {
            vpbroadcastd(vreg_t(AReg + imm), ptr[reg_tmp1]);
            add(reg_tmp1, reg_astride);
            for (int i = 0; i < NRegs; i++) {
              vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg + imm),
                         ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
            }
          }
        }
      } else {
        assert(0);
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vxor(vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j));
      }
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
      }
      add(reg_matCptr, reg_cstride);
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(CReg + i * NRegs + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    outLocalLabel();
  }
};

template <int _NTILE, int _MTILE = 0>
class Amxbf16N16P2 : protected jblas::xbyak::JitAmxbf16 {
 public:
  static int constexpr RegLen = 16, PackRow = 2;
  static_assert(_NTILE % RegLen == 0);
  static_assert(_MTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr MRegs = _MTILE == 0 ? 1 : _MTILE / RegLen;
  static_assert(NRegs * MRegs + 2 <= TileCount);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs * RegLen, KTILE = 32;
  static int constexpr KUNROLL = 2;
  static uint32_t constexpr ISA = (uint32_t)JBLAS_ISA::JblasAMX_BF16;
  static uint32_t constexpr COMPUTE = (uint32_t)CompType::COMP_BF16_FP32;
  typedef utils::bf16 AType;
  typedef utils::bf16 BType;
  typedef float CType;

  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* matC;
    int cstride;
    int k;
    int n;
    int init;
    void* workspace;
  };
  typedef long long (*func_t)(params*);

  int TmpRegCount = RegCount;
  int TmpReg = 0;
  int CTileCount = 0, ATileCount = 0, BTileCount = 0;
  int CTile = 0, ATile = 0, BTile = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 protected:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_tmp3;
  Xbyak::Reg64 reg_ret = rax;

  void assign_regs() {
    CTileCount = NRegs * MRegs;
    auto tile_re = TileCount - CTileCount;
    if (tile_re - 1 >= NRegs) {
      BTileCount = NRegs;
      ATileCount = tile_re - BTileCount;
    } else if (tile_re - 1 >= MRegs) {
      ATileCount = MRegs;
      BTileCount = tile_re - ATileCount;
    } else {
      ATileCount = 1;
      BTileCount = tile_re - ATileCount;
    }
    CTile = 0;
    ATile = CTile + CTileCount;
    BTile = ATile + ATileCount;
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();  // use local label for multiple instance
    Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_tmp3 = st.t[10];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }

  void generate_fma(int _mtile, int kunrll) {
    auto& reg_Bstride = reg_tmp1;
    mov(reg_Bstride, NTILE * 4);
    int mtiles = _mtile / RegLen;

    for (int kk = 0; kk < kunrll; kk++) {
      auto& reg_Atmp = reg_tmp2;
      if (mtiles == 1) {
        reg_Atmp = reg_matAptr;
      } else {
        mov(reg_Atmp, reg_matAptr);
      }
      if (BTileCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          tileloaddt1(Xbyak::Tmm(BTile + i), ptr[reg_matBptr + reg_Bstride + kk * BKStepSize + i * 64]);
        }
        for (int mm = 0; mm < mtiles; mm++) {
          tileloadd(Xbyak::Tmm(ATile), ptr[reg_Atmp + reg_astride + kk * AKStepSize]);
          for (int i = 0; i < NRegs; i++) {
            tdpbf16ps(Xbyak::Tmm(CTile + mm * NRegs + i), Xbyak::Tmm(ATile), Xbyak::Tmm(BTile + i));
          }
          if (mm != mtiles - 1) {
            lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
            lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
          }
        }
      } else {
        if (ATileCount == mtiles) {
          for (int mm = 0; mm < mtiles; mm++) {
            tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_Atmp + reg_astride + kk * AKStepSize]);
            if (mm != mtiles - 1) {
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
            }
          }
          for (int i = 0; i < NRegs; i++) {
            tileloaddt1(Xbyak::Tmm(BTile), ptr[reg_matBptr + reg_Bstride + kk * BKStepSize + i * 64]);
            for (int mm = 0; mm < mtiles; mm++) {
              tdpbf16ps(Xbyak::Tmm(CTile + mm * NRegs + i), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile));
            }
          }
        } else {
          for (int mm = 0; mm < mtiles; mm++) {
            tileloadd(Xbyak::Tmm(ATile), ptr[reg_Atmp + reg_astride + kk * AKStepSize]);
            for (int i = 0; i < NRegs; i++) {
              tileloaddt1(Xbyak::Tmm(BTile), ptr[reg_matBptr + reg_Bstride + kk * BKStepSize + i * 64]);
              tdpbf16ps(Xbyak::Tmm(CTile + mm * NRegs + i), Xbyak::Tmm(ATile), Xbyak::Tmm(BTile));
            }
            if (mm != mtiles - 1) {
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
            }
          }
        }
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < CTileCount; i++) {
      tilezero(Xbyak::Tmm(CTile + i));
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    int mtnum = _mtile / 16;
    for (int mm = 0; mm < mtnum; mm++) {
      for (int i = 0; i < NRegs; i++) {
        tileloaddt1(Xbyak::Tmm(CTile + mm * NRegs + i), ptr[reg_matCptr + reg_cstride + i * 64]);
      }
      if (mm != mtnum - 1) {
        lea(reg_matCptr, ptr[reg_matCptr + 8 * reg_cstride]);
        lea(reg_matCptr, ptr[reg_matCptr + 8 * reg_cstride]);
      }
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, dword[parambase + OFFSET(workspace)]);
    mov(reg_tmp1, NTILE * 4);
    for (int mm = 0; mm < MRegs; mm++) {
      for (int i = 0; i < NRegs; i++) {
        tilestored(ptr[reg_tmp + reg_tmp1 + i * 64 + mm * 16 * NTILE * 4], Xbyak::Tmm(CTile + mm * NRegs + i));
      }
    }
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    int zunroll = TmpRegCount / NRegs;
    for (int i = 0; i < _mtile; i += zunroll) {
      int m_re = utils::remainsize(i, _mtile, zunroll);
      for (int im = 0; im < m_re; im++) {
        for (int j = 0; j < NRegs; j++) {
          vmovups(vreg_t(TmpReg + im * NRegs + j), ptr[reg_tmp + j * 64 + (i + im) * NTILE * 4]);
          vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(TmpReg + im * NRegs + j));
        }
        add(reg_matCptr, reg_cstride);
      }
    }
    outLocalLabel();
  }
};

template <typename AT, typename BT, int _NTILE, int _MTILE = 0>
class Amxint8N16P4 : protected jblas::xbyak::JitAmxint8 {
 public:
  static int constexpr RegLen = 16, PackRow = 4;
  static_assert(_NTILE % RegLen == 0);
  static_assert(_MTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr MRegs = _MTILE == 0 ? 1 : _MTILE / RegLen;
  static_assert(NRegs * MRegs + 2 <= TileCount);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs * RegLen, KTILE = 64;
  static int constexpr KUNROLL = 2;
  static uint32_t constexpr ISA = (uint32_t)JBLAS_ISA::JblasAMX_INT8;
  static uint32_t constexpr COMPUTE =
      (uint32_t)(std::is_same_v<AT, int8_t>
                     ? std::is_same_v<BT, int8_t> ? CompType::COMP_INT8_SS_INT32 : CompType::COMP_INT8_SU_INT32
                 : std::is_same_v<BT, int8_t> ? CompType::COMP_INT8_US_INT32
                                              : CompType::COMP_INT8_UU_INT32);
  using AType = AT;
  using BType = BT;
  typedef int32_t CType;

  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* matC;
    int cstride;
    int k;
    int n;
    int init;
    void* workspace;
  };
  typedef long long (*func_t)(params*);

  int TmpRegCount = RegCount;
  int TmpReg = 0;
  int CTileCount = 0, ATileCount = 0, BTileCount = 0;
  int CTile = 0, ATile = 0, BTile = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 protected:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_tmp3;
  Xbyak::Reg64 reg_ret = rax;

  void assign_regs() {
    CTileCount = NRegs * MRegs;
    auto tile_re = TileCount - CTileCount;
    if (tile_re - 1 >= NRegs) {
      BTileCount = NRegs;
      ATileCount = tile_re - BTileCount;
    } else if (tile_re - 1 >= MRegs) {
      ATileCount = MRegs;
      BTileCount = tile_re - ATileCount;
    } else {
      ATileCount = 1;
      BTileCount = tile_re - ATileCount;
    }
    CTile = 0;
    ATile = CTile + CTileCount;
    BTile = ATile + ATileCount;
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();  // use local label for multiple instance
    Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_tmp3 = st.t[10];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }

  void generate_fma(int _mtile, int kunrll) {
    auto& reg_Bstride = reg_tmp1;
    mov(reg_Bstride, NTILE * 4);
    int mtiles = _mtile / RegLen;

    for (int kk = 0; kk < kunrll; kk++) {
      auto& reg_Atmp = reg_tmp2;
      if (mtiles == 1) {
        reg_Atmp = reg_matAptr;
      } else {
        mov(reg_Atmp, reg_matAptr);
      }
      if (BTileCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          tileloaddt1(Xbyak::Tmm(BTile + i), ptr[reg_matBptr + reg_Bstride + kk * BKStepSize + i * 64]);
        }
        for (int mm = 0; mm < mtiles; mm++) {
          tileloadd(Xbyak::Tmm(ATile), ptr[reg_Atmp + reg_astride + kk * AKStepSize]);
          for (int i = 0; i < NRegs; i++) {
            _tdpb<AT, BT>(Xbyak::Tmm(CTile + mm * NRegs + i), Xbyak::Tmm(ATile), Xbyak::Tmm(BTile + i));
          }
          if (mm != mtiles - 1) {
            lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
            lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
          }
        }
      } else {
        if (ATileCount == mtiles) {
          for (int mm = 0; mm < mtiles; mm++) {
            tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_Atmp + reg_astride + kk * AKStepSize]);
            if (mm != mtiles - 1) {
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
            }
          }
          for (int i = 0; i < NRegs; i++) {
            tileloaddt1(Xbyak::Tmm(BTile), ptr[reg_matBptr + reg_Bstride + kk * BKStepSize + i * 64]);
            for (int mm = 0; mm < mtiles; mm++) {
              _tdpb<AT, BT>(Xbyak::Tmm(CTile + mm * NRegs + i), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile));
            }
          }
        } else {
          for (int mm = 0; mm < mtiles; mm++) {
            tileloadd(Xbyak::Tmm(ATile), ptr[reg_Atmp + reg_astride + kk * AKStepSize]);
            for (int i = 0; i < NRegs; i++) {
              tileloaddt1(Xbyak::Tmm(BTile), ptr[reg_matBptr + reg_Bstride + kk * BKStepSize + i * 64]);
              _tdpb<AT, BT>(Xbyak::Tmm(CTile + mm * NRegs + i), Xbyak::Tmm(ATile), Xbyak::Tmm(BTile));
            }
            if (mm != mtiles - 1) {
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
              lea(reg_Atmp, ptr[reg_Atmp + 8 * reg_astride]);
            }
          }
        }
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < CTileCount; i++) {
      tilezero(Xbyak::Tmm(CTile + i));
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    int mtnum = _mtile / 16;
    for (int mm = 0; mm < mtnum; mm++) {
      for (int i = 0; i < NRegs; i++) {
        tileloaddt1(Xbyak::Tmm(CTile + mm * NRegs + i), ptr[reg_matCptr + reg_cstride + i * 64]);
      }
      if (mm != mtnum - 1) {
        lea(reg_matCptr, ptr[reg_matCptr + 8 * reg_cstride]);
        lea(reg_matCptr, ptr[reg_matCptr + 8 * reg_cstride]);
      }
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, dword[parambase + OFFSET(workspace)]);
    mov(reg_tmp1, NTILE * 4);
    for (int mm = 0; mm < MRegs; mm++) {
      for (int i = 0; i < NRegs; i++) {
        tilestored(ptr[reg_tmp + reg_tmp1 + i * 64 + mm * 16 * NTILE * 4], Xbyak::Tmm(CTile + mm * NRegs + i));
      }
    }
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    int zunroll = TmpRegCount / NRegs;
    for (int i = 0; i < _mtile; i += zunroll) {
      int m_re = utils::remainsize(i, _mtile, zunroll);
      for (int im = 0; im < m_re; im++) {
        for (int j = 0; j < NRegs; j++) {
          vmovups(vreg_t(TmpReg + im * NRegs + j), ptr[reg_tmp + j * 64 + (i + im) * NTILE * 4]);
          vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(TmpReg + im * NRegs + j));
        }
        add(reg_matCptr, reg_cstride);
      }
    }
    outLocalLabel();
  }
};
template <int N, int M>
using Amxint8N16P4US = Amxint8N16P4<uint8_t, int8_t, N, M>;

template <int N, int M>
using Amxint8N16P4SS = Amxint8N16P4<int8_t, int8_t, N, M>;

class AmxConfigure : protected jblas::xbyak::JitAmxtile {
 public:
  typedef long long (*func_t)(tileconfig_t*);

  static void configure(int TILE_M, int TILE_N, int TILE_K, int elesize, int ANum, int BNum, int CNum) {
    static AmxConfigure code;
    tileconfig_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    configure_tiles(cfg, TILE_M, TILE_N, TILE_K, elesize, ANum, BNum, CNum);
    code.mKernel(&cfg);
  }

 protected:
  AmxConfigure() {
    generate_config(this);
    mKernel = getCode<func_t>();
  }

  func_t mKernel = nullptr;
};

namespace kblock {
// optimize for kblock gemm, each block size in k dimension has dequant operation
// all accumulators use fp32 dtype.
template <int _NTILE, int _MTILE = 0>
class Avx512fN16P1 : protected jblas::xbyak::JitAvx512f {
 public:
  static int constexpr RegLen = 16, PackRow = 1;
  static_assert(_NTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr MRegs = _MTILE == 0 ? (RegCount - 1) / NRegs : _MTILE;
  static_assert(NRegs * MRegs <= RegCount - 1);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs, KTILE = 1;
  static int constexpr KUNROLL = 2;
  static uint32_t constexpr ISA = (uint32_t)JBLAS_ISA::JblasAVX512F;
  static uint32_t constexpr COMPUTE = (uint32_t)CompType::COMP_FP32;
  typedef float AType;
  typedef float BType;
  typedef float CType;

  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* matC;
    int cstride;
    int k;
    int n;
    int init;
  };
  typedef long long (*func_t)(params*);

  int CRegCount = 0, BRegCount = 0, ARegCount = 0, TmpRegCount = 0;
  int CReg = 0, BReg = 0, AReg = 0, TmpReg = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 protected:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_ret = rax;
  Xbyak::Opmask msk_wr = k1;

  void assign_regs() {
    CRegCount = MRegs * NRegs;
    ARegCount = 1;
    BRegCount = RegCount - ARegCount - CRegCount;
    if (BRegCount < NRegs) {
      BRegCount = 0;
      ARegCount = BRegCount + 1;
    }
    if (BRegCount > NRegs) {
      BRegCount = NRegs;
    }
    CReg = 0;
    BReg = CReg + CRegCount;
    AReg = BReg + BRegCount;
    TmpReg = AReg + ARegCount;
    assert(TmpReg <= RegCount);
    TmpRegCount = RegCount - TmpReg;
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();  // use local label for multiple instance
    Xbyak::util::StackFrame st(this, 1, 10, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    mov(reg_tmp, reg_ksize);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kloop", T_NEAR);
    L(".unkloop");
    generate_fma(_mtile, KUNROLL);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_iterk, KUNROLL * KTILE);
    cmp(reg_iterk, reg_tmp);  // k iteration variable
    jb(".unkloop");
    cmp(reg_tmp, reg_ksize);
    jge(".kend", T_NEAR);
    L(".kloop");
    generate_fma(_mtile, 1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_iterk, 1 * KTILE);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");
    L(".kend");
    outLocalLabel();
  }

  void generate_fma(int _mtile, int _ktile) {
    for (int kk = 0; kk < _ktile; kk++) {
      lea(reg_tmp1, ptr[reg_matAptr + kk * AKStepSize]);
      if (BRegCount == NRegs) {
        for (int i = 0; i < NRegs; i++) {
          vmovups(vreg_t(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vbroadcastss(vreg_t(AReg), ptr[reg_tmp1]);
          add(reg_tmp1, reg_astride);
          for (int i = 0; i < NRegs; i++) {
            vfmadd231ps(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), vreg_t(BReg + i));
          }
        }
      } else if (BRegCount == 0) {
        for (int mm = 0; mm < _mtile; mm += ARegCount) {
          int mm_re = utils::remainsize(mm, _mtile, ARegCount);
          for (int imm = 0; imm < mm_re; imm++) {
            vbroadcastss(vreg_t(AReg + imm), ptr[reg_tmp1]);
            add(reg_tmp1, reg_astride);
            for (int i = 0; i < NRegs; i++) {
              vfmadd231ps(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg + imm),
                          ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
            }
          }
        }
      } else {
        assert(0);
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vxor(vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j), vreg_t(CReg + i * NRegs + j));
      }
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
      }
      add(reg_matCptr, reg_cstride);
    }
    L(".end");
    outLocalLabel();
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(CReg + i * NRegs + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    outLocalLabel();
  }
};

template <int _NTILE, int _MTILE = 0>
class Avx512vnniN16P4 : protected jblas::xbyak::JitAvx512vnni {
 public:
  static int constexpr RegLen = 16, PackRow = 4;
  static_assert(_NTILE % RegLen == 0);
  static int constexpr NRegs = _NTILE / RegLen;
  static int constexpr MRegs = _MTILE == 0 ? (RegCount - 1 - NRegs) / (NRegs * 2) : _MTILE;
  static_assert(NRegs * MRegs <= RegCount - 1);
  static int constexpr NTILE = RegLen * NRegs, MTILE = MRegs, KTILE = 4;
  static int constexpr KUNROLL = 2;
  static uint32_t constexpr ISA = (uint32_t)JBLAS_ISA::JblasAVX512_VNNI;
  static uint32_t constexpr COMPUTE = (uint32_t)CompType::COMP_INT8_US_FP32;
  typedef uint8_t AType;
  typedef int8_t BType;
  typedef float CType;

  struct params {
    AType* matA;
    int astride;
    BType* matB;
    int bstride;
    CType* matC;
    int cstride;
    uint8_t* zpA;
    float* scaleA;
    int ldsa;
    float* scaleB;
    float* reduceB;
    int ldsb;
    int k;
    int n;
    int kblock;
    int init;
  };
  typedef long long (*func_t)(params*);

  int CRegCount = 0, BRegCount = 0, ARegCount = 0, TmpRegCount = 0;
  int CReg = 0, CF32Reg = 0, BReg = 0, AReg = 0, TmpReg = 0;
  static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
  static int constexpr AKStepSize = KTILE * sizeof(AType);

  void generate_code(int _mtile) {
    assign_regs();
    reset();
    generate_mtile(_mtile);
    ready();
    mKernel = getCode<func_t>();
  }
  func_t mKernel = nullptr;

 protected:
  Xbyak::Reg64 parambase;
  Xbyak::Reg64 reg_matAptr;
  Xbyak::Reg64 reg_matBptr;
  Xbyak::Reg64 reg_matCptr;
  Xbyak::Reg64 reg_ksize;
  Xbyak::Reg64 reg_nsize;
  Xbyak::Reg64 reg_cstride;
  Xbyak::Reg64 reg_astride;
  Xbyak::Reg64 reg_iterk;
  Xbyak::Reg64 reg_iterkb;
  Xbyak::Reg64 reg_itern;
  Xbyak::Reg64 reg_tmp;
  Xbyak::Reg64 reg_tmp1;
  Xbyak::Reg64 reg_tmp2;
  Xbyak::Reg64 reg_tmp3;
  Xbyak::Reg64 reg_tmp4;
  Xbyak::Reg64 reg_ret = rax;

  void assign_regs() {
    CRegCount = MRegs * NRegs;
    ARegCount = 1;
    BRegCount = NRegs;
    CReg = 0;
    CF32Reg = CReg + CRegCount;
    BReg = CF32Reg + CRegCount;
    AReg = BReg + BRegCount;
    TmpReg = AReg + ARegCount;
    assert(TmpReg < RegCount);
    TmpRegCount = RegCount - TmpReg;
    assert(TmpRegCount >= 1);
  }

  void generate_mtile(int _mtile) {
    inLocalLabel();  // use local label for multiple instance
    Xbyak::util::StackFrame st(this, 1, 13, 16 * 10);
    parambase = st.p[0];
    reg_matAptr = st.t[0];
    reg_matBptr = st.t[1];
    reg_matCptr = st.t[0];
    reg_ksize = st.t[2];
    reg_astride = st.t[3];
    reg_cstride = st.t[3];
    reg_iterk = st.t[4];
    reg_iterkb = st.t[12];
    reg_tmp = st.t[5];
    reg_tmp1 = st.t[6];
    reg_tmp2 = st.t[7];
    reg_tmp3 = st.t[10];
    reg_tmp4 = st.t[11];
    reg_nsize = st.t[8];
    reg_itern = st.t[9];
    reg_ret = rax;

    vreg_push(rsp);

    load32(reg_ksize, ptr[parambase + OFFSET(k)]);
    load32(reg_nsize, ptr[parambase + OFFSET(n)]);
    xor_(reg_itern, reg_itern);
    L(".nloop");
    init_regs(_mtile);
    mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
    load32(reg_astride, ptr[parambase + OFFSET(astride)]);
    mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
    load32(reg_tmp, ptr[parambase + OFFSET(bstride)]);
    imul(reg_tmp, reg_itern);
    lea(reg_matBptr, ptr[reg_matBptr + reg_tmp]);
    xor_(reg_iterk, reg_iterk);
    generate_kloop(_mtile);
    write_back(_mtile);
    add(reg_itern, NTILE);
    cmp(reg_itern, reg_nsize);
    jb(".nloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);

    outLocalLabel();  // end of local label
  }

  void generate_kloop(int _mtile) {
    inLocalLabel();
    xor_(reg_iterkb, reg_iterkb);
    L(".kloop");
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vpxorq(Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j));
      }
    }
    xor_(reg_tmp2, reg_tmp2);
    load32(reg_tmp3, ptr[parambase + OFFSET(kblock)]);
    mov(reg_tmp, reg_tmp3);
    padto_le(reg_tmp, KUNROLL * KTILE);
    cmp(reg_tmp, 0);
    jz(".kbloop", T_NEAR);
    L(".unkbloop");
    generate_fma(_mtile, KUNROLL, reg_tmp1);
    add(reg_matAptr, KUNROLL * AKStepSize);
    add(reg_matBptr, KUNROLL * BKStepSize);
    add(reg_tmp2, KUNROLL * KTILE);
    cmp(reg_tmp2, reg_tmp);
    jb(".unkbloop");
    cmp(reg_tmp, reg_tmp3);
    jge(".kend", T_NEAR);
    L(".kbloop");
    generate_fma(_mtile, 1, reg_tmp1);
    add(reg_matAptr, 1 * AKStepSize);
    add(reg_matBptr, 1 * BKStepSize);
    add(reg_tmp2, 1 * KTILE);
    cmp(reg_tmp2, reg_tmp3);
    jb(".kbloop");
    L(".kend");
    add(reg_iterk, reg_tmp2);
    generate_f32_accumulate(_mtile);
    generate_zp_correction(_mtile);
    inc(reg_iterkb);
    cmp(reg_iterk, reg_ksize);  // k iteration variable
    jb(".kloop");

    outLocalLabel();
  }

  void generate_fma(int _mtile, int _ktile, Xbyak::Reg64& tmp) {
    for (int kk = 0; kk < _ktile; kk++) {
      lea(tmp, ptr[reg_matAptr + kk * AKStepSize]);
      for (int i = 0; i < NRegs; i++) {
        vmovups(vreg_t(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
      }
      for (int mm = 0; mm < _mtile; mm++) {
        vpbroadcastd(vreg_t(AReg), ptr[reg_tmp1]);
        add(reg_tmp1, reg_astride);
        for (int i = 0; i < NRegs; i++) {
          vpdpbusds_(vreg_t(CReg + mm * NRegs + i), vreg_t(AReg), vreg_t(BReg + i));
        }
      }
    }
  }

  void init_regs(int _mtile) {
    inLocalLabel();
    load32(reg_tmp, ptr[parambase + OFFSET(init)]);
    cmp(reg_tmp, 0);
    je(".read", T_NEAR);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vxor(vreg_t(CF32Reg + i * NRegs + j), vreg_t(CF32Reg + i * NRegs + j), vreg_t(CF32Reg + i * NRegs + j));
      }
    }
    jmp(".end", T_NEAR);
    L(".read");
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(vreg_t(CF32Reg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
      }
      add(reg_matCptr, reg_cstride);
    }
    L(".end");
    outLocalLabel();
  }

  void generate_f32_accumulate(int _mtile) {
    load32(reg_tmp, ptr[parambase + OFFSET(ldsb)]);
    imul(reg_tmp, reg_iterkb);
    mov(reg_tmp2, ptr[parambase + OFFSET(scaleB)]);
    lea(reg_tmp2, ptr[reg_tmp2 + reg_tmp * sizeof(float)]);
    lea(reg_tmp2, ptr[reg_tmp2 + reg_itern * sizeof(float)]);

    mov(reg_tmp, ptr[parambase + OFFSET(scaleA)]);
    lea(reg_tmp, ptr[reg_tmp + reg_iterkb * sizeof(float)]);
    load32(reg_tmp1, ptr[parambase + OFFSET(ldsa)]);
    for (int i = 0; i < NRegs; i++) {
      vmovups(Xbyak::Zmm(BReg + i), ptr[reg_tmp2 + i * VecBytes]);
    }
    for (int mm = 0; mm < _mtile; mm++) {
      vbroadcastss(Xbyak::Zmm(TmpReg), ptr[reg_tmp]);
      lea(reg_tmp, ptr[reg_tmp + reg_tmp1 * sizeof(float)]);
      for (int i = 0; i < NRegs; i++) {
        vcvtdq2ps(Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(CReg + mm * NRegs + i));
        vmulps(Xbyak::Zmm(AReg), Xbyak::Zmm(TmpReg), Xbyak::Zmm(BReg + i));
        vmulps(Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(AReg));
        vaddps(Xbyak::Zmm(CF32Reg + mm * NRegs + i), Xbyak::Zmm(CReg + mm * NRegs + i));
      }
    }
  }

  void generate_zp_correction(int _mtile) {
    load32(reg_tmp1, ptr[parambase + OFFSET(ldsb)]);
    imul(reg_tmp1, reg_iterkb);
    mov(reg_tmp2, ptr[parambase + OFFSET(reduceB)]);
    lea(reg_tmp2, ptr[reg_tmp2 + reg_tmp1 * sizeof(float)]);
    lea(reg_tmp2, ptr[reg_tmp2 + reg_itern * sizeof(float)]);
    auto& reg_redB = reg_tmp2;

    mov(reg_tmp, ptr[parambase + OFFSET(zpA)]);
    lea(reg_tmp, ptr[reg_tmp + reg_iterkb * sizeof(AType)]);
    auto& reg_zpA = reg_tmp;

    mov(reg_tmp1, ptr[parambase + OFFSET(scaleA)]);
    lea(reg_tmp1, ptr[reg_tmp1 + reg_iterkb * sizeof(float)]);
    auto& reg_scaleA = reg_tmp1;

    load32(reg_tmp3, ptr[parambase + OFFSET(ldsa)]);
    auto& reg_ldsa = reg_tmp3;
    for (int i = 0; i < NRegs; i++) {
      vmovups(Xbyak::Zmm(BReg + i), ptr[reg_redB + i * VecBytes]);
    }

    for (int i = 0; i < _mtile; i++) {
      vpbroadcastb(Xbyak::Xmm(AReg), ptr[reg_zpA]);
      vpmovzxbd(Xbyak::Zmm(AReg), Xbyak::Xmm(AReg));
      vcvtdq2ps(Xbyak::Zmm(AReg), Xbyak::Zmm(AReg));
      vmulps(Xbyak::Zmm(AReg), Xbyak::Zmm(AReg), zword_b[reg_scaleA]);
      for (int j = 0; j < NRegs; j++) {
        vmulps(Xbyak::Zmm(CReg + j), Xbyak::Zmm(AReg), Xbyak::Zmm(BReg + j));
        vsubps(Xbyak::Zmm(CF32Reg + i * NRegs + j), Xbyak::Zmm(CReg + j));
      }
      lea(reg_zpA, ptr[reg_zpA + reg_ldsa * sizeof(AType)]);
      lea(reg_scaleA, ptr[reg_scaleA + reg_ldsa * sizeof(float)]);
    }
  }

  void write_back(int _mtile) {
    inLocalLabel();
    mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
    load32(reg_cstride, ptr[parambase + OFFSET(cstride)]);
    lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
    for (int i = 0; i < _mtile; i++) {
      for (int j = 0; j < NRegs; j++) {
        vmovups(ptr[reg_matCptr + j * VecBytes], vreg_t(CF32Reg + i * NRegs + j));
      }
      add(reg_matCptr, reg_cstride);
    }
    outLocalLabel();
  }
};

}  // namespace kblock
}  // namespace code
template <template <int, int> class CodeT, int _NTILE, int _MTILE = 0>
class CoreCodeBase {
 public:
  using Code = CodeT<_NTILE, _MTILE>;
  using AType = typename Code::AType;
  using BType = typename Code::BType;
  using CType = typename Code::CType;
  static int constexpr NTILE = Code::NTILE;
  static int constexpr MTILE = Code::MTILE;
  static int constexpr KTILE = Code::KTILE;
  static int constexpr PACK_ROW = Code::PackRow;
  static int constexpr COMP = Code::COMPUTE;
  static int constexpr PREFERRED_N = NTILE * 3;
  static JBLAS_ISA constexpr ISA = (JBLAS_ISA)Code::ISA;
  static uint32_t constexpr ID = CoreAttr::make_core_id(NTILE, PACK_ROW, COMP, ISA);
  void configure() { (void)(0); }

 protected:
  CoreCodeBase() {
    for (int i = 0; i < mCodes.size(); i++) {
      mCodes[i].generate_code(i + 1);
    }
  }
  std::array<Code, Code::MTILE> mCodes;
};

template <template <int, int> class CodeT, int _NTILE, int _MTILE = 0>
class CoreCodeBaseAMX {
 public:
  using Code = CodeT<_NTILE, _MTILE>;
  using AType = typename Code::AType;
  using BType = typename Code::BType;
  using CType = typename Code::CType;
  static int constexpr NTILE = Code::NTILE;
  static int constexpr MTILE = Code::MTILE;
  static int constexpr KTILE = Code::KTILE;
  static int constexpr PACK_ROW = Code::PackRow;
  static int constexpr COMP = Code::COMPUTE;
  static int constexpr PREFERRED_N = NTILE * 3;
  static JBLAS_ISA constexpr ISA = (JBLAS_ISA)Code::ISA;
  static uint32_t constexpr ID = CoreAttr::make_core_id(_NTILE, PACK_ROW, COMP, ISA);
  Xbyak::CodeGenerator cfgcode;

 protected:
  CoreCodeBaseAMX() {
    for (int i = 0; i < mCodes.size(); i++) {
      mCodes[i].generate_code((i + 1) * 16);
    }
  }
  std::array<Code, Code::MRegs> mCodes;
};

template <int _NTILE, int _MTILE = 0>
class SCoreRowNAvx2 : public CoreCodeBase<code::Avx2N8P1, _NTILE, _MTILE> {
 public:
  using Code = typename CoreCodeBase<code::Avx2N8P1, _NTILE, _MTILE>::Code;
  void forward(float* matA, float* matB, float* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
               int kpos, void* tmpcache, size_t cachesize) {
    auto param = typename Code::params{matA, _astride, matB, _bstride, matC, _cstride, _k, _n, kpos == 0 ? 1 : 0};
    if (_m <= Code::MTILE) {
      this->mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class SCoreRowNAvx512f : public CoreCodeBase<code::Avx512fN16P1, _NTILE, _MTILE> {
 public:
  using Code = typename CoreCodeBase<code::Avx512fN16P1, _NTILE, _MTILE>::Code;
  void forward(float* matA, float* matB, float* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
               int kpos, void* tmpcache, size_t cachesize) {
    auto param = typename Code::params{matA, _astride, matB, _bstride, matC, _cstride, _k, _n, kpos == 0 ? 1 : 0};
    if (_m <= Code::MTILE) {
      this->mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class HCoreRowNAvx512fp16 : public CoreCodeBase<code::Avx512fp16N32P1, _NTILE, _MTILE> {
 public:
  using Code = typename CoreCodeBase<code::Avx512fp16N32P1, _NTILE, _MTILE>::Code;

  void forward(utils::fp16* matA, utils::fp16* matB, utils::fp16* matC, int _m, int _n, int _k, int _astride,
               int _bstride, int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param = typename Code::params{matA, _astride, matB, _bstride, matC, _cstride, _k, _n, kpos == 0 ? 1 : 0};
    if (_m <= Code::MTILE) {
      this->mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class HCoreRowNAvx512bf16 : public CoreCodeBase<code::Avx512bf16N16P2, _NTILE, _MTILE> {
 public:
  using Code = typename CoreCodeBase<code::Avx512bf16N16P2, _NTILE, _MTILE>::Code;
  void forward(utils::bf16* matA, utils::bf16* matB, float* matC, int _m, int _n, int _k, int _astride, int _bstride,
               int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param = typename Code::params{matA, _astride, matB, _bstride, matC, _cstride, _k, _n, kpos == 0 ? 1 : 0};
    if (_m <= Code::MTILE) {
      this->mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class HCoreRowNAmxbf16 : public CoreCodeBaseAMX<code::Amxbf16N16P2, _NTILE, _MTILE> {
 public:
  using Code = typename CoreCodeBaseAMX<code::Amxbf16N16P2, _NTILE, _MTILE>::Code;
  using AType = typename Code::AType;
  using BType = typename Code::BType;
  using CType = typename Code::CType;

  void configure() {
    code::AmxConfigure::configure(16, 16, Code::KTILE, sizeof(BType), this->mCodes[0].ATileCount,
                                  this->mCodes[0].BTileCount, this->mCodes[0].CTileCount);
  }

  void forward(AType* matA, BType* matB, CType* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
               int kpos, void* tmpcache, size_t cachesize) {
    auto param =
        typename Code::params{matA, _astride, matB, _bstride, matC, _cstride, _k, _n, kpos == 0 ? 1 : 0, tmpcache};
    if (_m <= Code::MTILE) {
      int idx = utils::updiv(_m, 16) - 1;
      this->mCodes[idx].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class ICoreRowNAvx512vnni : public CoreCodeBase<code::Avx512vnniN16P4, _NTILE, _MTILE> {
 public:
  using Code = typename CoreCodeBase<code::Avx512vnniN16P4, _NTILE, _MTILE>::Code;
  void forward(uint8_t* matA, int8_t* matB, int32_t* matC, int _m, int _n, int _k, int _astride, int _bstride,
               int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param = typename Code::params{matA, _astride, matB, _bstride, matC, _cstride, _k, _n, kpos == 0 ? 1 : 0};
    if (_m <= Code::MTILE) {
      this->mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class ICoreRowNAvx512vnniKBlock : public CoreCodeBase<code::kblock::Avx512vnniN16P4, _NTILE, _MTILE> {
 public:
  using Code = typename CoreCodeBase<code::kblock::Avx512vnniN16P4, _NTILE, _MTILE>::Code;
  void forward(uint8_t* matA, int8_t* matB, float* matC, uint8_t* zpA, float* scaleA, int _ldsa, float* scaleB,
               float* reduceB, int _ldsb, int _m, int _n, int _k, int _kblock, int _astride, int _bstride, int _cstride,
               int kpos, void* tmpcache, size_t cachesize) {
    auto param = typename Code::params{matA,  _astride, matB,    _bstride, matC, _cstride, zpA,     scaleA,
                                       _ldsa, scaleB,   reduceB, _ldsb,    _k,   _n,       _kblock, kpos == 0 ? 1 : 0};
    if (_m <= Code::MTILE) {
      this->mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class ICoreRowNAvxvnni : public CoreCodeBase<code::AvxvnniN8P4, _NTILE, _MTILE> {
 public:
  using Code = typename CoreCodeBase<code::AvxvnniN8P4, _NTILE, _MTILE>::Code;

  void forward(uint8_t* matA, int8_t* matB, int32_t* matC, int _m, int _n, int _k, int _astride, int _bstride,
               int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param = typename Code::params{matA, _astride, matB, _bstride, matC, _cstride, _k, _n, kpos == 0 ? 1 : 0};
    if (_m <= Code::MTILE) {
      this->mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class ICoreRowNAmxint8 : public CoreCodeBaseAMX<code::Amxint8N16P4US, _NTILE, _MTILE> {
 public:
  using Code = typename CoreCodeBaseAMX<code::Amxint8N16P4US, _NTILE, _MTILE>::Code;
  using AType = typename Code::AType;
  using BType = typename Code::BType;
  using CType = typename Code::CType;
  void configure() {
    code::AmxConfigure::configure(16, 16, Code::KTILE, sizeof(BType), this->mCodes[0].ATileCount,
                                  this->mCodes[0].BTileCount, this->mCodes[0].CTileCount);
  }

  void forward(uint8_t* matA, int8_t* matB, int32_t* matC, int _m, int _n, int _k, int _astride, int _bstride,
               int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param =
        typename Code::params{matA, _astride, matB, _bstride, matC, _cstride, _k, _n, kpos == 0 ? 1 : 0, tmpcache};
    if (_m <= Code::MTILE) {
      int idx = utils::updiv(_m, 16) - 1;
      this->mCodes[idx].mKernel(&param);
    } else {
      assert(0);
    }
  }
};

template <int _NTILE, int _MTILE = 0>
class ICoreRowNAmxint8SS : public CoreCodeBaseAMX<code::Amxint8N16P4SS, _NTILE, _MTILE> {
 public:
  using Code = typename CoreCodeBaseAMX<code::Amxint8N16P4SS, _NTILE, _MTILE>::Code;
  using AType = typename Code::AType;
  using BType = typename Code::BType;
  using CType = typename Code::CType;
  void configure() {
    code::AmxConfigure::configure(16, 16, Code::KTILE, sizeof(BType), this->mCodes[0].ATileCount,
                                  this->mCodes[0].BTileCount, this->mCodes[0].CTileCount);
  }

  void forward(int8_t* matA, int8_t* matB, int32_t* matC, int _m, int _n, int _k, int _astride, int _bstride,
               int _cstride, int kpos, void* tmpcache, size_t cachesize) {
    auto param =
        typename Code::params{matA, _astride, matB, _bstride, matC, _cstride, _k, _n, kpos == 0 ? 1 : 0, tmpcache};
    if (_m <= Code::MTILE) {
      int idx = utils::updiv(_m, 16) - 1;
      this->mCodes[idx].mKernel(&param);
    } else {
      assert(0);
    }
  }
};
}  // namespace gemm
}  // namespace jblas
