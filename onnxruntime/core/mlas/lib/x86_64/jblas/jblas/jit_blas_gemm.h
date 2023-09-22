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

#include "jit_base.hpp"
#include "jit_blas_utils.h"

namespace jblas {
namespace gemm {

enum class GemmCoreType : int {
  Undef = 0,
  AVX2_4X24,
  AVX2_2X48,
  AVX_VNNI_2x48,
  AVX_VNNI_1x48_KBLOCK,
  AVX512F_8x48,
  AVX512_VNNI_8x48,
  AMX_BF16_16x64,
  AMX_BF16_16x48,
  AMX_INT8_16x64,
  AMX_INT8_16x48,
  AVX512_VNNI_3x48_KBLOCK,
  AVX512_VNNI_4x48_KBLOCK,
  AMX_INT8_16x48_KBLOCK,
  AVX512_FP16_8x64,
  AVX512_FP16_8x96,
  AMX_INT8_16x48_SS,
};

class GemmCore_Row_NN_4x24_AVX2 {
 public:
  struct params {
    float *matA, *matB, *matC;
    int k, nsize;
    int astep, bstep, cstep;
    int kpos;
  };
  typedef long long (*func_t)(params*);
  typedef float AType;
  typedef float BType;
  typedef float CType;
  static JBLAS_ISA constexpr ISA = JblasAVX2;
  static GemmCoreType constexpr TYPE = GemmCoreType::AVX2_4X24;
  static int constexpr NTILE = 24, MTILE = 4, KTILE = 4 / sizeof(BType);
  static int constexpr KUNROLL = 2;
  static int constexpr PACK_ROW = 1;
  static int constexpr PREFERED_N = 144;
  class MicroKernel : protected jblas::xbyak::JitAvx2 {
   public:
    MicroKernel() {}
    static int constexpr VecBytes = 32;
    static int constexpr VecElements = VecBytes / sizeof(CType);
    int CRegCount = 12, BRegCount = 3, ARegCount = 1;
    int CReg = 0, BReg = 12, AReg = 15, TmpReg = BReg;
    int const NRegs = NTILE / VecElements;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);

    void generate_code(int _mtile) {
      reset();
      generate_mtile(_mtile);
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    void generate_mtile(int _mtile) {
      CRegCount = _mtile * NRegs;
      BRegCount = NRegs;
      BReg = CReg + CRegCount;
      AReg = BReg + BRegCount;
      TmpReg = AReg + ARegCount;
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_nsize = st.t[9];
      reg_cstep = st.t[3];
      reg_astep = st.t[5];
      reg_iterk = st.t[4];
      reg_itern = st.t[7];
      reg_tmp = st.t[6];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[10];
      reg_ret = rax;

      vreg_push(rsp);

      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_nsize, ptr[parambase + OFFSET(nsize)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      xor_(reg_itern, reg_itern);
      L(".nloop");
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vpxor(Xbyak::Ymm(CReg + i * NRegs + j), Xbyak::Ymm(CReg + i * NRegs + j), Xbyak::Ymm(CReg + i * NRegs + j));
        }
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_tmp1, reg_matBptr);

      xor_(reg_iterk, reg_iterk);

      mov(reg_tmp, reg_nsize);
      sub(reg_tmp, reg_itern);
      cmp(reg_tmp, NTILE);
      jl(".n16", T_NEAR);
      generate_kloop(_mtile, NRegs);
      write_back(_mtile, NRegs, parambase, reg_matCptr, reg_cstep, reg_itern);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n16");
      cmp(reg_tmp, 16);
      jl(".n8", T_NEAR);
      generate_kloop(_mtile, 2);
      write_back(_mtile, 2, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 16);
      add(reg_matBptr, 16 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n8");
      xor_(reg_iterk, reg_iterk);
      generate_kloop(_mtile, 1);
      write_back(_mtile, 1, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 8);
      add(reg_matBptr, 8 * sizeof(BType));
      L(".nend");
      cmp(reg_itern, reg_nsize);
      jb(".nloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _mtile, int _nregs) {
      inLocalLabel();
      L(".kloop");
      mov(reg_tmp, reg_ksize);
      sub(reg_tmp, reg_iterk);
      cmp(reg_tmp, KUNROLL * KTILE);
      jl(".k1loop", T_NEAR);
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_mtile, _nregs, 1, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _ktile, const Xbyak::Reg64& reg_tmp, const Xbyak::Reg64& reg_matAptr,
                      const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      for (int kk = 0; kk < _ktile; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Ymm(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vbroadcastss(Xbyak::Ymm(AReg), ptr[reg_tmp]);
          add(reg_tmp, reg_astep);
          for (int i = 0; i < _NRegs; i++) {
            vfmadd231ps(Xbyak::Ymm(CReg + mm * NRegs + i), Xbyak::Ymm(BReg + i), Xbyak::Ymm(AReg));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs, const Xbyak::Reg64& parambase, const Xbyak::Reg64& reg_matCptr,
                    const Xbyak::Reg64& reg_cstep, const Xbyak::Reg64& reg_itern) {
      inLocalLabel();
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".LACC", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Ymm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      jmp(".LEND", T_NEAR);
      L(".LACC");
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vaddps(Xbyak::Ymm(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Ymm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      L(".LEND");
      nop();
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_nsize;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_itern;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_ret = rax;
    Xbyak::Opmask msk_wr = k1;
  };

 public:
  GemmCore_Row_NN_4x24_AVX2() {
    for (int i = 0; i < MTILE; i++) {
      mCodes[i].generate_code(i + 1);
    }
  }

  void forward(float* matA, float* matB, float* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
               int kpos) {
    auto param = params{matA, matB, matC, _k, _n, _astride, _bstride, _cstride, kpos};
    if (_m <= MTILE) {
      mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }

 private:
  std::array<MicroKernel, MTILE> mCodes;
};

class GemmCore_Row_NN_2x48_AVX2 {
 public:
  struct params {
    float *matA, *matB, *matC;
    int k, nsize;
    int astep, bstep, cstep;
    int kpos;
  };
  typedef long long (*func_t)(params*);
  typedef float AType;
  typedef float BType;
  typedef float CType;
  static JBLAS_ISA constexpr ISA = JblasAVX2;
  static GemmCoreType constexpr TYPE = GemmCoreType::AVX2_2X48;
  static int constexpr NTILE = 48, MTILE = 2, KTILE = 4 / sizeof(BType);
  static int constexpr KUNROLL = 2;
  static int constexpr PACK_ROW = 1;
  static int constexpr PREFERED_N = 144;
  class MicroKernel : protected jblas::xbyak::JitAvx2 {
   public:
    MicroKernel() {}
    static int constexpr VecBytes = 32;
    static int constexpr VecElements = VecBytes / sizeof(CType);
    int CRegCount = 12, BRegCount = 1, ARegCount = 2;
    int CReg = 0, BReg = 12, AReg = 13, TmpReg = BReg;
    int const NRegs = NTILE / VecElements;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);

    void generate_code(int _mtile) {
      reset();
      generate_mtile(_mtile);
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    void generate_mtile(int _mtile) {
      CRegCount = _mtile * NRegs;
      BRegCount = 1;
      ARegCount = _mtile;
      BReg = CReg + CRegCount;
      AReg = BReg + BRegCount;
      TmpReg = AReg + ARegCount;
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_nsize = st.t[9];
      reg_cstep = st.t[3];
      reg_astep = st.t[5];
      reg_iterk = st.t[4];
      reg_itern = st.t[7];
      reg_tmp = st.t[6];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[10];
      reg_ret = rax;

      vreg_push(rsp);

      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_nsize, ptr[parambase + OFFSET(nsize)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      xor_(reg_itern, reg_itern);
      L(".nloop");
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vpxor(Xbyak::Ymm(CReg + i * NRegs + j), Xbyak::Ymm(CReg + i * NRegs + j), Xbyak::Ymm(CReg + i * NRegs + j));
        }
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_tmp1, reg_matBptr);

      xor_(reg_iterk, reg_iterk);

      generate_kloop(_mtile, NRegs);
      write_back(_mtile, NRegs, parambase, reg_matCptr, reg_cstep, reg_itern);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);

      cmp(reg_itern, reg_nsize);
      jb(".nloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _mtile, int _nregs) {
      inLocalLabel();
      L(".kloop");
      mov(reg_tmp, reg_ksize);
      sub(reg_tmp, reg_iterk);
      cmp(reg_tmp, KUNROLL * KTILE);
      jl(".k1loop", T_NEAR);
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_mtile, _nregs, 1, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _ktile, const Xbyak::Reg64& reg_tmp, const Xbyak::Reg64& reg_matAptr,
                      const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      for (int kk = 0; kk < _ktile; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _mtile; ++i)
          vbroadcastss(Xbyak::Ymm(AReg + i), ptr[reg_matAptr + kk * AKStepSize + reg_astep * i]);
        for (int j = 0; j < _NRegs; j++) {
          vmovups(Xbyak::Ymm(BReg), ptr[reg_matBptr + kk * BKStepSize + j * VecBytes]);
          for (int i = 0; i < _mtile; ++i)
            vfmadd231ps(Xbyak::Ymm(CReg + i * NRegs + j), Xbyak::Ymm(BReg), Xbyak::Ymm(AReg + i));
        }
      }
    }

    void write_back(int _mtile, int _NRegs, const Xbyak::Reg64& parambase, const Xbyak::Reg64& reg_matCptr,
                    const Xbyak::Reg64& reg_cstep, const Xbyak::Reg64& reg_itern) {
      inLocalLabel();
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".LACC", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Ymm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      jmp(".LEND", T_NEAR);
      L(".LACC");
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vaddps(Xbyak::Ymm(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Ymm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      L(".LEND");
      nop();
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_nsize;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_itern;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_ret = rax;
    Xbyak::Opmask msk_wr = k1;
  };

 public:
  GemmCore_Row_NN_2x48_AVX2() {
    for (int i = 0; i < MTILE; i++) {
      mCodes[i].generate_code(i + 1);
    }
  }

  void forward(float* matA, float* matB, float* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
               int kpos) {
    auto param = params{matA, matB, matC, _k, _n, _astride, _bstride, _cstride, kpos};
    if (_m <= MTILE) {
      mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }

 private:
  std::array<MicroKernel, MTILE> mCodes;
};

class GemmCore_Row_NN_4x24_AVX_VNNI {
 public:
  struct params {
    uint8_t* matA;
    int8_t* matB;
    int32_t* matC;
    int k, nsize;
    int astep, bstep, cstep;
    int kpos;
  };
  typedef long long (*func_t)(params*);
  typedef uint8_t AType;
  typedef int8_t BType;
  typedef int32_t CType;
  static JBLAS_ISA constexpr ISA = JblasAVX_VNNI;
  static GemmCoreType constexpr TYPE = GemmCoreType::AVX_VNNI_2x48;
  static int constexpr NTILE = 24, MTILE = 4, KTILE = 4 / sizeof(BType);
  static int constexpr PACK_ROW = KTILE;
  static int constexpr KUNROLL = 2;
  static int constexpr PREFERED_N = 192;

  class MicroKernel : protected jblas::xbyak::JitAvxvnni {
   public:
    MicroKernel() {}
    int CRegCount = 12, BRegCount = 1, ARegCount = 1;
    int CReg = 0, BReg = 12, AReg = 13, TmpReg = 14;
    int const NRegs = 3;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);
    static int constexpr VecBytes = 32;

    void generate_code(int _mtile) {
      reset();
      generate_mtile(_mtile);
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    void generate_mtile(int _mtile) {
      CRegCount = _mtile * NRegs;
      BRegCount = NRegs;
      BReg = CReg + CRegCount;
      AReg = BReg + BRegCount;
      TmpReg = AReg + ARegCount;
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_nsize = st.t[9];
      reg_cstep = st.t[3];
      reg_astep = st.t[5];
      reg_iterk = st.t[4];
      reg_itern = st.t[7];
      reg_tmp = st.t[6];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[10];
      reg_ret = rax;

      vreg_push(rsp);

      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_nsize, ptr[parambase + OFFSET(nsize)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      xor_(reg_itern, reg_itern);
      L(".nloop");
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vxorps(Xbyak::Ymm(CReg + i * NRegs + j), Xbyak::Ymm(CReg + i * NRegs + j), Xbyak::Ymm(CReg + i * NRegs + j));
        }
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_tmp1, reg_matBptr);

      xor_(reg_iterk, reg_iterk);

      mov(reg_tmp, reg_nsize);
      sub(reg_tmp, reg_itern);
      cmp(reg_tmp, NTILE);

      generate_kloop(_mtile, NRegs);
      write_back(_mtile, NRegs, parambase, reg_matCptr, reg_cstep, reg_itern);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);

      cmp(reg_itern, reg_nsize);
      jb(".nloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _mtile, int _nregs) {
      inLocalLabel();
      L(".kloop");
      mov(reg_tmp, reg_ksize);
      sub(reg_tmp, reg_iterk);
      cmp(reg_tmp, KTILE * KUNROLL);
      jl(".k1loop", T_NEAR);
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, AKStepSize * KUNROLL);
      add(reg_tmp1, BKStepSize * KUNROLL);
      add(reg_iterk, KTILE * KUNROLL);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_mtile, _nregs, 1, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _kunroll, const Xbyak::Reg64& reg_tmp,
                      const Xbyak::Reg64& reg_matAptr, const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      for (int kk = 0; kk < _kunroll; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Ymm(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastd(Xbyak::Ymm(AReg), ptr[reg_tmp]);
          add(reg_tmp, reg_astep);
          for (int i = 0; i < _NRegs; i++) {
            vpdpbusds(Xbyak::Ymm(CReg + mm * NRegs + i), Xbyak::Ymm(AReg), Xbyak::Ymm(BReg + i));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs, const Xbyak::Reg64& parambase, const Xbyak::Reg64& reg_matCptr,
                    const Xbyak::Reg64& reg_cstep, const Xbyak::Reg64& reg_itern) {
      inLocalLabel();
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".LACC", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Ymm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      jmp(".LEND", T_NEAR);
      L(".LACC");
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vpaddd(Xbyak::Ymm(CReg + i * NRegs + j), Xbyak::Ymm(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Ymm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      L(".LEND");
      nop();
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_nsize;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_itern;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_ret = rax;
  };

 public:
  GemmCore_Row_NN_4x24_AVX_VNNI() {
    for (int i = 0; i < MTILE; i++) {
      mCodes[i].generate_code(i + 1);
    }
  }

  void forward(AType* matA, BType* matB, CType* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
               int kpos) {
    auto param = params{matA, matB, matC, _k, _n, _astride, _bstride, _cstride, kpos};
    if (_m <= MTILE) {
      mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }

 private:
  std::array<MicroKernel, MTILE> mCodes;
};

class GemmCore_Row_NN_2x48_AVX_VNNI {
 public:
  struct params {
    uint8_t* matA;
    int8_t* matB;
    int32_t* matC;
    int k, nsize;
    int astep, bstep, cstep;
    int kpos;
  };
  typedef long long (*func_t)(params*);
  typedef uint8_t AType;
  typedef int8_t BType;
  typedef int32_t CType;
  static JBLAS_ISA constexpr ISA = JblasAVX_VNNI;
  static GemmCoreType constexpr TYPE = GemmCoreType::AVX_VNNI_2x48;
  static int constexpr NTILE = 48, MTILE = 2, KTILE = 4 / sizeof(BType);
  static int constexpr PACK_ROW = KTILE;
  static int constexpr KUNROLL = 2;
  static int constexpr PREFERED_N = 192;

  class MicroKernel : protected jblas::xbyak::JitAvxvnni {
   public:
    MicroKernel() {}
    int CRegCount = 12, BRegCount = 1, ARegCount = 1;
    int CReg = 0, BReg = 12, AReg = 13, TmpReg = 14;
    int const NRegs = 6;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);
    static int constexpr VecBytes = 32;

    void generate_code(int _mtile) {
      reset();
      generate_mtile(_mtile);
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    void generate_mtile(int _mtile) {
      CRegCount = _mtile * NRegs;
      BRegCount = NRegs;
      BReg = CReg + CRegCount;
      AReg = BReg + BRegCount;
      TmpReg = AReg + ARegCount;
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_nsize = st.t[9];
      reg_cstep = st.t[3];
      reg_astep = st.t[5];
      reg_iterk = st.t[4];
      reg_itern = st.t[7];
      reg_tmp = st.t[6];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[10];
      reg_ret = rax;

      vreg_push(rsp);

      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_nsize, ptr[parambase + OFFSET(nsize)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      xor_(reg_itern, reg_itern);
      L(".nloop");
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vxorps(Xbyak::Ymm(CReg + i * NRegs + j), Xbyak::Ymm(CReg + i * NRegs + j), Xbyak::Ymm(CReg + i * NRegs + j));
        }
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_tmp1, reg_matBptr);

      xor_(reg_iterk, reg_iterk);

      mov(reg_tmp, reg_nsize);
      sub(reg_tmp, reg_itern);
      cmp(reg_tmp, NTILE);

      generate_kloop(_mtile, NRegs);
      write_back(_mtile, NRegs, parambase, reg_matCptr, reg_cstep, reg_itern);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);

      cmp(reg_itern, reg_nsize);
      jb(".nloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _mtile, int _nregs) {
      inLocalLabel();
      L(".kloop");
      mov(reg_tmp, reg_ksize);
      sub(reg_tmp, reg_iterk);
      cmp(reg_tmp, KTILE * KUNROLL);
      jl(".k1loop", T_NEAR);
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, AKStepSize * KUNROLL);
      add(reg_tmp1, BKStepSize * KUNROLL);
      add(reg_iterk, KTILE * KUNROLL);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_mtile, _nregs, 1, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _kunroll, const Xbyak::Reg64& reg_tmp,
                      const Xbyak::Reg64& reg_matAptr, const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      for (int kk = 0; kk < _kunroll; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Ymm(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastd(Xbyak::Ymm(AReg), ptr[reg_tmp]);
          add(reg_tmp, reg_astep);
          for (int i = 0; i < _NRegs; i++) {
            vpdpbusds(Xbyak::Ymm(CReg + mm * NRegs + i), Xbyak::Ymm(AReg), Xbyak::Ymm(BReg + i));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs, const Xbyak::Reg64& parambase, const Xbyak::Reg64& reg_matCptr,
                    const Xbyak::Reg64& reg_cstep, const Xbyak::Reg64& reg_itern) {
      inLocalLabel();
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".LACC", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Ymm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      jmp(".LEND", T_NEAR);
      L(".LACC");
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vpaddd(Xbyak::Ymm(CReg + i * NRegs + j), Xbyak::Ymm(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Ymm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      L(".LEND");
      nop();
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_nsize;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_itern;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_ret = rax;
  };

 public:
  GemmCore_Row_NN_2x48_AVX_VNNI() {
    for (int i = 0; i < MTILE; i++) {
      mCodes[i].generate_code(i + 1);
    }
  }

  void forward(AType* matA, BType* matB, CType* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
               int kpos) {
    auto param = params{matA, matB, matC, _k, _n, _astride, _bstride, _cstride, kpos};
    if (_m <= MTILE) {
      mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }

 private:
  std::array<MicroKernel, MTILE> mCodes;
};

class GemmCore_Row_NN_8x48_AVX512F {
 public:
  struct params {
    float *matA, *matB, *matC;
    int k, nsize;
    int astep, bstep, cstep;
    int kpos;
  };
  typedef long long (*func_t)(params*);
  typedef float AType;
  typedef float BType;
  typedef float CType;
  static JBLAS_ISA constexpr ISA = JblasAVX512F;
  static GemmCoreType constexpr TYPE = GemmCoreType::AVX512F_8x48;
  static int constexpr NTILE = 48, MTILE = 8, KTILE = 4 / sizeof(BType);
  static int constexpr KUNROLL = 2;
  static int constexpr PACK_ROW = 1;
  static int constexpr PREFERED_N = 144;
  class MicroKernel : protected jblas::xbyak::JitAvx512f {
   public:
    MicroKernel() {}
    int CRegCount = 24, BRegCount = 6, ARegCount = 1;
    int CReg = 0, BReg = 24, AReg = 27, TmpReg = 28;
    int const NRegs = 3;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);
    static int constexpr VecBytes = 64;

    void generate_code(int _mtile) {
      reset();
      generate_mtile(_mtile);
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    void generate_mtile(int _mtile) {
      CRegCount = _mtile * NRegs;
      BRegCount = NRegs;
      BReg = CReg + CRegCount;
      AReg = BReg + BRegCount;
      TmpReg = AReg + ARegCount;
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_nsize = st.t[9];
      reg_cstep = st.t[3];
      reg_astep = st.t[5];
      reg_iterk = st.t[4];
      reg_itern = st.t[7];
      reg_tmp = st.t[6];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[10];
      reg_ret = rax;

      vreg_push(rsp);

      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_nsize, ptr[parambase + OFFSET(nsize)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      xor_(reg_itern, reg_itern);
      L(".nloop");
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vpxorq(Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j));
        }
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_tmp1, reg_matBptr);

      xor_(reg_iterk, reg_iterk);

      mov(reg_tmp, reg_nsize);
      sub(reg_tmp, reg_itern);
      cmp(reg_tmp, NTILE);
      jl(".n32", T_NEAR);
      generate_kloop(_mtile, NRegs);
      write_back(_mtile, NRegs, parambase, reg_matCptr, reg_cstep, reg_itern);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n32");
      cmp(reg_tmp, 32);
      jl(".n16", T_NEAR);
      generate_kloop(_mtile, 2);
      write_back(_mtile, 2, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 32);
      add(reg_matBptr, 32 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n16");
      xor_(reg_iterk, reg_iterk);
      generate_kloop(_mtile, 1);
      write_back(_mtile, 1, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 16);
      add(reg_matBptr, 16 * sizeof(BType));
      L(".nend");
      cmp(reg_itern, reg_nsize);
      jb(".nloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _mtile, int _nregs) {
      inLocalLabel();
      L(".kloop");
      mov(reg_tmp, reg_ksize);
      sub(reg_tmp, reg_iterk);
      cmp(reg_tmp, KUNROLL * KTILE);
      jl(".k1loop", T_NEAR);
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_mtile, _nregs, 1, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _ktile, const Xbyak::Reg64& reg_tmp, const Xbyak::Reg64& reg_matAptr,
                      const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      for (int kk = 0; kk < _ktile; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Zmm(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vbroadcastss(Xbyak::Zmm(AReg), ptr[reg_tmp]);
          add(reg_tmp, reg_astep);
          for (int i = 0; i < _NRegs; i++) {
            vfmadd231ps(Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(BReg + i), Xbyak::Zmm(AReg));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs, const Xbyak::Reg64& parambase, const Xbyak::Reg64& reg_matCptr,
                    const Xbyak::Reg64& reg_cstep, const Xbyak::Reg64& reg_itern) {
      inLocalLabel();
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".LACC", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      jmp(".LEND", T_NEAR);
      L(".LACC");
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vaddps(Xbyak::Zmm(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      L(".LEND");
      nop();
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_nsize;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_itern;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_ret = rax;
    Xbyak::Opmask msk_wr = k1;
  };

 public:
  GemmCore_Row_NN_8x48_AVX512F() {
    for (int i = 0; i < MTILE; i++) {
      mCodes[i].generate_code(i + 1);
    }
  }

  void forward(float* matA, float* matB, float* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
               int kpos) {
    auto param = params{matA, matB, matC, _k, _n, _astride, _bstride, _cstride, kpos};
    if (_m <= MTILE) {
      mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }

 private:
  std::array<MicroKernel, MTILE> mCodes;
};

class GemmCore_Row_NN_8x64_AVX512_FP16 {
 public:
  typedef utils::fp16 AType;
  typedef utils::fp16 BType;
  typedef utils::fp16 CType;
  struct params {
    AType* matA;
    BType* matB;
    CType* matC;
    int k, nsize;
    int astep, bstep, cstep;
    int kpos;
  };
  typedef long long (*func_t)(params*);

  static JBLAS_ISA constexpr ISA = JblasAVX512_FP16;
  static GemmCoreType constexpr TYPE = GemmCoreType::AVX512_FP16_8x64;
  static int constexpr NTILE = 64, MTILE = 12, KTILE = 1;
  static int constexpr KUNROLL = 2;
  static int constexpr PACK_ROW = 1;
  static int constexpr PREFERED_N = 128;
  class MicroKernel : protected jblas::xbyak::JitAvx512_fp16 {
   public:
    MicroKernel() {}
    int CRegCount = 24, BRegCount = 2, ARegCount = 1;
    int CReg = 0, BReg = 24, AReg = 26, TmpReg = 27;
    int const NRegs = 2;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);
    static int constexpr VecBytes = 64;

    void generate_code(int _mtile) {
      reset();
      generate_mtile(_mtile);
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    void generate_mtile(int _mtile) {
      CRegCount = _mtile * NRegs;
      BRegCount = NRegs;
      BReg = CReg + CRegCount;
      AReg = BReg + BRegCount;
      TmpReg = AReg + ARegCount;
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_nsize = st.t[9];
      reg_cstep = st.t[3];
      reg_astep = st.t[5];
      reg_iterk = st.t[4];
      reg_itern = st.t[7];
      reg_tmp = st.t[6];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[10];
      reg_ret = rax;

      vreg_push(rsp);

      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_nsize, ptr[parambase + OFFSET(nsize)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      xor_(reg_itern, reg_itern);
      L(".nloop");
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vpxorq(Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j));
        }
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_tmp1, reg_matBptr);

      xor_(reg_iterk, reg_iterk);

      mov(reg_tmp, reg_nsize);
      sub(reg_tmp, reg_itern);
      cmp(reg_tmp, NTILE);
      jl(".n32", T_NEAR);
      generate_kloop(_mtile, NRegs);
      write_back(_mtile, NRegs, parambase, reg_matCptr, reg_cstep, reg_itern);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n32");
      generate_kloop(_mtile, 1);
      write_back(_mtile, 1, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 32);
      add(reg_matBptr, 32 * sizeof(BType));

      L(".nend");
      cmp(reg_itern, reg_nsize);
      jb(".nloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _mtile, int _nregs) {
      inLocalLabel();
      L(".kloop");
      mov(reg_tmp, reg_ksize);
      sub(reg_tmp, reg_iterk);
      cmp(reg_tmp, KUNROLL * KTILE);
      jl(".k1loop", T_NEAR);
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_mtile, _nregs, 1, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _ktile, const Xbyak::Reg64& reg_tmp, const Xbyak::Reg64& reg_matAptr,
                      const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      for (int kk = 0; kk < _ktile; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Zmm(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastw(Xbyak::Zmm(AReg), ptr[reg_tmp]);
          add(reg_tmp, reg_astep);
          for (int i = 0; i < _NRegs; i++) {
            vfmadd231ph(Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(BReg + i), Xbyak::Zmm(AReg));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs, const Xbyak::Reg64& parambase, const Xbyak::Reg64& reg_matCptr,
                    const Xbyak::Reg64& reg_cstep, const Xbyak::Reg64& reg_itern) {
      inLocalLabel();
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".LACC", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      jmp(".LEND", T_NEAR);
      L(".LACC");
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vaddph(Xbyak::Zmm(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      L(".LEND");
      nop();
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_nsize;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_itern;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_ret = rax;
    Xbyak::Opmask msk_wr = k1;
  };

 public:
  GemmCore_Row_NN_8x64_AVX512_FP16() {
    for (int i = 0; i < MTILE; i++) {
      mCodes[i].generate_code(i + 1);
    }
  }

  void forward(utils::fp16* matA, utils::fp16* matB, utils::fp16* matC, int _m, int _n, int _k, int _astride,
               int _bstride, int _cstride, int kpos) {
    auto param = params{matA, matB, matC, _k, _n, _astride, _bstride, _cstride, kpos};
    if (_m <= MTILE) {
      mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }

 private:
  std::array<MicroKernel, MTILE> mCodes;
};

class GemmCore_Row_NN_8x96_AVX512_FP16 {
 public:
  typedef utils::fp16 AType;
  typedef utils::fp16 BType;
  typedef utils::fp16 CType;
  struct params {
    AType* matA;
    BType* matB;
    CType* matC;
    int k, nsize;
    int astep, bstep, cstep;
    int kpos;
  };
  typedef long long (*func_t)(params*);

  static JBLAS_ISA constexpr ISA = JblasAVX512_FP16;
  static GemmCoreType constexpr TYPE = GemmCoreType::AVX512_FP16_8x96;
  static int constexpr NTILE = 96, MTILE = 8, KTILE = 1;
  static int constexpr KUNROLL = 2;
  static int constexpr PACK_ROW = 1;
  static int constexpr PREFERED_N = 192;
  class MicroKernel : protected jblas::xbyak::JitAvx512_fp16 {
   public:
    MicroKernel() {}
    int CRegCount = 24, BRegCount = 3, ARegCount = 1;
    int CReg = 0, BReg = 24, AReg = 27, TmpReg = 28;
    int const NRegs = 3;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);
    static int constexpr VecBytes = 64;

    void generate_code(int _mtile) {
      reset();
      generate_mtile(_mtile);
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    void generate_mtile(int _mtile) {
      CRegCount = _mtile * NRegs;
      BRegCount = NRegs;
      BReg = CReg + CRegCount;
      AReg = BReg + BRegCount;
      TmpReg = AReg + ARegCount;
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_nsize = st.t[9];
      reg_cstep = st.t[3];
      reg_astep = st.t[5];
      reg_iterk = st.t[4];
      reg_itern = st.t[7];
      reg_tmp = st.t[6];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[10];
      reg_ret = rax;

      vreg_push(rsp);

      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_nsize, ptr[parambase + OFFSET(nsize)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      xor_(reg_itern, reg_itern);
      L(".nloop");
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vpxorq(Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j));
        }
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_tmp1, reg_matBptr);

      xor_(reg_iterk, reg_iterk);

      mov(reg_tmp, reg_nsize);
      sub(reg_tmp, reg_itern);
      cmp(reg_tmp, NTILE);
      jl(".n64", T_NEAR);
      generate_kloop(_mtile, NRegs);
      write_back(_mtile, NRegs, parambase, reg_matCptr, reg_cstep, reg_itern);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n64");
      cmp(reg_tmp, 64);
      jl(".n32", T_NEAR);
      generate_kloop(_mtile, 2);
      write_back(_mtile, 2, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 64);
      add(reg_matBptr, 64 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n32");
      generate_kloop(_mtile, 1);
      write_back(_mtile, 1, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 32);
      add(reg_matBptr, 32 * sizeof(BType));

      L(".nend");
      cmp(reg_itern, reg_nsize);
      jb(".nloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _mtile, int _nregs) {
      inLocalLabel();
      L(".kloop");
      mov(reg_tmp, reg_ksize);
      sub(reg_tmp, reg_iterk);
      cmp(reg_tmp, KUNROLL * KTILE);
      jl(".k1loop", T_NEAR);
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_mtile, _nregs, 1, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _ktile, const Xbyak::Reg64& reg_tmp, const Xbyak::Reg64& reg_matAptr,
                      const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      for (int kk = 0; kk < _ktile; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Zmm(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastw(Xbyak::Zmm(AReg), ptr[reg_tmp]);
          add(reg_tmp, reg_astep);
          for (int i = 0; i < _NRegs; i++) {
            vfmadd231ph(Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(BReg + i), Xbyak::Zmm(AReg));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs, const Xbyak::Reg64& parambase, const Xbyak::Reg64& reg_matCptr,
                    const Xbyak::Reg64& reg_cstep, const Xbyak::Reg64& reg_itern) {
      inLocalLabel();
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".LACC", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      jmp(".LEND", T_NEAR);
      L(".LACC");
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vaddph(Xbyak::Zmm(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      L(".LEND");
      nop();
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_nsize;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_itern;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_ret = rax;
    Xbyak::Opmask msk_wr = k1;
  };

 public:
  GemmCore_Row_NN_8x96_AVX512_FP16() {
    for (int i = 0; i < MTILE; i++) {
      mCodes[i].generate_code(i + 1);
    }
  }

  void forward(utils::fp16* matA, utils::fp16* matB, utils::fp16* matC, int _m, int _n, int _k, int _astride,
               int _bstride, int _cstride, int kpos) {
    auto param = params{matA, matB, matC, _k, _n, _astride, _bstride, _cstride, kpos};
    if (_m <= MTILE) {
      mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }

 private:
  std::array<MicroKernel, MTILE> mCodes;
};

class GemmCore_Row_NN_8x48_AVX512_VNNI {
 public:
  struct params {
    uint8_t* matA;
    int8_t* matB;
    int32_t* matC;
    int k, nsize;
    int astep, bstep, cstep;
    int kpos;
  };
  typedef long long (*func_t)(params*);
  typedef uint8_t AType;
  typedef int8_t BType;
  typedef int32_t CType;
  static JBLAS_ISA constexpr ISA = JblasAVX512_VNNI;
  static GemmCoreType constexpr TYPE = GemmCoreType::AVX512_VNNI_8x48;
  static int constexpr NTILE = 48, MTILE = 8, KTILE = 4 / sizeof(BType);
  static int constexpr PACK_ROW = KTILE;
  static int constexpr KUNROLL = 2;
  static int constexpr PREFERED_N = 192;

  class MicroKernel : protected jblas::xbyak::JitAvx512vnni {
   public:
    MicroKernel() {}
    int CRegCount = 24, BRegCount = 6, ARegCount = 1;
    int CReg = 0, BReg = 24, AReg = 27, TmpReg = 28;
    int const NRegs = 3;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);
    static int constexpr VecBytes = 64;

    void generate_code(int _mtile) {
      reset();
      generate_mtile(_mtile);
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    void generate_mtile(int _mtile) {
      CRegCount = _mtile * NRegs;
      BRegCount = NRegs;
      BReg = CReg + CRegCount;
      AReg = BReg + BRegCount;
      TmpReg = AReg + ARegCount;
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_nsize = st.t[9];
      reg_cstep = st.t[3];
      reg_astep = st.t[5];
      reg_iterk = st.t[4];
      reg_itern = st.t[7];
      reg_tmp = st.t[6];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[10];
      reg_ret = rax;

      vreg_push(rsp);

      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_nsize, ptr[parambase + OFFSET(nsize)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      xor_(reg_itern, reg_itern);
      L(".nloop");
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vpxorq(Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j));
        }
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_tmp1, reg_matBptr);

      xor_(reg_iterk, reg_iterk);

      mov(reg_tmp, reg_nsize);
      sub(reg_tmp, reg_itern);
      cmp(reg_tmp, NTILE);
      jl(".n32", T_NEAR);
      generate_kloop(_mtile, NRegs);
      write_back(_mtile, NRegs, parambase, reg_matCptr, reg_cstep, reg_itern);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n32");
      cmp(reg_tmp, 32);
      jl(".n16", T_NEAR);
      generate_kloop(_mtile, 2);
      write_back(_mtile, 2, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 32);
      add(reg_matBptr, 32 * sizeof(int8_t) * 4);
      jmp(".nend", T_NEAR);

      L(".n16");
      xor_(reg_iterk, reg_iterk);
      generate_kloop(_mtile, 1);
      write_back(_mtile, 1, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 16);
      add(reg_matBptr, 16 * sizeof(int8_t) * 4);
      L(".nend");
      cmp(reg_itern, reg_nsize);
      jb(".nloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _mtile, int _nregs) {
      inLocalLabel();
      L(".kloop");
      mov(reg_tmp, reg_ksize);
      sub(reg_tmp, reg_iterk);
      cmp(reg_tmp, KTILE * KUNROLL);
      jl(".k1loop", T_NEAR);
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, AKStepSize * KUNROLL);
      add(reg_tmp1, BKStepSize * KUNROLL);
      add(reg_iterk, KTILE * KUNROLL);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_mtile, _nregs, 1, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _kunroll, const Xbyak::Reg64& reg_tmp,
                      const Xbyak::Reg64& reg_matAptr, const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      for (int kk = 0; kk < _kunroll; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Zmm(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastd(Xbyak::Zmm(AReg), ptr[reg_tmp]);
          add(reg_tmp, reg_astep);
          for (int i = 0; i < _NRegs; i++) {
            vpdpbusds(Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(AReg), Xbyak::Zmm(BReg + i));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs, const Xbyak::Reg64& parambase, const Xbyak::Reg64& reg_matCptr,
                    const Xbyak::Reg64& reg_cstep, const Xbyak::Reg64& reg_itern) {
      inLocalLabel();
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".LACC", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      jmp(".LEND", T_NEAR);
      L(".LACC");
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vpaddd(Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j), ptr[reg_matCptr + j * VecBytes]);
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      L(".LEND");
      nop();
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_nsize;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_itern;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_ret = rax;
  };

 public:
  GemmCore_Row_NN_8x48_AVX512_VNNI() {
    for (int i = 0; i < MTILE; i++) {
      mCodes[i].generate_code(i + 1);
    }
  }

  void forward(AType* matA, BType* matB, CType* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
               int kpos) {
    auto param = params{matA, matB, matC, _k, _n, _astride, _bstride, _cstride, kpos};
    if (_m <= MTILE) {
      mCodes[_m - 1].mKernel(&param);
    } else {
      assert(0);
    }
  }

 private:
  std::array<MicroKernel, MTILE> mCodes;
};

class GemmCore_Row_NN_16x64_AMX_BF16 {
 public:
  typedef utils::bf16 AType;
  typedef utils::bf16 BType;
  typedef float CType;
  struct params {
    AType* matA;
    BType* matB;
    CType* matC;
    int k, msize, nsize;
    int astep, bstep, cstep;
    int kpos;
    void *workspace, *cfg;
  };
  typedef long long (*func_t)(params*);

  static JBLAS_ISA constexpr ISA = JblasAMX_BF16;
  static GemmCoreType constexpr TYPE = GemmCoreType::AMX_BF16_16x64;
  static int constexpr NTILE = 64, MTILE = 16, KTILE = 64 / sizeof(BType);
  static int constexpr PACK_ROW = 2;
  static int constexpr KUNROLL = 2;
  static int constexpr PREFERED_N = 256;
  class MicroKernel : protected jblas::xbyak::JitAmxbf16 {
   public:
    friend GemmCore_Row_NN_16x64_AMX_BF16;
    MicroKernel() {}
    static int constexpr CReg = 0, TmpReg = 4;
    static int constexpr NRegs = 4;
    static int constexpr CRegCount = NRegs;
    static int constexpr C_tilenum = 4, A_tilenum = 1, B_tilenum = 3;
    static int constexpr CTile = 0, ATile = CTile + C_tilenum, BTile = ATile + A_tilenum;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);
    static int constexpr VecBytes = 64;

    void generate_code() {
      reset();
      generate_mtile();
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    void generate_mtile() {
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_nsize = st.t[9];
      reg_cstep = st.t[3];
      reg_astep = st.t[5];
      reg_iterk = st.t[4];
      reg_itern = st.t[7];
      reg_tmp = st.t[6];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[10];
      reg_ret = rax;

      vreg_push(rsp);
      mov(reg_tmp, ptr[parambase + OFFSET(cfg)]);
      ldtilecfg(ptr[reg_tmp]);

      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_nsize, ptr[parambase + OFFSET(nsize)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      xor_(reg_itern, reg_itern);
      L(".nloop");
      for (int i = 0; i < C_tilenum; i++) {
        tilezero(Xbyak::Tmm(CTile + i));
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_tmp1, reg_matBptr);

      xor_(reg_iterk, reg_iterk);

      mov(reg_tmp, reg_nsize);
      sub(reg_tmp, reg_itern);
      cmp(reg_tmp, NTILE);
      jl(".n48", T_NEAR);
      generate_kloop(NRegs);
      write_back(MTILE, NRegs, parambase, reg_matCptr, reg_cstep, reg_itern);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n48");
      cmp(reg_tmp, 48);
      jl(".n32", T_NEAR);
      generate_kloop(3);
      write_back(MTILE, 3, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 48);
      add(reg_matBptr, 48 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n32");
      cmp(reg_tmp, 32);
      jl(".n16", T_NEAR);
      generate_kloop(2);
      write_back(MTILE, 2, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 32);
      add(reg_matBptr, 32 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n16");
      xor_(reg_iterk, reg_iterk);
      generate_kloop(1);
      write_back(MTILE, 1, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 16);
      add(reg_matBptr, 16 * sizeof(BType));
      L(".nend");
      cmp(reg_itern, reg_nsize);
      jb(".nloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _nregs) {
      inLocalLabel();
      L(".kloop");
      mov(reg_tmp, reg_ksize);
      sub(reg_tmp, reg_iterk);
      cmp(reg_tmp, KUNROLL * KTILE);
      jl(".k1loop", T_NEAR);
      generate_fma(_nregs, KUNROLL, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_nregs, 1, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _NTile, int _kunroll, const Xbyak::Reg64& reg_tmp, const Xbyak::Reg64& reg_matAptr,
                      const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      mov(reg_tmp, NTILE * 4);
      if (_NTile <= B_tilenum) {
        for (int kk = 0; kk < _kunroll; kk++) {
          for (int i = 0; i < _NTile; i++) {
            tileloaddt1(Xbyak::Tmm(BTile + i), ptr[reg_matBptr + reg_tmp + kk * BKStepSize + i * 64]);
          }

          for (int mm = 0; mm < 1; mm++) {
            tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_matAptr + reg_astep + kk * AKStepSize]);
            for (int i = 0; i < _NTile; i++) {
              tdpbf16ps(Xbyak::Tmm(CTile + mm * C_tilenum + i), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile + i));
            }
          }
        }
      } else {
        for (int kk = 0; kk < _kunroll; kk++) {
          for (int i = 0; i < _NTile - 1; i++) {
            tileloaddt1(Xbyak::Tmm(BTile + i), ptr[reg_matBptr + reg_tmp + kk * BKStepSize + i * 64]);
          }

          for (int mm = 0; mm < 1; mm++) {
            tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_matAptr + reg_astep + kk * AKStepSize]);
            for (int i = 0; i < _NTile - 1; i++) {
              tdpbf16ps(Xbyak::Tmm(CTile + mm * C_tilenum + i), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile + i));
            }
            tileloaddt1(Xbyak::Tmm(BTile + 0), ptr[reg_matBptr + reg_tmp + kk * BKStepSize + (_NTile - 1) * 64]);
            tdpbf16ps(Xbyak::Tmm(CTile + mm * C_tilenum + _NTile - 1), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile + 0));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs, const Xbyak::Reg64& parambase, const Xbyak::Reg64& reg_matCptr,
                    const Xbyak::Reg64& reg_cstep, const Xbyak::Reg64& reg_itern) {
      inLocalLabel();
      mov(reg_tmp, dword[parambase + OFFSET(workspace)]);
      mov(reg_tmp1, NTILE * 4);
      for (int mm = 0; mm < 1; mm++) {
        for (int i = 0; i < _NRegs; i++) {
          tilestored(ptr[reg_tmp + reg_tmp1 + i * 64 + mm * 16 * NTILE * 4], Xbyak::Tmm(CTile + mm * C_tilenum + i));
        }
      }
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".LACC", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      inLocalLabel();
      xor_(reg_tmp1, reg_tmp1);
      L(".mloop");
      for (int j = 0; j < _NRegs; j++) {
        vmovups(Xbyak::Zmm(CReg + j), ptr[reg_tmp + j * 64]);
        vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + j));
      }
      add(reg_matCptr, reg_cstep);
      add(reg_tmp, NTILE * 4);
      add(reg_tmp1, 1);
      cmp(reg_tmp1.cvt32(), ptr[parambase + OFFSET(msize)]);
      jb(".mloop");
      outLocalLabel();
      jmp(".LEND", T_NEAR);
      L(".LACC");
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      inLocalLabel();
      xor_(reg_tmp1, reg_tmp1);
      L(".mloop");
      for (int j = 0; j < _NRegs; j++) {
        vmovups(Xbyak::Zmm(CReg + j), ptr[reg_tmp + j * 64]);
        vaddps(Xbyak::Zmm(CReg + j), ptr[reg_matCptr + j * VecBytes]);
        vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + j));
      }
      add(reg_matCptr, reg_cstep);
      add(reg_tmp, NTILE * 4);
      add(reg_tmp1, 1);
      cmp(reg_tmp1.cvt32(), ptr[parambase + OFFSET(msize)]);
      jb(".mloop");
      outLocalLabel();
      L(".LEND");
      nop();
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_nsize;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_itern;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_ret = rax;
    Xbyak::Opmask msk_wr = k1;
  };

 public:
  GemmCore_Row_NN_16x64_AMX_BF16() { mCodes.generate_code(); }

  void forward(AType* matA, BType* matB, CType* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
               int kpos) {
    char tmp[NTILE * MTILE * sizeof(CType)];
    MicroKernel::tileconfig_t mCfg;
    memset(&mCfg, 0, sizeof(mCfg));
    auto param = params{matA, matB, matC, _k, _m, _n, _astride, _bstride, _cstride, kpos, tmp, &mCfg};
    if (_m <= MTILE) {
      jblas::xbyak::JitAmxtile::configure_tiles(mCfg, _m < 16 ? _m : 16, _n < 16 ? _n : 16, _k < KTILE ? _k : KTILE,
                                                sizeof(BType), MicroKernel::A_tilenum, MicroKernel::B_tilenum,
                                                MicroKernel::C_tilenum);
      mCodes.mKernel(&param);
    } else {
      assert(0);
    }
  }

 private:
  MicroKernel mCodes;
};

class GemmCore_Row_NN_16x48_AMX_BF16 {
 public:
  typedef utils::bf16 AType;
  typedef utils::bf16 BType;
  typedef float CType;
  struct params {
    AType* matA;
    BType* matB;
    CType* matC;
    int k, msize, nsize;
    int astep, bstep, cstep;
    int kpos;
    void *workspace, *cfg;
  };
  typedef long long (*func_t)(params*);

  static JBLAS_ISA constexpr ISA = JblasAMX_BF16;
  static GemmCoreType constexpr TYPE = GemmCoreType::AMX_BF16_16x48;
  static int constexpr NTILE = 48, MTILE = 16, KTILE = 64 / sizeof(BType);
  static int constexpr PACK_ROW = 2;
  static int constexpr KUNROLL = 2;
  static int constexpr PREFERED_N = 240;
  class MicroKernel : protected jblas::xbyak::JitAmxbf16 {
   public:
    friend GemmCore_Row_NN_16x48_AMX_BF16;
    MicroKernel() {}
    static int constexpr CReg = 0, TmpReg = 4;
    static int constexpr NRegs = 3;
    static int constexpr CRegCount = NRegs;
    static int constexpr C_tilenum = 3, A_tilenum = 1, B_tilenum = 3;
    static int constexpr CTile = 0, ATile = CTile + C_tilenum, BTile = ATile + A_tilenum;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);
    static int constexpr VecBytes = 64;

    void generate_code() {
      reset();
      generate_mtile();
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    void generate_mtile() {
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_nsize = st.t[9];
      reg_cstep = st.t[3];
      reg_astep = st.t[5];
      reg_iterk = st.t[4];
      reg_itern = st.t[7];
      reg_tmp = st.t[6];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[10];
      reg_ret = rax;

      vreg_push(rsp);
      mov(reg_tmp, ptr[parambase + OFFSET(cfg)]);
      ldtilecfg(ptr[reg_tmp]);

      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_nsize, ptr[parambase + OFFSET(nsize)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      xor_(reg_itern, reg_itern);
      L(".nloop");
      for (int i = 0; i < C_tilenum; i++) {
        tilezero(Xbyak::Tmm(CTile + i));
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_tmp1, reg_matBptr);

      xor_(reg_iterk, reg_iterk);

      mov(reg_tmp, reg_nsize);
      sub(reg_tmp, reg_itern);
      cmp(reg_tmp, NTILE);
      jl(".n32", T_NEAR);
      generate_kloop(NRegs);
      write_back(MTILE, NRegs, parambase, reg_matCptr, reg_cstep, reg_itern);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n32");
      cmp(reg_tmp, 32);
      jl(".n16", T_NEAR);
      generate_kloop(2);
      write_back(MTILE, 2, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 32);
      add(reg_matBptr, 32 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n16");
      xor_(reg_iterk, reg_iterk);
      generate_kloop(1);
      write_back(MTILE, 1, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 16);
      add(reg_matBptr, 16 * sizeof(BType));
      L(".nend");
      cmp(reg_itern, reg_nsize);
      jb(".nloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _nregs) {
      inLocalLabel();
      L(".kloop");
      mov(reg_tmp, reg_ksize);
      sub(reg_tmp, reg_iterk);
      cmp(reg_tmp, KUNROLL * KTILE);
      jl(".k1loop", T_NEAR);
      generate_fma(_nregs, KUNROLL, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_nregs, 1, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _NTile, int _kunroll, const Xbyak::Reg64& reg_tmp, const Xbyak::Reg64& reg_matAptr,
                      const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      mov(reg_tmp, NTILE * 4);
      for (int kk = 0; kk < _kunroll; kk++) {
        for (int i = 0; i < _NTile; i++) {
          tileloaddt1(Xbyak::Tmm(BTile + i), ptr[reg_matBptr + reg_tmp + kk * BKStepSize + i * 64]);
        }

        for (int mm = 0; mm < 1; mm++) {
          tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_matAptr + reg_astep + kk * AKStepSize]);
          for (int i = 0; i < _NTile; i++) {
            tdpbf16ps(Xbyak::Tmm(CTile + mm * C_tilenum + i), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile + i));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs, const Xbyak::Reg64& parambase, const Xbyak::Reg64& reg_matCptr,
                    const Xbyak::Reg64& reg_cstep, const Xbyak::Reg64& reg_itern) {
      inLocalLabel();
      mov(reg_tmp, dword[parambase + OFFSET(workspace)]);
      mov(reg_tmp1, NTILE * 4);
      for (int mm = 0; mm < 1; mm++) {
        for (int i = 0; i < _NRegs; i++) {
          tilestored(ptr[reg_tmp + reg_tmp1 + i * 64 + mm * 16 * NTILE * 4], Xbyak::Tmm(CTile + mm * C_tilenum + i));
        }
      }
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".LACC", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      inLocalLabel();
      xor_(reg_tmp1, reg_tmp1);
      L(".mloop");
      for (int j = 0; j < _NRegs; j++) {
        vmovups(Xbyak::Zmm(CReg + j), ptr[reg_tmp + j * 64]);
        vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + j));
      }
      add(reg_matCptr, reg_cstep);
      add(reg_tmp, NTILE * 4);
      add(reg_tmp1, 1);
      cmp(reg_tmp1.cvt32(), ptr[parambase + OFFSET(msize)]);
      jb(".mloop");
      outLocalLabel();
      jmp(".LEND", T_NEAR);
      L(".LACC");
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      inLocalLabel();
      xor_(reg_tmp1, reg_tmp1);
      L(".mloop");
      for (int j = 0; j < _NRegs; j++) {
        vmovups(Xbyak::Zmm(CReg + j), ptr[reg_tmp + j * 64]);
        vaddps(Xbyak::Zmm(CReg + j), ptr[reg_matCptr + j * VecBytes]);
        vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + j));
      }
      add(reg_matCptr, reg_cstep);
      add(reg_tmp, NTILE * 4);
      add(reg_tmp1, 1);
      cmp(reg_tmp1.cvt32(), ptr[parambase + OFFSET(msize)]);
      jb(".mloop");
      outLocalLabel();
      L(".LEND");
      nop();
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_nsize;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_itern;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_ret = rax;
    Xbyak::Opmask msk_wr = k1;
  };

 public:
  GemmCore_Row_NN_16x48_AMX_BF16() { mCodes.generate_code(); }

  void forward(AType* matA, BType* matB, CType* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
               int kpos) {
    char tmp[NTILE * MTILE * sizeof(CType)];
    MicroKernel::tileconfig_t mCfg;
    std::memset(&mCfg, 0, sizeof(mCfg));
    auto param = params{matA, matB, matC, _k, _m, _n, _astride, _bstride, _cstride, kpos, tmp, &mCfg};
    if (_m <= MTILE) {
      jblas::xbyak::JitAmxtile::configure_tiles(mCfg, _m < 16 ? _m : 16, _n < 16 ? _n : 16, _k < KTILE ? _k : KTILE,
                                                sizeof(BType), MicroKernel::A_tilenum, MicroKernel::B_tilenum,
                                                MicroKernel::C_tilenum);
      mCodes.mKernel(&param);
    } else {
      assert(0);
    }
  }

 private:
  MicroKernel mCodes;
};

template <class T_A_, class T_B_>
class GemmCore_Row_NN_16x64_AMX_I8 {
 public:
  typedef T_A_ AType;
  typedef T_B_ BType;
  typedef int32_t CType;
  struct params {
    AType* matA;
    BType* matB;
    CType* matC;
    int k, msize, nsize;
    int astep, bstep, cstep;
    int kpos;
    void *workspace, *cfg;
  };
  typedef long long (*func_t)(params*);

  static JBLAS_ISA constexpr ISA = JblasAMX_INT8;
  static GemmCoreType constexpr TYPE = GemmCoreType::AMX_INT8_16x64;
  static int constexpr NTILE = 64, MTILE = 16, KTILE = 64 / sizeof(BType);
  static int constexpr PACK_ROW = 4;
  static int constexpr KUNROLL = 2;
  static int constexpr PREFERED_N = 256;
  class MicroKernel : protected jblas::xbyak::JitAmxint8 {
   public:
    friend GemmCore_Row_NN_16x64_AMX_I8<T_A_, T_B_>;
    MicroKernel() {}
    static int constexpr CReg = 0, TmpReg = 4;
    static int constexpr NRegs = 4;
    static int constexpr CRegCount = NRegs;
    static int constexpr C_tilenum = 4, A_tilenum = 1, B_tilenum = 3;
    static int constexpr CTile = 0, ATile = CTile + C_tilenum, BTile = ATile + A_tilenum;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);
    static int constexpr VecBytes = 64;

    void generate_code() {
      reset();
      generate_mtile();
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    void generate_mtile() {
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_nsize = st.t[9];
      reg_cstep = st.t[3];
      reg_astep = st.t[5];
      reg_iterk = st.t[4];
      reg_itern = st.t[7];
      reg_tmp = st.t[6];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[10];
      reg_ret = rax;

      vreg_push(rsp);
      mov(reg_tmp, ptr[parambase + OFFSET(cfg)]);
      ldtilecfg(ptr[reg_tmp]);

      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_nsize, ptr[parambase + OFFSET(nsize)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      xor_(reg_itern, reg_itern);
      L(".nloop");
      for (int i = 0; i < C_tilenum; i++) {
        tilezero(Xbyak::Tmm(CTile + i));
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_tmp1, reg_matBptr);

      xor_(reg_iterk, reg_iterk);

      mov(reg_tmp, reg_nsize);
      sub(reg_tmp, reg_itern);
      cmp(reg_tmp, NTILE);
      jl(".n48", T_NEAR);
      generate_kloop(NRegs);
      write_back(MTILE, NRegs, parambase, reg_matCptr, reg_cstep, reg_itern);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n48");
      cmp(reg_tmp, 48);
      jl(".n32", T_NEAR);
      generate_kloop(3);
      write_back(MTILE, 3, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 48);
      add(reg_matBptr, 48 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n32");
      cmp(reg_tmp, 32);
      jl(".n16", T_NEAR);
      generate_kloop(2);
      write_back(MTILE, 2, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 32);
      add(reg_matBptr, 32 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n16");
      xor_(reg_iterk, reg_iterk);
      generate_kloop(1);
      write_back(MTILE, 1, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 16);
      add(reg_matBptr, 16 * sizeof(BType));
      L(".nend");
      cmp(reg_itern, reg_nsize);
      jb(".nloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _nregs) {
      inLocalLabel();
      L(".kloop");
      mov(reg_tmp, reg_ksize);
      sub(reg_tmp, reg_iterk);
      cmp(reg_tmp, KUNROLL * KTILE);
      jl(".k1loop", T_NEAR);
      generate_fma(_nregs, KUNROLL, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_nregs, 1, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _NTile, int _kunroll, const Xbyak::Reg64& reg_tmp, const Xbyak::Reg64& reg_matAptr,
                      const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      mov(reg_tmp, NTILE * 4);
      if (_NTile <= B_tilenum) {
        for (int kk = 0; kk < _kunroll; kk++) {
          for (int i = 0; i < _NTile; i++) {
            tileloaddt1(Xbyak::Tmm(BTile + i), ptr[reg_matBptr + reg_tmp + kk * BKStepSize + i * 64]);
          }

          for (int mm = 0; mm < 1; mm++) {
            tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_matAptr + reg_astep + kk * AKStepSize]);
            for (int i = 0; i < _NTile; i++) {
              _tdpb<AType, BType>(Xbyak::Tmm(CTile + mm * C_tilenum + i), Xbyak::Tmm(ATile + mm),
                                  Xbyak::Tmm(BTile + i));
            }
          }
        }
      } else {
        for (int kk = 0; kk < _kunroll; kk++) {
          for (int i = 0; i < _NTile - 1; i++) {
            tileloaddt1(Xbyak::Tmm(BTile + i), ptr[reg_matBptr + reg_tmp + kk * BKStepSize + i * 64]);
          }

          for (int mm = 0; mm < 1; mm++) {
            tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_matAptr + reg_astep + kk * AKStepSize]);
            for (int i = 0; i < _NTile - 1; i++) {
              _tdpb<AType, BType>(Xbyak::Tmm(CTile + mm * C_tilenum + i), Xbyak::Tmm(ATile + mm),
                                  Xbyak::Tmm(BTile + i));
            }
            tileloaddt1(Xbyak::Tmm(BTile + 0), ptr[reg_matBptr + reg_tmp + kk * BKStepSize + (_NTile - 1) * 64]);
            _tdpb<AType, BType>(Xbyak::Tmm(CTile + mm * C_tilenum + _NTile - 1), Xbyak::Tmm(ATile + mm),
                                Xbyak::Tmm(BTile + 0));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs, const Xbyak::Reg64& parambase, const Xbyak::Reg64& reg_matCptr,
                    const Xbyak::Reg64& reg_cstep, const Xbyak::Reg64& reg_itern) {
      inLocalLabel();
      mov(reg_tmp, dword[parambase + OFFSET(workspace)]);
      mov(reg_tmp1, NTILE * 4);
      for (int mm = 0; mm < 1; mm++) {
        for (int i = 0; i < _NRegs; i++) {
          tilestored(ptr[reg_tmp + reg_tmp1 + i * 64 + mm * 16 * NTILE * 4], Xbyak::Tmm(CTile + mm * C_tilenum + i));
        }
      }
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".LACC", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      inLocalLabel();
      xor_(reg_tmp1, reg_tmp1);
      L(".mloop");
      for (int j = 0; j < _NRegs; j++) {
        vmovups(Xbyak::Zmm(CReg + j), ptr[reg_tmp + j * 64]);
        vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + j));
      }
      add(reg_matCptr, reg_cstep);
      add(reg_tmp, NTILE * 4);
      add(reg_tmp1, 1);
      cmp(reg_tmp1.cvt32(), ptr[parambase + OFFSET(msize)]);
      jb(".mloop");
      outLocalLabel();
      jmp(".LEND", T_NEAR);
      L(".LACC");
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      inLocalLabel();
      xor_(reg_tmp1, reg_tmp1);
      L(".mloop");
      for (int j = 0; j < _NRegs; j++) {
        vmovups(Xbyak::Zmm(CReg + j), ptr[reg_tmp + j * 64]);
        vpaddd(Xbyak::Zmm(CReg + j), Xbyak::Zmm(CReg + j), ptr[reg_matCptr + j * VecBytes]);
        vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + j));
      }
      add(reg_matCptr, reg_cstep);
      add(reg_tmp, NTILE * 4);
      add(reg_tmp1, 1);
      cmp(reg_tmp1.cvt32(), ptr[parambase + OFFSET(msize)]);
      jb(".mloop");
      outLocalLabel();
      L(".LEND");
      nop();
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_nsize;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_itern;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_ret = rax;
    Xbyak::Opmask msk_wr = k1;
  };

 public:
  GemmCore_Row_NN_16x64_AMX_I8() { mCodes.generate_code(); }

  void forward(AType* matA, BType* matB, CType* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
               int kpos) {
    char tmp[NTILE * MTILE * sizeof(CType)];
    typename MicroKernel::tileconfig_t mCfg;
    memset(&mCfg, 0, sizeof(mCfg));
    auto param = params{matA, matB, matC, _k, _m, _n, _astride, _bstride, _cstride, kpos, tmp, &mCfg};
    if (_m <= MTILE) {
      jblas::xbyak::JitAmxint8::configure_tiles(mCfg, _m < 16 ? _m : 16, _n < 16 ? _n : 16, _k < KTILE ? _k : KTILE,
                                                sizeof(BType), MicroKernel::A_tilenum, MicroKernel::B_tilenum,
                                                MicroKernel::C_tilenum);
      mCodes.mKernel(&param);
    } else {
      assert(0);
    }
  }

 private:
  MicroKernel mCodes;
};
using GemmCore_Row_NN_16x64_AMX_U8S8 = GemmCore_Row_NN_16x64_AMX_I8<uint8_t, int8_t>;

template <class T_A_, class T_B_>
class GemmCore_Row_NN_16x48_AMX_I8 {
 public:
  typedef T_A_ AType;
  typedef T_B_ BType;
  typedef int32_t CType;
  struct params {
    AType* matA;
    BType* matB;
    CType* matC;
    int k, msize, nsize;
    int astep, bstep, cstep;
    int kpos;
    void *workspace, *cfg;
  };
  typedef long long (*func_t)(params*);

  static JBLAS_ISA constexpr ISA = JblasAMX_INT8;
  static GemmCoreType constexpr TYPE = GemmCoreType::AMX_INT8_16x48;
  static int constexpr NTILE = 48, MTILE = 16, KTILE = 64 / sizeof(BType);
  static int constexpr PACK_ROW = 4 / sizeof(BType);
  static int constexpr KUNROLL = 2;
  static int constexpr PREFERED_N = 240;  // TODO?
  class MicroKernel : protected jblas::xbyak::JitAmxint8 {
   public:
    friend GemmCore_Row_NN_16x48_AMX_I8<T_A_, T_B_>;
    MicroKernel() {}
    static int constexpr CReg = 0;
    static int constexpr C_tilenum = 3, A_tilenum = 1, B_tilenum = 3;
    static int constexpr CTile = 0, ATile = CTile + C_tilenum, BTile = ATile + A_tilenum;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);
    static int constexpr VecBytes = 64;

    void generate_code() {
      reset();
      generate_mtile();
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    void generate_mtile() {
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 11, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_nsize = st.t[9];
      reg_cstep = st.t[3];
      reg_astep = st.t[5];
      reg_iterk = st.t[4];
      reg_itern = st.t[7];
      reg_tmp = st.t[6];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[10];
      reg_ret = rax;

      vreg_push(rsp);
      mov(reg_tmp, ptr[parambase + OFFSET(cfg)]);
      ldtilecfg(ptr[reg_tmp]);

      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_nsize, ptr[parambase + OFFSET(nsize)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      xor_(reg_itern, reg_itern);
      L(".nloop");
      for (int i = 0; i < C_tilenum; i++) {
        tilezero(Xbyak::Tmm(CTile + i));
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_tmp1, reg_matBptr);

      xor_(reg_iterk, reg_iterk);

      mov(reg_tmp, reg_nsize);
      sub(reg_tmp, reg_itern);
      cmp(reg_tmp, NTILE);
      jl(".n32", T_NEAR);
      generate_kloop(C_tilenum);
      write_back(MTILE, C_tilenum, parambase, reg_matCptr, reg_cstep, reg_itern);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n32");
      cmp(reg_tmp, 32);
      jl(".n16", T_NEAR);
      generate_kloop(2);
      write_back(MTILE, 2, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 32);
      add(reg_matBptr, 32 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n16");
      xor_(reg_iterk, reg_iterk);
      generate_kloop(1);
      write_back(MTILE, 1, parambase, reg_matCptr, reg_cstep, reg_itern);
      add(reg_itern, 16);
      add(reg_matBptr, 16 * sizeof(BType));
      L(".nend");
      cmp(reg_itern, reg_nsize);
      jb(".nloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _nregs) {
      inLocalLabel();
      L(".kloop");
      mov(reg_tmp, reg_ksize);
      sub(reg_tmp, reg_iterk);
      cmp(reg_tmp, KUNROLL * KTILE);
      jl(".k1loop", T_NEAR);
      generate_fma(_nregs, KUNROLL, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_nregs, 1, reg_tmp, reg_matAptr, reg_tmp1, reg_astep);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _NTile, int _kunroll, const Xbyak::Reg64& reg_tmp, const Xbyak::Reg64& reg_matAptr,
                      const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      mov(reg_tmp, NTILE * 4);
      if (_NTile <= B_tilenum) {
        for (int kk = 0; kk < _kunroll; kk++) {
          for (int i = 0; i < _NTile; i++) {
            tileloaddt1(Xbyak::Tmm(BTile + i), ptr[reg_matBptr + reg_tmp + kk * BKStepSize + i * 64]);
          }

          for (int mm = 0; mm < 1; mm++) {
            tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_matAptr + reg_astep + kk * AKStepSize]);
            for (int i = 0; i < _NTile; i++) {
              _tdpb<AType, BType>(Xbyak::Tmm(CTile + mm * C_tilenum + i), Xbyak::Tmm(ATile + mm),
                                  Xbyak::Tmm(BTile + i));
            }
          }
        }
      } else {
        for (int kk = 0; kk < _kunroll; kk++) {
          for (int i = 0; i < _NTile - 1; i++) {
            tileloaddt1(Xbyak::Tmm(BTile + i), ptr[reg_matBptr + reg_tmp + kk * BKStepSize + i * 64]);
          }

          for (int mm = 0; mm < 1; mm++) {
            tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_matAptr + reg_astep + kk * AKStepSize]);
            for (int i = 0; i < _NTile - 1; i++) {
              _tdpb<AType, BType>(Xbyak::Tmm(CTile + mm * C_tilenum + i), Xbyak::Tmm(ATile + mm),
                                  Xbyak::Tmm(BTile + i));
            }
            tileloaddt1(Xbyak::Tmm(BTile + 0), ptr[reg_matBptr + reg_tmp + kk * BKStepSize + (_NTile - 1) * 64]);
            _tdpb<AType, BType>(Xbyak::Tmm(CTile + mm * C_tilenum + _NTile - 1), Xbyak::Tmm(ATile + mm),
                                Xbyak::Tmm(BTile + 0));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs, const Xbyak::Reg64& parambase, const Xbyak::Reg64& reg_matCptr,
                    const Xbyak::Reg64& reg_cstep, const Xbyak::Reg64& reg_itern) {
      inLocalLabel();
      mov(reg_tmp, dword[parambase + OFFSET(workspace)]);
      mov(reg_tmp1, NTILE * 4);
      for (int mm = 0; mm < 1; mm++) {
        for (int i = 0; i < _NRegs; i++) {
          tilestored(ptr[reg_tmp + reg_tmp1 + i * 64 + mm * 16 * NTILE * 4], Xbyak::Tmm(CTile + mm * C_tilenum + i));
        }
      }
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".LACC", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      inLocalLabel();
      xor_(reg_tmp1, reg_tmp1);
      L(".mloop");
      for (int j = 0; j < _NRegs; j++) {
        vmovups(Xbyak::Zmm(CReg + j), ptr[reg_tmp + j * 64]);
        vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + j));
      }
      add(reg_matCptr, reg_cstep);
      add(reg_tmp, NTILE * 4);
      add(reg_tmp1, 1);
      cmp(reg_tmp1.cvt32(), ptr[parambase + OFFSET(msize)]);
      jb(".mloop");
      outLocalLabel();
      jmp(".LEND", T_NEAR);
      L(".LACC");
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr + reg_itern * sizeof(CType)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      inLocalLabel();
      xor_(reg_tmp1, reg_tmp1);
      L(".mloop");
      for (int j = 0; j < _NRegs; j++) {
        vmovups(Xbyak::Zmm(CReg + j), ptr[reg_tmp + j * 64]);
        vpaddd(Xbyak::Zmm(CReg + j), Xbyak::Zmm(CReg + j), ptr[reg_matCptr + j * VecBytes]);
        vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + j));
      }
      add(reg_matCptr, reg_cstep);
      add(reg_tmp, NTILE * 4);
      add(reg_tmp1, 1);
      cmp(reg_tmp1.cvt32(), ptr[parambase + OFFSET(msize)]);
      jb(".mloop");
      outLocalLabel();
      L(".LEND");
      nop();
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_nsize;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_itern;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_ret = rax;
    Xbyak::Opmask msk_wr = k1;
  };

 public:
  GemmCore_Row_NN_16x48_AMX_I8() { mCodes.generate_code(); }

  void forward(AType* matA, BType* matB, CType* matC, int _m, int _n, int _k, int _astride, int _bstride, int _cstride,
               int kpos) {
    char tmp[NTILE * MTILE * sizeof(CType)];
    typename MicroKernel::tileconfig_t mCfg;
    memset(&mCfg, 0, sizeof(mCfg));
    auto param = params{matA, matB, matC, _k, _m, _n, _astride, _bstride, _cstride, kpos, tmp, &mCfg};
    if (_m <= MTILE) {
      jblas::xbyak::JitAmxint8::configure_tiles(mCfg, _m < 16 ? _m : 16, _n < 16 ? _n : 16, _k < KTILE ? _k : KTILE,
                                                sizeof(BType), MicroKernel::A_tilenum, MicroKernel::B_tilenum,
                                                MicroKernel::C_tilenum);
      mCodes.mKernel(&param);
    } else {
      assert(0);
    }
  }

 private:
  MicroKernel mCodes;
};
using GemmCore_Row_NN_16x48_AMX_U8S8 = GemmCore_Row_NN_16x48_AMX_I8<uint8_t, int8_t>;
using GemmCore_Row_NN_16x48_AMX_S8S8 = GemmCore_Row_NN_16x48_AMX_I8<int8_t, int8_t>;

// special kblock core: A:u8/s8 B:s8 intra-block accumulator:s32 inter-block
// accumulator:f32
//  KBlocks= K/kblock
//  Weight scale=KBlocks*N
//  Activation zp=M*KBlocks scale=M*KBlocks
namespace kblock {

// KBLOCK>=KUNROLL*KTILE
class GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK {
 public:
  typedef uint8_t AType;
  typedef int8_t BType;
  typedef float CType;
  struct params {
    uint8_t* matA;
    int8_t* matB;
    CType* matC;
    uint8_t* zpA;
    float* scaleA;
    void* scaleB;
    int ldsa, ldsb;
    int kblock;
    int k, nsize;
    int astep, cstep;
    int kpos;
  };
  typedef long long (*func_t)(params*);

  static JBLAS_ISA constexpr ISA = JblasAVX512_VNNI;
  static GemmCoreType constexpr TYPE = GemmCoreType::AVX512_VNNI_3x48_KBLOCK;
  static int constexpr NTILE = 48, MTILE = 3, KTILE = 4 / sizeof(BType);
  static int constexpr PACK_ROW = KTILE;
  static int constexpr KUNROLL = 2;
  static int constexpr PREFERED_N = 192;

  class MicroKernel : protected jblas::xbyak::JitAvx512vnni {
   public:
    MicroKernel() {}
    int CRegCount = 9, BRegCount = 3, ARegCount = 1, ZpARegCount = MTILE;
    int CReg = 0, CF32Reg = 9, BReg = 18, AReg = 21, ZpAReg = 22, ZpTmp = 25;
    int const NRegs = 3;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);
    static int constexpr VecBytes = 64;

    void generate_code(int _mtile, JBLAS_DTYPE _scale_dt) {
      mScaleType = _scale_dt;
      reset();
      generate_mtile(_mtile);
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    JBLAS_DTYPE mScaleType = JblasF32;

    void generate_mtile(int _mtile) {
      CRegCount = _mtile * NRegs;
      ZpARegCount = _mtile;
      BRegCount = NRegs;
      CF32Reg = CReg + CRegCount;
      BReg = CF32Reg + CRegCount;
      AReg = BReg + BRegCount;
      ZpAReg = AReg + ARegCount;
      ZpTmp = ZpAReg + ZpARegCount;
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 13, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_cstep = st.t[3];
      reg_iterk = st.t[4];
      reg_astep = st.t[5];
      reg_kblock = st.t[6];
      reg_tmp = st.t[7];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[9];
      reg_zpAptr = st.t[10];
      reg_scaleAptr = st.t[11];
      reg_scaleBptr = st.t[12];
      reg_ret = rax;

      vreg_push(rsp);

      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_kblock, ptr[parambase + OFFSET(kblock)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vpxorq(Xbyak::Zmm(CF32Reg + i * NRegs + j), Xbyak::Zmm(CF32Reg + i * NRegs + j),
                 Xbyak::Zmm(CF32Reg + i * NRegs + j));
        }
      }
      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      mov(reg_zpAptr, ptr[parambase + OFFSET(zpA)]);
      mov(reg_scaleAptr, ptr[parambase + OFFSET(scaleA)]);
      mov(reg_scaleBptr, ptr[parambase + OFFSET(scaleB)]);
      xor_(reg_iterk, reg_iterk);

      load32(reg_tmp, ptr[parambase + OFFSET(nsize)]);
      cmp(reg_tmp, NTILE);
      jl(".n32", T_NEAR);
      generate_kloop(_mtile, NRegs);
      write_back(_mtile, NRegs, reg_matCptr, reg_cstep);
      jmp(".nend", T_NEAR);

      L(".n32");
      cmp(reg_tmp, 32);
      jl(".n16", T_NEAR);
      generate_kloop(_mtile, 2);
      write_back(_mtile, 2, reg_matCptr, reg_cstep);
      jmp(".nend", T_NEAR);

      L(".n16");
      generate_kloop(_mtile, 1);
      write_back(_mtile, 1, reg_matCptr, reg_cstep);

      L(".nend");
      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _mtile, int _nregs) {
      inLocalLabel();
      L(".kloop");
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _nregs; j++) {
          vpxorq(Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j));
        }
      }
      mov(reg_tmp, reg_zpAptr);
      load32(reg_tmp1, ptr[parambase + OFFSET(ldsa)]);
      for (int i = 0; i < _mtile; i++) {
        vpbroadcastb(Xbyak::Zmm(ZpAReg + i), ptr[reg_tmp]);
        add(reg_tmp, reg_tmp1);
      }
      xor_(reg_tmp2, reg_tmp2);
      L(".kbloop");
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp, reg_matAptr, reg_matBptr, reg_astep);
      generate_zp_fma(_mtile, _nregs, KUNROLL, reg_tmp, reg_matBptr);
      add(reg_matAptr, AKStepSize * KUNROLL);
      add(reg_matBptr, BKStepSize * KUNROLL);
      add(reg_iterk, KTILE * KUNROLL);
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jge(".kbend");
      add(reg_tmp2, KTILE * KUNROLL);
      cmp(reg_tmp2.cvt32(), ptr[parambase + OFFSET(kblock)]);
      jb(".kbloop");
      L(".kbend");
      mov(reg_tmp, reg_scaleAptr);
      load32(reg_tmp1, ptr[parambase + OFFSET(ldsa)]);
      for (int i = 0; i < _mtile; i++) {
        vbroadcastss(Xbyak::Zmm(ZpAReg + i), ptr[reg_tmp]);
        lea(reg_tmp, ptr[reg_tmp + reg_tmp1 * sizeof(float)]);
      }
      for (int i = 0; i < _nregs; i++) {
        if (mScaleType == JblasF32) {
          vmovups(Xbyak::Zmm(BReg + i), ptr[reg_scaleBptr + i * VecBytes]);
        } else if (mScaleType == JblasBF16) {
          loadbf16_f32(Xbyak::Zmm(BReg + i), ptr[reg_scaleBptr + i * VecBytes / 2]);
        }
      }
      generate_f32_accumulate(_mtile, _nregs);
      add(reg_zpAptr, sizeof(AType));
      add(reg_scaleAptr, sizeof(float));
      load32(reg_tmp, ptr[parambase + OFFSET(ldsb)]);
      if (mScaleType == JblasF32) {
        lea(reg_scaleBptr, ptr[reg_scaleBptr + reg_tmp * 4]);
      } else if (mScaleType == JblasBF16) {
        lea(reg_scaleBptr, ptr[reg_scaleBptr + reg_tmp * 2]);
      }
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _kunroll, const Xbyak::Reg64& reg_tmp,
                      const Xbyak::Reg64& reg_matAptr, const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      for (int kk = 0; kk < _kunroll; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Zmm(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastd(Xbyak::Zmm(AReg), ptr[reg_tmp]);
          add(reg_tmp, reg_astep);
          for (int i = 0; i < _NRegs; i++) {
            vpdpbusds(Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(AReg), Xbyak::Zmm(BReg + i));
          }
        }
      }
    }

    void generate_zp_fma(int _mtile, int _NRegs, int _kunroll, const Xbyak::Reg64& reg_tmp,
                         const Xbyak::Reg64& reg_matBptr) {
      for (int kk = 0; kk < _kunroll; kk++) {
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Zmm(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          for (int i = 0; i < _NRegs; i++) {
            vpxorq(Xbyak::Zmm(ZpTmp), Xbyak::Zmm(ZpTmp), Xbyak::Zmm(ZpTmp));
            vpdpbusds(Xbyak::Zmm(ZpTmp), Xbyak::Zmm(ZpAReg + mm), Xbyak::Zmm(BReg + i));
            vpsubd(Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(ZpTmp));
          }
        }
      }
    }

    void generate_f32_accumulate(int _mtile, int _NRegs) {
      for (int mm = 0; mm < _mtile; mm++) {
        for (int i = 0; i < _NRegs; i++) {
          vcvtdq2ps(Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(CReg + mm * NRegs + i));
          vmulps(Xbyak::Zmm(AReg), Xbyak::Zmm(ZpAReg + mm), Xbyak::Zmm(BReg + i));
          vmulps(Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(AReg));
          vaddps(Xbyak::Zmm(CF32Reg + mm * NRegs + i), Xbyak::Zmm(CReg + mm * NRegs + i));
        }
      }
    }

    void write_back(int _mtile, int _NRegs, const Xbyak::Reg64& reg_matCptr, const Xbyak::Reg64& reg_cstep) {
      inLocalLabel();
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".LACC", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CF32Reg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      jmp(".LEND", T_NEAR);
      L(".LACC");
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      lea(reg_matCptr, ptr[reg_matCptr]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _NRegs; j++) {
          vaddps(Xbyak::Zmm(CF32Reg + i * NRegs + j), Xbyak::Zmm(CF32Reg + i * NRegs + j),
                 ptr[reg_matCptr + j * VecBytes]);
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CF32Reg + i * NRegs + j));
        }
        add(reg_matCptr, reg_cstep);
      }
      L(".LEND");
      nop();
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_zpAptr;
    Xbyak::Reg64 reg_scaleAptr;
    Xbyak::Reg64 reg_scaleBptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_kblock;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_ret = rax;
  };

 public:
  GemmCore_Row_NN_3x48_AVX512_VNNI_KBLOCK() {
    for (int i = 0; i < MTILE; i++) {
      mCodes[i].generate_code(i + 1, JblasF32);
      mCodesBf16[i].generate_code(i + 1, JblasBF16);
    }
  }

  void forward(AType* matA, BType* matB, CType* matC, AType* zpA, float* scaleA, int _ldsa, float* scaleB, int _ldsb,
               int _m, int _n, int _k, int _kblock, int _astride, int _bstride, int _cstride, int kpos) {
    int ldb = _bstride / sizeof(BType);
    auto param = params{matA, matB, matC, zpA, scaleA, scaleB, _ldsa, _ldsb, _kblock, _k, _n, _astride, _cstride, kpos};
    if (_m <= MTILE) {
      for (int i = 0; i < _n; i += NTILE) {
        param.matB = matB + i * ldb;
        param.matC = matC + i;
        param.nsize = i + NTILE <= _n ? NTILE : _n - i;
        param.scaleB = scaleB + i;
        mCodes[_m - 1].mKernel(&param);
      }
    } else {
      assert(0);
    }
  }

  void forward(AType* matA, BType* matB, CType* matC, AType* zpA, float* scaleA, int _ldsa, utils::bf16* scaleB,
               int _ldsb, int _m, int _n, int _k, int _kblock, int _astride, int _bstride, int _cstride, int kpos) {
    int ldb = _bstride / sizeof(BType);
    auto param = params{matA, matB, matC, zpA, scaleA, scaleB, _ldsa, _ldsb, _kblock, _k, _n, _astride, _cstride, kpos};
    if (_m <= MTILE) {
      for (int i = 0; i < _n; i += NTILE) {
        param.matB = matB + i * ldb;
        param.matC = matC + i;
        param.nsize = i + NTILE <= _n ? NTILE : _n - i;
        param.scaleB = scaleB + i;
        mCodesBf16[_m - 1].mKernel(&param);
      }
    } else {
      assert(0);
    }
  }

  void reference(AType* matA, BType* matB, CType* matC, AType* zpA, float* scaleA, int _ldsa, float* scaleB, int _ldsb,
                 int _m, int _n, int _k, int _kblock, int _astride, int _bstride, int _cstride, int kpos) {
    int lda = _astride / sizeof(matA[0]);
    int ldb = _bstride / sizeof(matB[0]);
    int ldc = _cstride / sizeof(matC[0]);
    for (int i = 0; i < _m; i++) {
      for (int j = 0; j < _n; j += NTILE) {
        for (int ij = 0; ij < NTILE; ij++) {
          if (j + ij >= _n) {
            break;
          }
          float tmpf = 0.f;
          for (int k = 0; k < _k; k += _kblock) {
            int tmp = 0;
            int zpval = int(zpA[i * _ldsa + k / _kblock]);
            for (int ik = 0; ik < _kblock; ik += 4) {
              if (k + ik >= _k) {
                break;
              }
              for (int ikk = 0; ikk < 4; ikk++) {
                tmp +=
                    (int(matA[i * lda + k + ik + ikk]) - zpval) * int(matB[(k + ik) * NTILE + ij * 4 + ikk + j * ldb]);
              }
            }
            tmpf += tmp * scaleA[i * _ldsa + k / _kblock] * scaleB[j + ij + k / _kblock * _ldsb];
          }
          matC[i * ldc + j + ij] = tmpf;
        }
      }
    }
  }
  void reference(AType* matA, BType* matB, CType* matC, AType* zpA, float* scaleA, int _ldsa, utils::bf16* scaleB,
                 int _ldsb, int _m, int _n, int _k, int _kblock, int _astride, int _bstride, int _cstride, int kpos) {
    int lda = _astride / sizeof(matA[0]);
    int ldb = _bstride / sizeof(matB[0]);
    int ldc = _cstride / sizeof(matC[0]);
    for (int i = 0; i < _m; i++) {
      for (int j = 0; j < _n; j += NTILE) {
        for (int ij = 0; ij < NTILE; ij++) {
          if (j + ij >= _n) {
            break;
          }
          float tmpf = 0.f;
          for (int k = 0; k < _k; k += _kblock) {
            int tmp = 0;
            int zpval = int(zpA[i * _ldsa + k / _kblock]);
            for (int ik = 0; ik < _kblock; ik += 4) {
              if (k + ik >= _k) {
                break;
              }
              for (int ikk = 0; ikk < 4; ikk++) {
                tmp +=
                    (int(matA[i * lda + k + ik + ikk]) - zpval) * int(matB[(k + ik) * NTILE + ij * 4 + ikk + j * ldb]);
              }
            }
            tmpf += tmp * scaleA[i * _ldsa + k / _kblock] * scaleB[j + ij + k / _kblock * _ldsb].tofloat();
          }
          matC[i * ldc + j + ij] = tmpf;
        }
      }
    }
  }

 private:
  std::array<MicroKernel, MTILE> mCodes, mCodesBf16;
};

class GemmCore_Row_NN_4x48_AVX512_VNNI_KBLOCK {
 public:
  typedef uint8_t AType;
  typedef int8_t BType;
  typedef float CType;
  struct params {
    uint8_t* matA;
    int8_t* matB;
    CType* matC;
    uint8_t* zpA;
    float* scaleA;
    void* scaleB;
    float* reduceB;
    int ldsa, ldsb;
    int kblock;
    int k, nsize;
    int astep, cstep;
    int kpos;
  };
  typedef long long (*func_t)(params*);

  static JBLAS_ISA constexpr ISA = JblasAVX512_VNNI;
  static GemmCoreType constexpr TYPE = GemmCoreType::AVX512_VNNI_4x48_KBLOCK;
  static int constexpr NTILE = 48, MTILE = 4, KTILE = 4 / sizeof(BType);
  static int constexpr PACK_ROW = KTILE;
  static int constexpr KUNROLL = 2;
  static int constexpr PREFERED_N = 192;

  class MicroKernel : protected jblas::xbyak::JitAvx512vnni {
   public:
    MicroKernel() {}
    int CRegCount = 12, BRegCount = 3, ARegCount = 1;
    int CReg = 0, CF32Reg = 12, BReg = 24, AReg = 27, ZpTmp = 28;
    int const NRegs = 3;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);
    static int constexpr VecBytes = 64;

    void generate_code(int _mtile, JBLAS_DTYPE _scale_dt) {
      mScaleType = _scale_dt;
      reset();
      generate_mtile(_mtile);
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    JBLAS_DTYPE mScaleType = JblasF32;

    void generate_mtile(int _mtile) {
      CRegCount = _mtile * NRegs;
      BRegCount = NRegs;
      CF32Reg = CReg + CRegCount;
      BReg = CF32Reg + CRegCount;
      AReg = BReg + BRegCount;
      ZpTmp = AReg + ARegCount;
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 13, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_iterkb = st.t[3];
      reg_iterk = st.t[4];
      reg_astep = st.t[5];
      reg_tmp = st.t[7];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[9];
      reg_tmp3 = st.t[6];
      reg_tmp4 = st.t[10];
      reg_ret = rax;

      vreg_push(rsp);

      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      xor_(reg_iterk, reg_iterk);

      init_accumulator(_mtile);

      generate_kloop(_mtile, NRegs);
      write_back(_mtile);

      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _mtile, int _nregs) {
      inLocalLabel();
      xor_(reg_iterkb, reg_iterkb);
      L(".kloop");
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _nregs; j++) {
          vpxorq(Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j), Xbyak::Zmm(CReg + i * NRegs + j));
        }
      }

      xor_(reg_tmp2, reg_tmp2);
      load32(reg_tmp3, ptr[parambase + OFFSET(kblock)]);
      L(".kbloop");
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp, reg_matAptr, reg_matBptr, reg_astep);
      add(reg_matAptr, AKStepSize * KUNROLL);
      add(reg_matBptr, BKStepSize * KUNROLL);
      add(reg_iterk, KTILE * KUNROLL);
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jge(".kbend");
      add(reg_tmp2, KTILE * KUNROLL);
      cmp(reg_tmp2, reg_tmp3);
      jb(".kbloop");
      L(".kbend");
      generate_f32_accumulate(_mtile);
      generate_zp_correction(_mtile);
      add(reg_iterkb, 1);         // next kblock
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _kunroll, const Xbyak::Reg64& reg_tmp,
                      const Xbyak::Reg64& reg_matAptr, const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      for (int kk = 0; kk < _kunroll; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Zmm(BReg + i), ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastd(Xbyak::Zmm(AReg), ptr[reg_tmp]);
          add(reg_tmp, reg_astep);
          for (int i = 0; i < _NRegs; i++) {
            vpdpbusds(Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(AReg), Xbyak::Zmm(BReg + i));
          }
        }
      }
    }

    void generate_f32_accumulate(int _mtile) {
      load32(reg_tmp, ptr[parambase + OFFSET(ldsb)]);
      imul(reg_tmp, reg_iterkb);
      mov(reg_tmp2, ptr[parambase + OFFSET(scaleB)]);
      if (mScaleType == JblasF32) {
        lea(reg_tmp2, ptr[reg_tmp2 + reg_tmp * sizeof(float)]);
      } else if (mScaleType == JblasBF16) {
        lea(reg_tmp2, ptr[reg_tmp2 + reg_tmp * sizeof(utils::bf16)]);
      }

      mov(reg_tmp, ptr[parambase + OFFSET(scaleA)]);
      lea(reg_tmp, ptr[reg_tmp + reg_iterkb * sizeof(float)]);
      load32(reg_tmp1, ptr[parambase + OFFSET(ldsa)]);
      for (int i = 0; i < NRegs; i++) {
        if (mScaleType == JblasF32) {
          vmovups(Xbyak::Zmm(BReg + i), ptr[reg_tmp2 + i * VecBytes]);
        } else if (mScaleType == JblasBF16) {
          loadbf16_f32(Xbyak::Zmm(BReg + i), ptr[reg_tmp2 + i * VecBytes / 2]);
        }
      }
      for (int mm = 0; mm < _mtile; mm++) {
        vbroadcastss(Xbyak::Zmm(ZpTmp), ptr[reg_tmp]);
        lea(reg_tmp, ptr[reg_tmp + reg_tmp1 * sizeof(float)]);
        for (int i = 0; i < NRegs; i++) {
          vcvtdq2ps(Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(CReg + mm * NRegs + i));
          vmulps(Xbyak::Zmm(AReg), Xbyak::Zmm(ZpTmp), Xbyak::Zmm(BReg + i));
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

    void init_accumulator(int _mtile) {
      inLocalLabel();
      load32(reg_tmp1, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_tmp1, 0);
      jg(".LACC", T_NEAR);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vpxorq(Xbyak::Zmm(CF32Reg + i * NRegs + j), Xbyak::Zmm(CF32Reg + i * NRegs + j),
                 Xbyak::Zmm(CF32Reg + i * NRegs + j));
        }
      }
      jmp(".LEND", T_NEAR);
      L(".LACC");
      load32(reg_tmp, ptr[parambase + OFFSET(cstep)]);
      mov(reg_tmp1, ptr[parambase + OFFSET(matC)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vmovups(Xbyak::Zmm(CF32Reg + i * NRegs + j), ptr[reg_tmp1 + j * VecBytes]);
        }
        add(reg_tmp1, reg_tmp);
      }
      L(".LEND");
      outLocalLabel();
    }

    void write_back(int _mtile) {
      inLocalLabel();
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      load32(reg_tmp, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CF32Reg + i * NRegs + j));
        }
        add(reg_matCptr, reg_tmp);
      }
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr, reg_matBptr, reg_matCptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_iterkb;
    Xbyak::Reg64 reg_tmp, reg_tmp1, reg_tmp2, reg_tmp3, reg_tmp4;
    Xbyak::Reg64 reg_ret = rax;
  };

 public:
  GemmCore_Row_NN_4x48_AVX512_VNNI_KBLOCK() {
    for (int i = 0; i < MTILE; i++) {
      mCodes[i].generate_code(i + 1, JblasF32);
      mCodesBf16[i].generate_code(i + 1, JblasBF16);
    }
  }

  void forward(AType* matA, BType* matB, CType* matC, AType* zpA, float* scaleA, int _ldsa, float* scaleB,
               float* reduceB, int _ldsb, int _m, int _n, int _k, int _kblock, int _astride, int _bstride, int _cstride,
               int kpos) {
    int ldb = _bstride / sizeof(BType);
    auto param =
        params{matA, matB, matC, zpA, scaleA, scaleB, reduceB, _ldsa, _ldsb, _kblock, _k, _n, _astride, _cstride, kpos};
    if (_m <= MTILE) {
      for (int i = 0; i < _n; i += NTILE) {
        param.matB = matB + i * ldb;
        param.matC = matC + i;
        param.nsize = i + NTILE <= _n ? NTILE : _n - i;
        param.scaleB = scaleB + i;
        param.reduceB = reduceB + i;
        mCodes[_m - 1].mKernel(&param);
      }
    } else {
      assert(0);
    }
  }

  void forward(AType* matA, BType* matB, CType* matC, AType* zpA, float* scaleA, int _ldsa, utils::bf16* scaleB,
               float* reduceB, int _ldsb, int _m, int _n, int _k, int _kblock, int _astride, int _bstride, int _cstride,
               int kpos) {
    int ldb = _bstride / sizeof(BType);
    auto param =
        params{matA, matB, matC, zpA, scaleA, scaleB, reduceB, _ldsa, _ldsb, _kblock, _k, _n, _astride, _cstride, kpos};
    if (_m <= MTILE) {
      for (int i = 0; i < _n; i += NTILE) {
        param.matB = matB + i * ldb;
        param.matC = matC + i;
        param.nsize = i + NTILE <= _n ? NTILE : _n - i;
        param.scaleB = scaleB + i;
        param.reduceB = reduceB + i;
        mCodesBf16[_m - 1].mKernel(&param);
      }
    } else {
      assert(0);
    }
  }

 private:
  std::array<MicroKernel, MTILE> mCodes, mCodesBf16;
};

class GemmCore_Row_NN_1x48_AVX_VNNI_KBLOCK {
 public:
  typedef uint8_t AType;
  typedef int8_t BType;
  typedef float CType;
  struct params {
    uint8_t* matA;
    int8_t* matB;
    CType* matC;
    uint8_t* zpA;
    float* scaleA;
    void* scaleB;
    float* reduceB;
    int ldsa, ldsb;
    int kblock;
    int k, nsize;
    int astep, cstep;
    int kpos;
  };
  typedef long long (*func_t)(params*);

  static JBLAS_ISA constexpr ISA = JblasAVX_VNNI;
  static GemmCoreType constexpr TYPE = GemmCoreType::AVX_VNNI_1x48_KBLOCK;
  static int constexpr NTILE = 48, MTILE = 1, KTILE = 4 / sizeof(BType);
  static int constexpr PACK_ROW = KTILE;
  static int constexpr KUNROLL = 2;
  static int constexpr PREFERED_N = 192;

  class MicroKernel : protected jblas::xbyak::JitAvxvnni {
   public:
    MicroKernel() {}
    int CRegCount = 6, BRegCount = 1, ARegCount = 1;
    int CReg = 0, CF32Reg = 6, BReg = 12, AReg = 13, ZpTmp = 14;
    int const NRegs = 6;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);
    static int constexpr VecBytes = 32;

    void generate_code(int _mtile, JBLAS_DTYPE _scale_dt) {
      mScaleType = _scale_dt;
      reset();
      generate_mtile(_mtile);
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    JBLAS_DTYPE mScaleType = JblasF32;

    void generate_mtile(int _mtile) {
      CRegCount = _mtile * NRegs;
      CF32Reg = CReg + CRegCount;
      BReg = CF32Reg + CRegCount;
      AReg = BReg + BRegCount;
      ZpTmp = AReg + ARegCount;
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 13, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_iterkb = st.t[3];
      reg_iterk = st.t[4];
      reg_astep = st.t[5];
      reg_tmp = st.t[7];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[9];
      reg_tmp3 = st.t[6];
      reg_tmp4 = st.t[10];
      reg_ret = rax;

      vreg_push(rsp);

      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);

      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      xor_(reg_iterk, reg_iterk);

      init_accumulator(_mtile);

      generate_kloop(_mtile, NRegs);
      write_back(_mtile);

      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void generate_kloop(int _mtile, int _nregs) {
      inLocalLabel();
      xor_(reg_iterkb, reg_iterkb);
      L(".kloop");
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < _nregs; j++) {
          vxorps(Xbyak::Ymm(CReg + i * NRegs + j), Xbyak::Ymm(CReg + i * NRegs + j), Xbyak::Ymm(CReg + i * NRegs + j));
        }
      }

      xor_(reg_tmp2, reg_tmp2);
      load32(reg_tmp3, ptr[parambase + OFFSET(kblock)]);
      L(".kbloop");
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp, reg_matAptr, reg_matBptr, reg_astep);
      add(reg_matAptr, AKStepSize * KUNROLL);
      add(reg_matBptr, BKStepSize * KUNROLL);
      add(reg_iterk, KTILE * KUNROLL);
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jge(".kbend");
      add(reg_tmp2, KTILE * KUNROLL);
      cmp(reg_tmp2, reg_tmp3);
      jb(".kbloop");
      L(".kbend");
      generate_f32_accumulate(_mtile);
      generate_zp_correction(_mtile);
      add(reg_iterkb, 1);         // next kblock
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _kunroll, const Xbyak::Reg64& reg_tmp,
                      const Xbyak::Reg64& reg_matAptr, const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      for (int kk = 0; kk < _kunroll; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastd(Xbyak::Ymm(AReg), ptr[reg_tmp]);
          if (mm != _mtile - 1) {
            add(reg_tmp, reg_astep);
          }
          for (int i = 0; i < _NRegs; i++) {
            vpdpbusds(Xbyak::Ymm(CReg + mm * NRegs + i), Xbyak::Ymm(AReg),
                      ptr[reg_matBptr + kk * BKStepSize + i * VecBytes]);
          }
        }
      }
    }

    void generate_f32_accumulate(int _mtile) {
      load32(reg_tmp, ptr[parambase + OFFSET(ldsb)]);
      imul(reg_tmp, reg_iterkb);
      mov(reg_tmp2, ptr[parambase + OFFSET(scaleB)]);
      if (mScaleType == JblasF32) {
        lea(reg_tmp2, ptr[reg_tmp2 + reg_tmp * sizeof(float)]);
      } else if (mScaleType == JblasBF16) {
        lea(reg_tmp2, ptr[reg_tmp2 + reg_tmp * sizeof(utils::bf16)]);
      }

      mov(reg_tmp, ptr[parambase + OFFSET(scaleA)]);
      lea(reg_tmp, ptr[reg_tmp + reg_iterkb * sizeof(float)]);
      load32(reg_tmp1, ptr[parambase + OFFSET(ldsa)]);
      for (int mm = 0; mm < _mtile; mm++) {
        vbroadcastss(Xbyak::Ymm(ZpTmp), ptr[reg_tmp]);
        lea(reg_tmp, ptr[reg_tmp + reg_tmp1 * sizeof(float)]);
        for (int i = 0; i < NRegs; i++) {
          if (mScaleType == JblasF32) {
            vmovups(Xbyak::Ymm(BReg), ptr[reg_tmp2 + i * VecBytes]);
          } else if (mScaleType == JblasBF16) {
            loadbf16_f32(Xbyak::Ymm(BReg), ptr[reg_tmp2 + i * VecBytes / 2]);
          }
          vcvtdq2ps(Xbyak::Ymm(CReg + mm * NRegs + i), Xbyak::Ymm(CReg + mm * NRegs + i));
          vmulps(Xbyak::Ymm(AReg), Xbyak::Ymm(ZpTmp), Xbyak::Ymm(BReg));
          vmulps(Xbyak::Ymm(CReg + mm * NRegs + i), Xbyak::Ymm(AReg));
          vaddps(Xbyak::Ymm(CF32Reg + mm * NRegs + i), Xbyak::Ymm(CReg + mm * NRegs + i));
        }
      }
    }

    void generate_zp_correction(int _mtile) {
      load32(reg_tmp1, ptr[parambase + OFFSET(ldsb)]);
      imul(reg_tmp1, reg_iterkb);
      mov(reg_tmp2, ptr[parambase + OFFSET(reduceB)]);
      lea(reg_tmp2, ptr[reg_tmp2 + reg_tmp1 * sizeof(float)]);
      auto& reg_redB = reg_tmp2;

      mov(reg_tmp, ptr[parambase + OFFSET(zpA)]);
      lea(reg_tmp, ptr[reg_tmp + reg_iterkb * sizeof(AType)]);
      auto& reg_zpA = reg_tmp;

      mov(reg_tmp1, ptr[parambase + OFFSET(scaleA)]);
      lea(reg_tmp1, ptr[reg_tmp1 + reg_iterkb * sizeof(float)]);
      auto& reg_scaleA = reg_tmp1;

      load32(reg_tmp3, ptr[parambase + OFFSET(ldsa)]);
      auto& reg_ldsa = reg_tmp3;

      for (int i = 0; i < _mtile; i++) {
        vpbroadcastb(Xbyak::Xmm(AReg), ptr[reg_zpA]);
        vpmovzxbd(Xbyak::Ymm(AReg), Xbyak::Xmm(AReg));
        vcvtdq2ps(Xbyak::Ymm(AReg), Xbyak::Ymm(AReg));
        vmulps(Xbyak::Ymm(AReg), Xbyak::Ymm(AReg), zword_b[reg_scaleA]);
        for (int j = 0; j < NRegs; j++) {
          vmulps(Xbyak::Ymm(CReg + j), Xbyak::Ymm(AReg), ptr[reg_redB + j * VecBytes]);
          vsubps(Xbyak::Ymm(CF32Reg + i * NRegs + j), Xbyak::Ymm(CReg + j));
        }
        lea(reg_zpA, ptr[reg_zpA + reg_ldsa * sizeof(AType)]);
        lea(reg_scaleA, ptr[reg_scaleA + reg_ldsa * sizeof(float)]);
      }
    }

    void init_accumulator(int _mtile) {
      inLocalLabel();
      load32(reg_tmp1, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_tmp1, 0);
      jg(".LACC", T_NEAR);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vxorps(Xbyak::Ymm(CF32Reg + i * NRegs + j), Xbyak::Ymm(CF32Reg + i * NRegs + j),
                 Xbyak::Ymm(CF32Reg + i * NRegs + j));
        }
      }
      jmp(".LEND", T_NEAR);
      L(".LACC");
      load32(reg_tmp, ptr[parambase + OFFSET(cstep)]);
      mov(reg_tmp1, ptr[parambase + OFFSET(matC)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vmovups(Xbyak::Ymm(CF32Reg + i * NRegs + j), ptr[reg_tmp1 + j * VecBytes]);
        }
        add(reg_tmp1, reg_tmp);
      }
      L(".LEND");
      outLocalLabel();
    }

    void write_back(int _mtile) {
      inLocalLabel();
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      load32(reg_tmp, ptr[parambase + OFFSET(cstep)]);
      for (int i = 0; i < _mtile; i++) {
        for (int j = 0; j < NRegs; j++) {
          vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Ymm(CF32Reg + i * NRegs + j));
        }
        add(reg_matCptr, reg_tmp);
      }
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr, reg_matBptr, reg_matCptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_iterkb;
    Xbyak::Reg64 reg_tmp, reg_tmp1, reg_tmp2, reg_tmp3, reg_tmp4;
    Xbyak::Reg64 reg_ret = rax;
  };

 public:
  GemmCore_Row_NN_1x48_AVX_VNNI_KBLOCK() {
    for (int i = 0; i < MTILE; i++) {
      mCodes[i].generate_code(i + 1, JblasF32);
      mCodesBf16[i].generate_code(i + 1, JblasBF16);
    }
  }

  void forward(AType* matA, BType* matB, CType* matC, AType* zpA, float* scaleA, int _ldsa, float* scaleB,
               float* reduceB, int _ldsb, int _m, int _n, int _k, int _kblock, int _astride, int _bstride, int _cstride,
               int kpos) {
    int ldb = _bstride / sizeof(BType);
    auto param =
        params{matA, matB, matC, zpA, scaleA, scaleB, reduceB, _ldsa, _ldsb, _kblock, _k, _n, _astride, _cstride, kpos};
    if (_m <= MTILE) {
      for (int i = 0; i < _n; i += NTILE) {
        param.matB = matB + i * ldb;
        param.matC = matC + i;
        param.nsize = i + NTILE <= _n ? NTILE : _n - i;
        param.scaleB = scaleB + i;
        param.reduceB = reduceB + i;
        mCodes[_m - 1].mKernel(&param);
      }
    } else {
      assert(0);
    }
  }

  void forward(AType* matA, BType* matB, CType* matC, AType* zpA, float* scaleA, int _ldsa, utils::bf16* scaleB,
               float* reduceB, int _ldsb, int _m, int _n, int _k, int _kblock, int _astride, int _bstride, int _cstride,
               int kpos) {
    int ldb = _bstride / sizeof(BType);
    auto param =
        params{matA, matB, matC, zpA, scaleA, scaleB, reduceB, _ldsa, _ldsb, _kblock, _k, _n, _astride, _cstride, kpos};
    if (_m <= MTILE) {
      for (int i = 0; i < _n; i += NTILE) {
        param.matB = matB + i * ldb;
        param.matC = matC + i;
        param.nsize = i + NTILE <= _n ? NTILE : _n - i;
        param.scaleB = scaleB + i;
        param.reduceB = reduceB + i;
        mCodesBf16[_m - 1].mKernel(&param);
      }
    } else {
      assert(0);
    }
  }

 private:
  std::array<MicroKernel, MTILE> mCodes, mCodesBf16;
};

class GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK {
 public:
  typedef int8_t AType;
  typedef int8_t BType;
  typedef float CType;
  struct params {
    AType* matA;
    BType* matB;
    CType* matC;
    float* scaleA;
    void* scaleB;
    int ldsa, ldsb;
    int kblock;
    int k, nsize, msize;
    int astep, cstep;
    int kpos;
    void *workspace, *cfg;
  };

  typedef long long (*func_t)(params*);

  static JBLAS_ISA constexpr ISA = JblasAMX_INT8;
  static GemmCoreType constexpr TYPE = GemmCoreType::AMX_INT8_16x48_KBLOCK;
  static int constexpr NTILE = 48, MTILE = 16, KTILE = 64 / sizeof(BType);
  static int constexpr PACK_ROW = 4;
  static int constexpr KUNROLL = 2;
  static int constexpr PREFERED_N = 256;

  class MicroKernel : protected jblas::xbyak::JitAmxint8 {
   public:
    friend GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK;
    MicroKernel() {}
    static int constexpr CReg = 0, TmpReg = 3;
    static int constexpr NRegs = 3;
    static int constexpr C_tilenum = 4, A_tilenum = 1, B_tilenum = 3;
    static int constexpr CTile = 0, ATile = CTile + C_tilenum, BTile = ATile + A_tilenum;
    static int constexpr BKStepSize = KTILE * NTILE * sizeof(BType);
    static int constexpr AKStepSize = KTILE * sizeof(AType);
    static int constexpr VecBytes = 64;

    void generate_code(JBLAS_DTYPE scaletype) {
      mScaleType = scaletype;
      reset();
      generate_mtile();
      ready();
      mKernel = getCode<func_t>();
    }
    func_t mKernel = nullptr;

   protected:
    JBLAS_DTYPE mScaleType = JblasF32;

    void generate_mtile() {
      inLocalLabel();  // use local label for multiple instance
      Xbyak::util::StackFrame st(this, 1, 13, 16 * 10);
      parambase = st.p[0];
      reg_matAptr = st.t[0];
      reg_matBptr = st.t[1];
      reg_matCptr = st.t[0];
      reg_ksize = st.t[2];
      reg_cstep = st.t[3];
      reg_iterk = st.t[4];
      reg_astep = st.t[5];
      reg_kblock = st.t[6];
      reg_tmp = st.t[7];
      reg_tmp1 = st.t[8];
      reg_tmp2 = st.t[9];
      reg_tmp3 = st.t[10];
      reg_scaleAptr = st.t[11];
      reg_scaleBptr = st.t[12];
      reg_ret = rax;

      vreg_push(rsp);
      mov(reg_tmp, ptr[parambase + OFFSET(cfg)]);
      ldtilecfg(ptr[reg_tmp]);

      load32(reg_ksize, ptr[parambase + OFFSET(k)]);
      load32(reg_kblock, ptr[parambase + OFFSET(kblock)]);
      load32(reg_astep, ptr[parambase + OFFSET(astep)]);
      load32(reg_cstep, ptr[parambase + OFFSET(cstep)]);

      mov(reg_matAptr, ptr[parambase + OFFSET(matA)]);
      mov(reg_matBptr, ptr[parambase + OFFSET(matB)]);
      mov(reg_scaleAptr, ptr[parambase + OFFSET(scaleA)]);
      mov(reg_scaleBptr, ptr[parambase + OFFSET(scaleB)]);
      xor_(reg_iterk, reg_iterk);

      load32(reg_tmp, ptr[parambase + OFFSET(nsize)]);
      cmp(reg_tmp, NTILE);
      jl(".n32", T_NEAR);
      init_accumulator_mem(NRegs);
      generate_kloop(NRegs);
      jmp(".nend", T_NEAR);

      L(".n32");
      cmp(reg_tmp, 32);
      jl(".n16", T_NEAR);
      init_accumulator_mem(2);
      generate_kloop(2);
      jmp(".nend", T_NEAR);

      L(".n16");
      init_accumulator_mem(1);
      generate_kloop(1);

      L(".nend");
      mov(reg_ret, 0);
      vreg_pop(rsp);

      outLocalLabel();  // end of local label
    }

    void init_accumulator_mem(int _NRegs) {
      inLocalLabel();
      push(reg_matCptr);
      load32(reg_matCptr, ptr[parambase + OFFSET(kpos)]);
      cmp(reg_matCptr, 0);
      jg(".END", T_NEAR);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      load32(reg_tmp, ptr[parambase + OFFSET(msize)]);
      for (int j = 0; j < _NRegs; j++) {
        vxorps(Xbyak::Zmm(CReg + j), Xbyak::Zmm(CReg + j));
      }
      xor_(reg_tmp1, reg_tmp1);
      L(".mloop");
      for (int j = 0; j < _NRegs; j++) {
        vmovups(ptr[reg_matCptr + j * VecBytes], Xbyak::Zmm(CReg + j));
      }
      add(reg_matCptr, reg_cstep);
      add(reg_tmp1, 1);
      cmp(reg_tmp1, reg_tmp);
      jb(".mloop");
      L(".END");
      pop(reg_matCptr);
      outLocalLabel();
    }

    void generate_kloop(int _nregs) {
      inLocalLabel();
      L(".kloop");
      for (int i = 0; i < C_tilenum; i++) {
        tilezero(Xbyak::Tmm(CTile + i));
      }
      xor_(reg_tmp2, reg_tmp2);
      L(".kbloop");
      generate_fma(_nregs, KUNROLL, reg_tmp, reg_matAptr, reg_matBptr, reg_astep);
      add(reg_matAptr, AKStepSize * KUNROLL);
      add(reg_matBptr, BKStepSize * KUNROLL);
      add(reg_iterk, KTILE * KUNROLL);
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jge(".kbend");
      add(reg_tmp2, KTILE * KUNROLL);
      cmp(reg_tmp2.cvt32(), ptr[parambase + OFFSET(kblock)]);
      jb(".kbloop");
      L(".kbend");
      generate_f32_accumulate(_nregs);
      add(reg_scaleAptr, sizeof(float));
      load32(reg_tmp, ptr[parambase + OFFSET(ldsb)]);
      if (mScaleType == JblasBF16) {
        lea(reg_scaleBptr, ptr[reg_scaleBptr + reg_tmp * 2]);
      } else if (mScaleType == JblasF32) {
        lea(reg_scaleBptr, ptr[reg_scaleBptr + reg_tmp * 4]);
      }
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _NNum, int _kunroll, const Xbyak::Reg64& reg_tmp, const Xbyak::Reg64& reg_matAptr,
                      const Xbyak::Reg64& reg_matBptr, const Xbyak::Reg64& reg_astep) {
      mov(reg_tmp, NTILE * 4);
      for (int kk = 0; kk < _kunroll; kk++) {
        for (int i = 0; i < _NNum; i++) {
          tileloaddt1(Xbyak::Tmm(BTile + i), ptr[reg_matBptr + reg_tmp + kk * BKStepSize + i * 64]);
        }

        for (int mm = 0; mm < 1; mm++) {
          tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_matAptr + reg_astep + kk * AKStepSize]);
          for (int i = 0; i < _NNum; i++) {
            tdpbssd(Xbyak::Tmm(CTile + mm * C_tilenum + i), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile + i));
          }
        }
      }
    }

    void generate_f32_accumulate(int _NRegs) {
      inLocalLabel();
      push(reg_matCptr);
      push(reg_astep);
      load32(reg_astep, ptr[parambase + OFFSET(msize)]);
      mov(reg_matCptr, ptr[parambase + OFFSET(matC)]);
      mov(reg_tmp3, qword[parambase + OFFSET(workspace)]);
      mov(reg_tmp1, NTILE * 4);
      for (int mm = 0; mm < 1; mm++) {
        for (int i = 0; i < _NRegs; i++) {
          tilestored(ptr[reg_tmp3 + reg_tmp1 + i * 64 + mm * 16 * NTILE * 4], Xbyak::Tmm(CTile + mm * C_tilenum + i));
        }
      }

      for (int i = 0; i < _NRegs; i++) {
        if (mScaleType == JblasF32) {
          vmovups(Xbyak::Zmm(TmpReg + i), ptr[reg_scaleBptr + i * VecBytes]);
        } else if (mScaleType == JblasBF16) {
          loadbf16_f32(Xbyak::Zmm(TmpReg + i), ptr[reg_scaleBptr + i * VecBytes / 2]);
        }
      }
      mov(reg_tmp, reg_scaleAptr);
      load32(reg_tmp2, ptr[parambase + OFFSET(ldsa)]);
      xor_(reg_tmp1, reg_tmp1);
      L(".mloop");
      for (int i = 0; i < _NRegs; i++) {
        vcvtdq2ps(Xbyak::Zmm(CReg + i), zword[reg_tmp3 + i * VecBytes]);
        vmulps(Xbyak::Zmm(TmpReg + _NRegs), Xbyak::Zmm(TmpReg + i), zword_b[reg_tmp]);
        vmulps(Xbyak::Zmm(CReg + i), Xbyak::Zmm(TmpReg + _NRegs));
        vaddps(Xbyak::Zmm(CReg + i), zword[reg_matCptr + i * VecBytes]);
        vmovups(zword[reg_matCptr + i * VecBytes], Xbyak::Zmm(CReg + i));
      }
      add(reg_tmp3, NTILE * sizeof(CType));
      add(reg_matCptr, reg_cstep);
      lea(reg_tmp, ptr[reg_tmp + reg_tmp2 * sizeof(float)]);
      add(reg_tmp1, 1);
      cmp(reg_tmp1.cvt32(), reg_astep);
      jb(".mloop");
      pop(reg_astep);
      pop(reg_matCptr);
      outLocalLabel();
    }

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_matAptr;
    Xbyak::Reg64 reg_matBptr;
    Xbyak::Reg64 reg_matCptr;
    Xbyak::Reg64 reg_scaleAptr;
    Xbyak::Reg64 reg_scaleBptr;
    Xbyak::Reg64 reg_ksize;
    Xbyak::Reg64 reg_kblock;
    Xbyak::Reg64 reg_cstep;
    Xbyak::Reg64 reg_astep;
    Xbyak::Reg64 reg_iterk;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_tmp2;
    Xbyak::Reg64 reg_tmp3;
    Xbyak::Reg64 reg_ret = rax;
  };

 public:
  GemmCore_Row_NN_16x48_AMX_INT8_KBLOCK() {
    mCodes.generate_code(JblasF32);
    mCodesBf16.generate_code(JblasBF16);
  }

  void forward(AType* matA, BType* matB, CType* matC, AType* zpA, float* scaleA, int _ldsa, float* scaleB, int _ldsb,
               int _m, int _n, int _k, int _kblock, int _astride, int _bstride, int _cstride, int kpos) {
    (void)zpA;  // keep the same parameter structure
    int ldb = _bstride / sizeof(BType);
    char tmp[NTILE * MTILE * sizeof(CType) * 2];  // s32+f32
    MicroKernel::tileconfig_t mCfg;
    memset(&mCfg, 0, sizeof(mCfg));
    MicroKernel::configure_tiles(mCfg, _m < 16 ? _m : 16, 16, _k < KTILE ? _k : KTILE, sizeof(BType),
                                 MicroKernel::A_tilenum, MicroKernel::B_tilenum, MicroKernel::C_tilenum);

    auto param = params{matA, matB, matC, scaleA,   scaleB,   _ldsa, _ldsb, _kblock,
                        _k,   _n,   _m,   _astride, _cstride, kpos,  tmp,   &mCfg};
    if (_m <= MTILE) {
      for (int i = 0; i < _n; i += NTILE) {
        param.matB = matB + i * ldb;
        param.matC = matC + i;
        param.nsize = i + NTILE <= _n ? NTILE : _n - i;
        param.scaleB = scaleB + i;

        mCodes.mKernel(&param);
      }
    } else {
      assert(0);
    }
  }

  void forward(AType* matA, BType* matB, CType* matC, AType* zpA, float* scaleA, int _ldsa, utils::bf16* scaleB,
               int _ldsb, int _m, int _n, int _k, int _kblock, int _astride, int _bstride, int _cstride, int kpos) {
    (void)zpA;  // keep the same parameter structure
    int ldb = _bstride / sizeof(BType);
    char tmp[NTILE * MTILE * sizeof(CType) * 2];  // s32+f32
    MicroKernel::tileconfig_t mCfg;
    memset(&mCfg, 0, sizeof(mCfg));
    MicroKernel::configure_tiles(mCfg, _m < 16 ? _m : 16, 16, _k < KTILE ? _k : KTILE, sizeof(BType),
                                 MicroKernel::A_tilenum, MicroKernel::B_tilenum, MicroKernel::C_tilenum);

    auto param = params{matA, matB, matC, scaleA,   scaleB,   _ldsa, _ldsb, _kblock,
                        _k,   _n,   _m,   _astride, _cstride, kpos,  tmp,   &mCfg};
    if (_m <= MTILE) {
      for (int i = 0; i < _n; i += NTILE) {
        param.matB = matB + i * ldb;
        param.matC = matC + i;
        param.nsize = i + NTILE <= _n ? NTILE : _n - i;
        param.scaleB = scaleB + i;

        mCodesBf16.mKernel(&param);
      }
    } else {
      assert(0);
    }
  }

  void reference(AType* matA, BType* matB, CType* matC, float* scaleA, int _ldsa, float* scaleB, int _ldsb, int _m,
                 int _n, int _k, int _kblock, int _astride, int _bstride, int _cstride, int kpos) {
    int lda = _astride / sizeof(matA[0]);
    int ldb = _bstride / sizeof(matB[0]);
    int ldc = _cstride / sizeof(matC[0]);
    for (int i = 0; i < _m; i++) {
      for (int j = 0; j < _n; j += NTILE) {
        for (int ij = 0; ij < NTILE; ij++) {
          if (j + ij >= _n) {
            break;
          }
          float tmpf = 0.f;
          for (int k = 0; k < _k; k += _kblock) {
            int tmp = 0;
            for (int ik = 0; ik < _kblock; ik += 4) {
              if (k + ik >= _k) {
                break;
              }
              for (int ikk = 0; ikk < 4; ikk++) {
                tmp += (int(matA[i * lda + k + ik + ikk])) * int(matB[(k + ik) * NTILE + ij * 4 + ikk + j * ldb]);
              }
            }
            tmpf += tmp * scaleA[i * _ldsa + k / _kblock] * scaleB[j + ij + k / _kblock * _ldsb];
          }
          matC[i * ldc + j + ij] = tmpf;
        }
      }
    }
  }
  void reference(AType* matA, BType* matB, CType* matC, float* scaleA, int _ldsa, utils::bf16* scaleB, int _ldsb,
                 int _m, int _n, int _k, int _kblock, int _astride, int _bstride, int _cstride, int kpos) {
    int lda = _astride / sizeof(matA[0]);
    int ldb = _bstride / sizeof(matB[0]);
    int ldc = _cstride / sizeof(matC[0]);
    for (int i = 0; i < _m; i++) {
      for (int j = 0; j < _n; j += NTILE) {
        for (int ij = 0; ij < NTILE; ij++) {
          if (j + ij >= _n) {
            break;
          }
          float tmpf = 0.f;
          for (int k = 0; k < _k; k += _kblock) {
            int tmp = 0;
            for (int ik = 0; ik < _kblock; ik += 4) {
              if (k + ik >= _k) {
                break;
              }
              for (int ikk = 0; ikk < 4; ikk++) {
                tmp += (int(matA[i * lda + k + ik + ikk])) * int(matB[(k + ik) * NTILE + ij * 4 + ikk + j * ldb]);
              }
            }
            tmpf += tmp * scaleA[i * _ldsa + k / _kblock] * scaleB[j + ij + k / _kblock * _ldsb].tofloat();
          }
          matC[i * ldc + j + ij] = tmpf;
        }
      }
    }
  }

 private:
  MicroKernel mCodes, mCodesBf16;
};

}  // namespace kblock

static inline size_t getWeightSize(GemmCoreType _type) {
  switch (_type) {
    case jblas::gemm::GemmCoreType::AVX2_4X24:
    case jblas::gemm::GemmCoreType::AVX2_2X48:
    case jblas::gemm::GemmCoreType::AVX512F_8x48:
      return 4;
    case jblas::gemm::GemmCoreType::AVX_VNNI_2x48:
    case jblas::gemm::GemmCoreType::AVX_VNNI_1x48_KBLOCK:
    case jblas::gemm::GemmCoreType::AVX512_VNNI_8x48:
    case jblas::gemm::GemmCoreType::AMX_INT8_16x64:
    case jblas::gemm::GemmCoreType::AMX_INT8_16x48:
    case jblas::gemm::GemmCoreType::AMX_INT8_16x48_SS:
    case jblas::gemm::GemmCoreType::AMX_INT8_16x48_KBLOCK:
    case jblas::gemm::GemmCoreType::AVX512_VNNI_3x48_KBLOCK:
    case jblas::gemm::GemmCoreType::AVX512_VNNI_4x48_KBLOCK:
      return 1;
    case jblas::gemm::GemmCoreType::AMX_BF16_16x48:
    case jblas::gemm::GemmCoreType::AMX_BF16_16x64:
    case jblas::gemm::GemmCoreType::AVX512_FP16_8x64:
    case jblas::gemm::GemmCoreType::AVX512_FP16_8x96:
      return 2;
    case jblas::gemm::GemmCoreType::Undef:
    default:
      return 0;
  }
}

}  // namespace gemm
}  // namespace jblas
