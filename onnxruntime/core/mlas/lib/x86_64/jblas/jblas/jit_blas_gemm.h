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
  static JBLAS_GEMM_CORE constexpr TYPE = JBLAS_GEMM_CORE::AVX2_4X24;
  static JBLAS_ISA constexpr ISA =
      utils::gemm_core_mask<TYPE, JBLAS_GEMM_CORE::ISA_MASK, JBLAS_GEMM_CORE::ISA_SHIFT, JBLAS_ISA>();
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
      write_back(_mtile, NRegs);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n16");
      cmp(reg_tmp, 16);
      jl(".n8", T_NEAR);
      generate_kloop(_mtile, 2);
      write_back(_mtile, 2);
      add(reg_itern, 16);
      add(reg_matBptr, 16 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n8");
      xor_(reg_iterk, reg_iterk);
      generate_kloop(_mtile, 1);
      write_back(_mtile, 1);
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
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp1);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_mtile, _nregs, 1, reg_tmp1);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _ktile, const Xbyak::Reg64& _reg_matBptr) {
      for (int kk = 0; kk < _ktile; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Ymm(BReg + i), ptr[_reg_matBptr + kk * BKStepSize + i * VecBytes]);
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

    void write_back(int _mtile, int _NRegs) {
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
               int kpos, void* tmpcache, size_t cachesize) {
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
  static JBLAS_GEMM_CORE constexpr TYPE = JBLAS_GEMM_CORE::AVX2_2X48;
  static JBLAS_ISA constexpr ISA =
      utils::gemm_core_mask<TYPE, JBLAS_GEMM_CORE::ISA_MASK, JBLAS_GEMM_CORE::ISA_SHIFT, JBLAS_ISA>();
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
      write_back(_mtile, NRegs);
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
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp1);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_mtile, _nregs, 1, reg_tmp1);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _ktile, const Xbyak::Reg64& _reg_matBptr) {
      for (int kk = 0; kk < _ktile; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _mtile; ++i)
          vbroadcastss(Xbyak::Ymm(AReg + i), ptr[reg_matAptr + kk * AKStepSize + reg_astep * i]);
        for (int j = 0; j < _NRegs; j++) {
          vmovups(Xbyak::Ymm(BReg), ptr[_reg_matBptr + kk * BKStepSize + j * VecBytes]);
          for (int i = 0; i < _mtile; ++i)
            vfmadd231ps(Xbyak::Ymm(CReg + i * NRegs + j), Xbyak::Ymm(BReg), Xbyak::Ymm(AReg + i));
        }
      }
    }

    void write_back(int _mtile, int _NRegs) {
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
               int kpos, void* tmpcache, size_t cachesize) {
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
  static JBLAS_GEMM_CORE constexpr TYPE = JBLAS_GEMM_CORE::AVX_VNNI_4x24;
  static JBLAS_ISA constexpr ISA =
      utils::gemm_core_mask<TYPE, JBLAS_GEMM_CORE::ISA_MASK, JBLAS_GEMM_CORE::ISA_SHIFT, JBLAS_ISA>();
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
      write_back(_mtile, NRegs);
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
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp1);
      add(reg_matAptr, AKStepSize * KUNROLL);
      add(reg_tmp1, BKStepSize * KUNROLL);
      add(reg_iterk, KTILE * KUNROLL);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_mtile, _nregs, 1, reg_tmp1);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _kunroll, const Xbyak::Reg64& _reg_matBptr) {
      for (int kk = 0; kk < _kunroll; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastd(Xbyak::Ymm(AReg), ptr[reg_tmp]);
          add(reg_tmp, reg_astep);
          for (int i = 0; i < _NRegs; i++) {
            vpdpbusds_vex(Xbyak::Ymm(CReg + mm * NRegs + i), Xbyak::Ymm(AReg),
                          ptr[_reg_matBptr + kk * BKStepSize + i * VecBytes]);
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs) {
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
               int kpos, void* tmpcache, size_t cachesize) {
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
  static JBLAS_GEMM_CORE constexpr TYPE = JBLAS_GEMM_CORE::AVX_VNNI_2x48;
  static JBLAS_ISA constexpr ISA =
      utils::gemm_core_mask<TYPE, JBLAS_GEMM_CORE::ISA_MASK, JBLAS_GEMM_CORE::ISA_SHIFT, JBLAS_ISA>();
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
      write_back(_mtile, NRegs);
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
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp1);
      add(reg_matAptr, AKStepSize * KUNROLL);
      add(reg_tmp1, BKStepSize * KUNROLL);
      add(reg_iterk, KTILE * KUNROLL);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_mtile, _nregs, 1, reg_tmp1);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _kunroll, const Xbyak::Reg64& _reg_matBptr) {
      for (int kk = 0; kk < _kunroll; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastd(Xbyak::Ymm(AReg), ptr[reg_tmp]);
          add(reg_tmp, reg_astep);
          for (int i = 0; i < _NRegs; i++) {
            vpdpbusds_vex(Xbyak::Ymm(CReg + mm * NRegs + i), Xbyak::Ymm(AReg),
                          ptr[_reg_matBptr + kk * BKStepSize + i * VecBytes]);
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs) {
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
               int kpos, void* tmpcache, size_t cachesize) {
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
  static JBLAS_GEMM_CORE constexpr TYPE = JBLAS_GEMM_CORE::AVX512F_8x48;
  static JBLAS_ISA constexpr ISA =
      utils::gemm_core_mask<TYPE, JBLAS_GEMM_CORE::ISA_MASK, JBLAS_GEMM_CORE::ISA_SHIFT, JBLAS_ISA>();
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
      write_back(_mtile, NRegs);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n32");
      cmp(reg_tmp, 32);
      jl(".n16", T_NEAR);
      generate_kloop(_mtile, 2);
      write_back(_mtile, 2);
      add(reg_itern, 32);
      add(reg_matBptr, 32 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n16");
      xor_(reg_iterk, reg_iterk);
      generate_kloop(_mtile, 1);
      write_back(_mtile, 1);
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
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp1);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_mtile, _nregs, 1, reg_tmp1);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _ktile, const Xbyak::Reg64& _reg_matBptr) {
      for (int kk = 0; kk < _ktile; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Zmm(BReg + i), ptr[_reg_matBptr + kk * BKStepSize + i * VecBytes]);
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

    void write_back(int _mtile, int _NRegs) {
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
               int kpos, void* tmpcache, size_t cachesize) {
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

  static JBLAS_GEMM_CORE constexpr TYPE = JBLAS_GEMM_CORE::AVX512_FP16_8x64;
  static JBLAS_ISA constexpr ISA =
      utils::gemm_core_mask<TYPE, JBLAS_GEMM_CORE::ISA_MASK, JBLAS_GEMM_CORE::ISA_SHIFT, JBLAS_ISA>();
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
      write_back(_mtile, NRegs);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n32");
      generate_kloop(_mtile, 1);
      write_back(_mtile, 1);
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
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp1);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_mtile, _nregs, 1, reg_tmp1);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _ktile, const Xbyak::Reg64& _reg_matBptr) {
      for (int kk = 0; kk < _ktile; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Zmm(BReg + i), ptr[_reg_matBptr + kk * BKStepSize + i * VecBytes]);
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

    void write_back(int _mtile, int _NRegs) {
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
               int _bstride, int _cstride, int kpos, void* tmpcache, size_t cachesize) {
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
  static JBLAS_GEMM_CORE constexpr TYPE = JBLAS_GEMM_CORE::AVX512_FP16_8x96;
  static JBLAS_ISA constexpr ISA =
      utils::gemm_core_mask<TYPE, JBLAS_GEMM_CORE::ISA_MASK, JBLAS_GEMM_CORE::ISA_SHIFT, JBLAS_ISA>();
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
      write_back(_mtile, NRegs);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n64");
      cmp(reg_tmp, 64);
      jl(".n32", T_NEAR);
      generate_kloop(_mtile, 2);
      write_back(_mtile, 2);
      add(reg_itern, 64);
      add(reg_matBptr, 64 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n32");
      generate_kloop(_mtile, 1);
      write_back(_mtile, 1);
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
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp1);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_mtile, _nregs, 1, reg_tmp1);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _ktile, const Xbyak::Reg64& _reg_matBptr) {
      for (int kk = 0; kk < _ktile; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Zmm(BReg + i), ptr[_reg_matBptr + kk * BKStepSize + i * VecBytes]);
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

    void write_back(int _mtile, int _NRegs) {
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
               int _bstride, int _cstride, int kpos, void* tmpcache, size_t cachesize) {
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
  static JBLAS_GEMM_CORE constexpr TYPE = JBLAS_GEMM_CORE::AVX512_VNNI_8x48;
  static JBLAS_ISA constexpr ISA =
      utils::gemm_core_mask<TYPE, JBLAS_GEMM_CORE::ISA_MASK, JBLAS_GEMM_CORE::ISA_SHIFT, JBLAS_ISA>();
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
      write_back(_mtile, NRegs);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n32");
      cmp(reg_tmp, 32);
      jl(".n16", T_NEAR);
      generate_kloop(_mtile, 2);
      write_back(_mtile, 2);
      add(reg_itern, 32);
      add(reg_matBptr, 32 * sizeof(int8_t) * 4);
      jmp(".nend", T_NEAR);

      L(".n16");
      xor_(reg_iterk, reg_iterk);
      generate_kloop(_mtile, 1);
      write_back(_mtile, 1);
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
      generate_fma(_mtile, _nregs, KUNROLL, reg_tmp1);
      add(reg_matAptr, AKStepSize * KUNROLL);
      add(reg_tmp1, BKStepSize * KUNROLL);
      add(reg_iterk, KTILE * KUNROLL);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_mtile, _nregs, 1, reg_tmp1);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _mtile, int _NRegs, int _kunroll, const Xbyak::Reg64& _reg_matBptr) {
      for (int kk = 0; kk < _kunroll; kk++) {
        lea(reg_tmp, ptr[reg_matAptr + kk * AKStepSize]);
        for (int i = 0; i < _NRegs; i++) {
          vmovups(Xbyak::Zmm(BReg + i), ptr[_reg_matBptr + kk * BKStepSize + i * VecBytes]);
        }
        for (int mm = 0; mm < _mtile; mm++) {
          vpbroadcastd(Xbyak::Zmm(AReg), ptr[reg_tmp]);
          add(reg_tmp, reg_astep);
          for (int i = 0; i < _NRegs; i++) {
            vpdpbusds_evex(Xbyak::Zmm(CReg + mm * NRegs + i), Xbyak::Zmm(AReg), Xbyak::Zmm(BReg + i));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs) {
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
               int kpos, void* tmpcache, size_t cachesize) {
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

  static JBLAS_GEMM_CORE constexpr TYPE = JBLAS_GEMM_CORE::AMX_BF16_16x64;
  static JBLAS_ISA constexpr ISA =
      utils::gemm_core_mask<TYPE, JBLAS_GEMM_CORE::ISA_MASK, JBLAS_GEMM_CORE::ISA_SHIFT, JBLAS_ISA>();
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
      write_back(MTILE, NRegs);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n48");
      cmp(reg_tmp, 48);
      jl(".n32", T_NEAR);
      generate_kloop(3);
      write_back(MTILE, 3);
      add(reg_itern, 48);
      add(reg_matBptr, 48 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n32");
      cmp(reg_tmp, 32);
      jl(".n16", T_NEAR);
      generate_kloop(2);
      write_back(MTILE, 2);
      add(reg_itern, 32);
      add(reg_matBptr, 32 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n16");
      xor_(reg_iterk, reg_iterk);
      generate_kloop(1);
      write_back(MTILE, 1);
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
      generate_fma(_nregs, KUNROLL, reg_tmp1);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_nregs, 1, reg_tmp1);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _NTile, int _kunroll, const Xbyak::Reg64& _reg_matBptr) {
      mov(reg_tmp, NTILE * 4);
      if (_NTile <= B_tilenum) {
        for (int kk = 0; kk < _kunroll; kk++) {
          for (int i = 0; i < _NTile; i++) {
            tileloaddt1(Xbyak::Tmm(BTile + i), ptr[_reg_matBptr + reg_tmp + kk * BKStepSize + i * 64]);
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
            tileloaddt1(Xbyak::Tmm(BTile + i), ptr[_reg_matBptr + reg_tmp + kk * BKStepSize + i * 64]);
          }

          for (int mm = 0; mm < 1; mm++) {
            tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_matAptr + reg_astep + kk * AKStepSize]);
            for (int i = 0; i < _NTile - 1; i++) {
              tdpbf16ps(Xbyak::Tmm(CTile + mm * C_tilenum + i), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile + i));
            }
            tileloaddt1(Xbyak::Tmm(BTile + 0), ptr[_reg_matBptr + reg_tmp + kk * BKStepSize + (_NTile - 1) * 64]);
            tdpbf16ps(Xbyak::Tmm(CTile + mm * C_tilenum + _NTile - 1), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile + 0));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs) {
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
               int kpos, void* tmpcache, size_t cachesize) {
    assert((NTILE * MTILE * sizeof(CType))<= cachesize);
    MicroKernel::tileconfig_t mCfg;
    memset(&mCfg, 0, sizeof(mCfg));
    auto param = params{matA, matB, matC, _k, _m, _n, _astride, _bstride, _cstride, kpos, tmpcache, &mCfg};
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

  static JBLAS_GEMM_CORE constexpr TYPE = JBLAS_GEMM_CORE::AMX_BF16_16x48;
  static JBLAS_ISA constexpr ISA =
      utils::gemm_core_mask<TYPE, JBLAS_GEMM_CORE::ISA_MASK, JBLAS_GEMM_CORE::ISA_SHIFT, JBLAS_ISA>();
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
      write_back(MTILE, NRegs);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n32");
      cmp(reg_tmp, 32);
      jl(".n16", T_NEAR);
      generate_kloop(2);
      write_back(MTILE, 2);
      add(reg_itern, 32);
      add(reg_matBptr, 32 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n16");
      xor_(reg_iterk, reg_iterk);
      generate_kloop(1);
      write_back(MTILE, 1);
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
      generate_fma(_nregs, KUNROLL, reg_tmp1);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_nregs, 1, reg_tmp1);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _NTile, int _kunroll, const Xbyak::Reg64& _reg_matBptr) {
      mov(reg_tmp, NTILE * 4);
      for (int kk = 0; kk < _kunroll; kk++) {
        for (int i = 0; i < _NTile; i++) {
          tileloaddt1(Xbyak::Tmm(BTile + i), ptr[_reg_matBptr + reg_tmp + kk * BKStepSize + i * 64]);
        }

        for (int mm = 0; mm < 1; mm++) {
          tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_matAptr + reg_astep + kk * AKStepSize]);
          for (int i = 0; i < _NTile; i++) {
            tdpbf16ps(Xbyak::Tmm(CTile + mm * C_tilenum + i), Xbyak::Tmm(ATile + mm), Xbyak::Tmm(BTile + i));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs) {
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
               int kpos, void* tmpcache, size_t cachesize) {
    assert((NTILE * MTILE * sizeof(CType)) <= cachesize);
    MicroKernel::tileconfig_t mCfg;
    std::memset(&mCfg, 0, sizeof(mCfg));
    auto param = params{matA, matB, matC, _k, _m, _n, _astride, _bstride, _cstride, kpos, tmpcache, &mCfg};
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

template <class T_A_, class T_B_, JBLAS_GEMM_CORE _ID>
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

  static JBLAS_GEMM_CORE constexpr TYPE = _ID;
  static JBLAS_ISA constexpr ISA =
      utils::gemm_core_mask<TYPE, JBLAS_GEMM_CORE::ISA_MASK, JBLAS_GEMM_CORE::ISA_SHIFT, JBLAS_ISA>();
  static int constexpr NTILE = 64, MTILE = 16, KTILE = 64 / sizeof(BType);
  static int constexpr PACK_ROW = 4;
  static int constexpr KUNROLL = 2;
  static int constexpr PREFERED_N = 256;
  class MicroKernel : protected jblas::xbyak::JitAmxint8 {
   public:
    friend GemmCore_Row_NN_16x64_AMX_I8<T_A_, T_B_, _ID>;
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
      write_back(MTILE, NRegs);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n48");
      cmp(reg_tmp, 48);
      jl(".n32", T_NEAR);
      generate_kloop(3);
      write_back(MTILE, 3);
      add(reg_itern, 48);
      add(reg_matBptr, 48 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n32");
      cmp(reg_tmp, 32);
      jl(".n16", T_NEAR);
      generate_kloop(2);
      write_back(MTILE, 2);
      add(reg_itern, 32);
      add(reg_matBptr, 32 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n16");
      xor_(reg_iterk, reg_iterk);
      generate_kloop(1);
      write_back(MTILE, 1);
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
      generate_fma(_nregs, KUNROLL, reg_tmp1);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_nregs, 1, reg_tmp1);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _NTile, int _kunroll, const Xbyak::Reg64& _reg_matBptr) {
      mov(reg_tmp, NTILE * 4);
      if (_NTile <= B_tilenum) {
        for (int kk = 0; kk < _kunroll; kk++) {
          for (int i = 0; i < _NTile; i++) {
            tileloaddt1(Xbyak::Tmm(BTile + i), ptr[_reg_matBptr + reg_tmp + kk * BKStepSize + i * 64]);
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
            tileloaddt1(Xbyak::Tmm(BTile + i), ptr[_reg_matBptr + reg_tmp + kk * BKStepSize + i * 64]);
          }

          for (int mm = 0; mm < 1; mm++) {
            tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_matAptr + reg_astep + kk * AKStepSize]);
            for (int i = 0; i < _NTile - 1; i++) {
              _tdpb<AType, BType>(Xbyak::Tmm(CTile + mm * C_tilenum + i), Xbyak::Tmm(ATile + mm),
                                  Xbyak::Tmm(BTile + i));
            }
            tileloaddt1(Xbyak::Tmm(BTile + 0), ptr[_reg_matBptr + reg_tmp + kk * BKStepSize + (_NTile - 1) * 64]);
            _tdpb<AType, BType>(Xbyak::Tmm(CTile + mm * C_tilenum + _NTile - 1), Xbyak::Tmm(ATile + mm),
                                Xbyak::Tmm(BTile + 0));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs) {
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
               int kpos, void* tmpcache, size_t cachesize) {
    assert((NTILE * MTILE * sizeof(CType)) <= cachesize);
    typename MicroKernel::tileconfig_t mCfg;
    memset(&mCfg, 0, sizeof(mCfg));
    auto param = params{matA, matB, matC, _k, _m, _n, _astride, _bstride, _cstride, kpos, tmpcache, &mCfg};
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
using GemmCore_Row_NN_16x64_AMX_U8S8 =
    GemmCore_Row_NN_16x64_AMX_I8<uint8_t, int8_t, JBLAS_GEMM_CORE::AMX_INT8_16x64_US>;
using GemmCore_Row_NN_16x64_AMX_S8S8 =
    GemmCore_Row_NN_16x64_AMX_I8<uint8_t, int8_t, JBLAS_GEMM_CORE::AMX_INT8_16x64_SS>;

template <class T_A_, class T_B_, JBLAS_GEMM_CORE _ID>
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

  static JBLAS_GEMM_CORE constexpr TYPE = _ID;
  static JBLAS_ISA constexpr ISA =
      utils::gemm_core_mask<TYPE, JBLAS_GEMM_CORE::ISA_MASK, JBLAS_GEMM_CORE::ISA_SHIFT, JBLAS_ISA>();
  static int constexpr NTILE = 48, MTILE = 16, KTILE = 64 / sizeof(BType);
  static int constexpr PACK_ROW = 4 / sizeof(BType);
  static int constexpr KUNROLL = 2;
  static int constexpr PREFERED_N = 240;  // TODO?
  class MicroKernel : protected jblas::xbyak::JitAmxint8 {
   public:
    friend GemmCore_Row_NN_16x48_AMX_I8<T_A_, T_B_, _ID>;
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
      write_back(MTILE, C_tilenum);
      load32(reg_tmp, ptr[parambase + OFFSET(bstep)]);
      imul(reg_tmp, reg_tmp, NTILE);
      add(reg_matBptr, reg_tmp);
      add(reg_itern, NTILE);
      jmp(".nend", T_NEAR);

      L(".n32");
      cmp(reg_tmp, 32);
      jl(".n16", T_NEAR);
      generate_kloop(2);
      write_back(MTILE, 2);
      add(reg_itern, 32);
      add(reg_matBptr, 32 * sizeof(BType));
      jmp(".nend", T_NEAR);

      L(".n16");
      xor_(reg_iterk, reg_iterk);
      generate_kloop(1);
      write_back(MTILE, 1);
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
      generate_fma(_nregs, KUNROLL, reg_tmp1);
      add(reg_matAptr, KUNROLL * AKStepSize);
      add(reg_tmp1, KUNROLL * BKStepSize);
      add(reg_iterk, KUNROLL * KTILE);
      jmp(".kloopend", T_NEAR);

      L(".k1loop");
      generate_fma(_nregs, 1, reg_tmp1);
      add(reg_matAptr, 1 * AKStepSize);
      add(reg_tmp1, 1 * BKStepSize);
      add(reg_iterk, 1 * KTILE);
      L(".kloopend");
      cmp(reg_iterk, reg_ksize);  // k iteration variable
      jb(".kloop");
      outLocalLabel();
    }

    void generate_fma(int _NTile, int _kunroll, const Xbyak::Reg64& _reg_matBptr) {
      mov(reg_tmp, NTILE * 4);
      if (_NTile <= B_tilenum) {
        for (int kk = 0; kk < _kunroll; kk++) {
          for (int i = 0; i < _NTile; i++) {
            tileloaddt1(Xbyak::Tmm(BTile + i), ptr[_reg_matBptr + reg_tmp + kk * BKStepSize + i * 64]);
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
            tileloaddt1(Xbyak::Tmm(BTile + i), ptr[_reg_matBptr + reg_tmp + kk * BKStepSize + i * 64]);
          }

          for (int mm = 0; mm < 1; mm++) {
            tileloadd(Xbyak::Tmm(ATile + mm), ptr[reg_matAptr + reg_astep + kk * AKStepSize]);
            for (int i = 0; i < _NTile - 1; i++) {
              _tdpb<AType, BType>(Xbyak::Tmm(CTile + mm * C_tilenum + i), Xbyak::Tmm(ATile + mm),
                                  Xbyak::Tmm(BTile + i));
            }
            tileloaddt1(Xbyak::Tmm(BTile + 0), ptr[_reg_matBptr + reg_tmp + kk * BKStepSize + (_NTile - 1) * 64]);
            _tdpb<AType, BType>(Xbyak::Tmm(CTile + mm * C_tilenum + _NTile - 1), Xbyak::Tmm(ATile + mm),
                                Xbyak::Tmm(BTile + 0));
          }
        }
      }
    }

    void write_back(int _mtile, int _NRegs) {
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
               int kpos, void* tmpcache, size_t cachesize) {
    assert((NTILE * MTILE * sizeof(CType)) <= cachesize);
    typename MicroKernel::tileconfig_t mCfg;
    memset(&mCfg, 0, sizeof(mCfg));
    auto param = params{matA, matB, matC, _k, _m, _n, _astride, _bstride, _cstride, kpos, tmpcache, &mCfg};
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
using GemmCore_Row_NN_16x48_AMX_U8S8 =
    GemmCore_Row_NN_16x48_AMX_I8<uint8_t, int8_t, JBLAS_GEMM_CORE::AMX_INT8_16x48_US>;
using GemmCore_Row_NN_16x48_AMX_S8S8 = GemmCore_Row_NN_16x48_AMX_I8<int8_t, int8_t, JBLAS_GEMM_CORE::AMX_INT8_16x48_SS>;
}  // namespace gemm
}  // namespace jblas
