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
#include <algorithm>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "jit_base.hpp"
#include "jit_blas_utils.h"
#include "kernel_jit_injector.h"

namespace jblas {
namespace kernel {
namespace jit {

class DequanS8F32 {
 public:
  class MicroKernelAVX512F : protected jblas::xbyak::JitAvx512f {
   public:
    struct params {
      void *srcptr, *dstptr;
      int row, col;
      int srcstride, dststride;
      float* scales;
      int8_t* zps;
    };
    typedef long long (*func_t)(params*);
    static int constexpr VBytes = 64;
    static int constexpr RegScale = 0;
    static int constexpr RegZP = 4;
    static int constexpr RegTmp = RegScale + 8;
    MicroKernelAVX512F(bool is_sym_) {
      is_sym = is_sym_;
      generate();
      this->ready();
      mKernel = this->getCode<func_t>();
    }

    void generate() {
      inLocalLabel();  // use local label for multiple instance
      int SF_TmpSize = 64;
      int SF_TmpPos = 16 * 14;
      Xbyak::util::StackFrame st(this, 1, 13, SF_TmpPos + SF_TmpSize);
      parambase = st.p[0];
      reg_srcptr = st.t[0];
      reg_dstptr = st.t[1];
      reg_srcstride = st.t[2];
      reg_dststride = st.t[3];
      reg_rowsize = st.t[4];
      reg_colsize = st.t[5];
      reg_iterrow = st.t[6];
      reg_itercol = st.t[7];
      reg_tmp = st.t[8];
      reg_scaleptr = st.t[9];
      reg_tmpdst = st.t[10];
      reg_tmp1 = st.t[12];
      reg_ret = rax;

      vreg_push(rsp);

      mov(reg_srcptr, ptr[parambase + OFFSET(srcptr)]);
      mov(reg_dstptr, ptr[parambase + OFFSET(dstptr)]);
      mov(reg_scaleptr, ptr[parambase + OFFSET(scales)]);
      xor_(reg_srcstride, reg_srcstride);
      mov(reg_srcstride.cvt32(), ptr[parambase + OFFSET(srcstride)]);
      xor_(reg_dststride, reg_dststride);
      mov(reg_dststride.cvt32(), ptr[parambase + OFFSET(dststride)]);

      load32(reg_colsize, ptr[parambase + OFFSET(col)]);
      load32(reg_rowsize, ptr[parambase + OFFSET(row)]);
      xor_(reg_itercol, reg_itercol);

      // reuse parambase reg
      if (!is_sym) {
        mov(reg_tmp1, ptr[parambase + OFFSET(zps)]);
        mov(reg_zpptr, reg_tmp1);
        xor_(reg_tmp1, reg_tmp1);
      }

      L(".colloop");
      mov(reg_tmp, reg_colsize);
      sub(reg_tmp, reg_itercol);
      cmp(reg_tmp, 64);
      jl(".proc48", T_NEAR);
      generateNTile(4);
      add(reg_itercol, 64);
      add(reg_srcptr, 1 * 64);
      add(reg_dstptr, 4 * 64);
      add(reg_scaleptr, 4 * 64);
      if (!is_sym) add(reg_zpptr, 1 * 64);
      jmp(".colend", T_NEAR);

      L(".proc48");
      cmp(reg_tmp, 48);
      jl(".proc32", T_NEAR);
      generateNTile(3);
      add(reg_itercol, 48);
      add(reg_srcptr, 1 * 48);
      add(reg_dstptr, 4 * 48);
      add(reg_scaleptr, 4 * 48);
      if (!is_sym) add(reg_zpptr, 1 * 48);
      jmp(".colend", T_NEAR);

      L(".proc32");
      generateNTile(2);
      add(reg_itercol, 32);
      add(reg_srcptr, 1 * 32);
      add(reg_dstptr, 4 * 32);
      add(reg_scaleptr, 4 * 32);
      if (!is_sym) add(reg_zpptr, 1 * 32);

      L(".colend");
      cmp(reg_itercol, reg_colsize);
      jb(".colloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);
      outLocalLabel();  // end of local label
    }

    void generateNTile(int N) {
      for (int i = 0; i < N; i++) {
        vmovups(Xbyak::Zmm(RegScale + i), ptr[reg_scaleptr + i * 64]);
        if (!is_sym) {
          vpmovsxbd(Xbyak::Zmm(RegZP + i), ptr[reg_zpptr + i * 16]);
        }
      }
      inLocalLabel();
      xor_(reg_iterrow, reg_iterrow);
      mov(reg_tmp, reg_srcptr);
      mov(reg_tmp1, reg_dstptr);
      L(".rowloop");
      for (int i = 0; i < N; i++) {
        vpmovsxbd(Xbyak::Zmm(RegTmp), ptr[reg_tmp + i * 16]);
        if (!is_sym) {
          vpsubd(Xbyak::Zmm(RegTmp), Xbyak::Zmm(RegTmp), Xbyak::Zmm(RegZP + i));
        }
        vcvtdq2ps(Xbyak::Zmm(RegTmp), Xbyak::Zmm(RegTmp));
        vmulps(Xbyak::Zmm(RegTmp), Xbyak::Zmm(RegScale + i));
        vmovups(ptr[reg_tmp1 + i * 64], Xbyak::Zmm(RegTmp));
      }
      add(reg_tmp, reg_srcstride);
      add(reg_tmp1, reg_dststride);
      add(reg_iterrow, 1);
      cmp(reg_iterrow, reg_rowsize);
      jb(".rowloop");
      outLocalLabel();
    }
    func_t mKernel = nullptr;

   private:
    Xbyak::Reg64 parambase;
    Xbyak::Reg64 reg_srcptr;
    Xbyak::Reg64 reg_dstptr;
    Xbyak::Reg64 reg_srcstride;
    Xbyak::Reg64 reg_dststride;
    Xbyak::Reg64 reg_rowsize;
    Xbyak::Reg64 reg_colsize;
    Xbyak::Reg64 reg_iterrow;
    Xbyak::Reg64 reg_itercol;
    Xbyak::Reg64 reg_tmp;
    Xbyak::Reg64 reg_scaleptr;
    Xbyak::Reg64 reg_tmpdst;
    Xbyak::Reg64 reg_tmp1;
    Xbyak::Reg64 reg_ret;
    Xbyak::Reg64 reg_zpptr = reg_ret;
    bool is_sym;
  };
  static void forward_avx512f(int8_t* srcptr, float* dstptr, int row, int col, int ld_src, int ld_dst, float* scales,
                              int8_t* zero_points) {
    static MicroKernelAVX512F mAVX512FSym(true);
    static MicroKernelAVX512F mAVX512FASym(false);
    auto param = MicroKernelAVX512F::params{
        srcptr, dstptr, row, col, int(ld_src * sizeof(int8_t)), int(ld_dst * sizeof(float)), scales, zero_points};
    if (zero_points == nullptr) {
      mAVX512FSym.mKernel(&param);
    } else {
      mAVX512FASym.mKernel(&param);
    }
  }
};

class DequanKBlockS8F32 {
 public:
  template <typename _ST>
  static inline JBLAS_CODE forward_avx512f(int8_t* srcptr, float* dstptr, int row, int col, int ld_src, int ld_dst,
                                           _ST* scales, int8_t* zero_points, int k_offset, int kblock, int NPad) {
    int row0 = kblock - k_offset % kblock;
    row0 = row0 == kblock ? 0 : row0;
    row0 = row0 > row ? row : row0;
    int row1 = row - row0;
    int row1_blk = utils::padto_le(row1, kblock);
    int row2 = row - row1_blk - row0;
    auto sptr = scales + k_offset / kblock * NPad;
    int8_t* zptr = nullptr;
    if (zero_points != nullptr) zptr = zero_points + k_offset / kblock * NPad;
    if (row0 > 0) {
      DequanS8F32::forward_avx512f(srcptr, dstptr, row0, col, ld_src, ld_dst, sptr, zptr);
      srcptr += row0 * ld_src;
      dstptr += row0 * ld_dst;
      sptr += NPad;
      if (zero_points != nullptr) zptr += NPad;
    }
    for (int i = 0; i < row1_blk; i += kblock) {
      DequanS8F32::forward_avx512f(srcptr, dstptr, kblock, col, ld_src, ld_dst, sptr, zptr);
      srcptr += kblock * ld_src;
      dstptr += kblock * ld_dst;
      sptr += NPad;
      if (zero_points != nullptr) zptr += NPad;
    }
    if (row2 > 0) {
      DequanS8F32::forward_avx512f(srcptr, dstptr, row2, col, ld_src, ld_dst, sptr, zptr);
    }
    return JblasNotSupport;
  }
};

class JitMemcpy2DAvx512f : protected jblas::xbyak::JitAvx512f {
 public:
  struct params {
    void *srcptr, *dstptr, *elt_const_v;
    int row, col;
    int srcstride, dststride;
  };
  typedef long long (*func_t)(params*);

 public:
  static int constexpr VBytes = 64;
  JitMemcpy2DAvx512f(int unroll_row, std::vector<kernel::jit_injector::eltwise_injector>& injectors) {
    generate(unroll_row, injectors);
  }

  template <typename _SRC_T, typename _DST_T, typename... Eltops>
  static JBLAS_CODE forward(const _SRC_T* srcptr, _DST_T* dstptr, int row, int col, int srcstep, int dststep,
                            void* elt_const_v = nullptr, Eltops... ops) {
    static std::vector<kernel::jit_injector::eltwise_injector> p = {static_cast<JBLAS_ELTWISEOP>(ops)...};
    if constexpr (sizeof...(ops) != 0)
      static_assert(std::is_same<_SRC_T, float>::value && std::is_same<_DST_T, float>::value);
    static JitMemcpy2DAvx512f instance_withops(1, p);
    static JitMemcpy2DAvx512f instance4_withops(4, p);
    static_assert(sizeof(_SRC_T) == sizeof(_DST_T));  // TODO SRC_T DST_T conversion copy
    auto param = params{(void*)srcptr,
                        (void*)dstptr,
                        elt_const_v,
                        row,
                        int(col * sizeof(_SRC_T)),
                        int(srcstep * sizeof(_SRC_T)),
                        int(dststep * sizeof(_DST_T))};
    int row4 = utils::padto_le(row, 4);
    if (row4) {
      param.row = row4;
      instance4_withops.mKernel(&param);
    }
    int rowtail = row - row4;
    if (rowtail) {
      param.srcptr = (char*)param.srcptr + row4 * srcstep * sizeof(_SRC_T);
      param.dstptr = (char*)param.dstptr + row4 * dststep * sizeof(_DST_T);
      param.row = rowtail;
      instance_withops.mKernel(&param);
    }
    return JblasSuccess;
  }

 protected:
  void generate(int unrollk, std::vector<kernel::jit_injector::eltwise_injector>& injectors) {  // unrollK=[1,2,4]
    if (unrollk != 1 && unrollk != 2 && unrollk != 4) {
      assert(false);
      return;
    }
    inLocalLabel();  // use local label for multiple instance
    {
      int SF_TmpSize = 64;
      Xbyak::util::StackFrame st(this, 1, 13, 16 * 10 + SF_TmpSize);
      const Xbyak::Reg64& parambase = st.p[0];
      const Xbyak::Reg64& reg_srcptr = st.t[0];
      const Xbyak::Reg64& reg_dstptr = st.t[1];
      const Xbyak::Reg64& reg_srcstride = st.t[2];
      const Xbyak::Reg64& reg_dststride = st.t[3];
      const Xbyak::Reg64& reg_rowsize = st.t[4];
      const Xbyak::Reg64& reg_colsize = st.t[5];
      const Xbyak::Reg64& reg_iterrow = st.t[6];
      const Xbyak::Reg64& reg_itercol = st.t[7];
      const Xbyak::Reg64& reg_tmp = st.t[8];
      const Xbyak::Reg64& reg_elt_constv = st.t[8];  // alias of reg_tmp.
      const Xbyak::Reg64& reg_tmpsrc = st.t[9];
      const Xbyak::Reg64& reg_tmpdst = st.t[10];
      const Xbyak::Reg64& reg_tmp1 = st.t[12];
      const Xbyak::Reg64& reg_tmp2 = st.t[11];
      const Xbyak::Reg64& reg_ret = rax;

      vreg_push(rsp);

      mov(reg_srcptr, ptr[parambase + OFFSET(srcptr)]);
      mov(reg_dstptr, ptr[parambase + OFFSET(dstptr)]);
      xor_(reg_srcstride, reg_srcstride);
      mov(reg_srcstride.cvt32(), ptr[parambase + OFFSET(srcstride)]);
      xor_(reg_dststride, reg_dststride);
      mov(reg_dststride.cvt32(), ptr[parambase + OFFSET(dststride)]);

      load32(reg_colsize, ptr[parambase + OFFSET(col)]);
      load32(reg_rowsize, ptr[parambase + OFFSET(row)]);
      if (unrollk == 4) {
        imul(reg_tmp1, reg_srcstride, 3);
        imul(reg_tmp2, reg_dststride, 3);
      }
      int const ColUnroll = 4;

      for (int i = 0; i < unrollk * ColUnroll; i++) used_zmm_idx.insert(i);
      for (auto&& injector : injectors) {
        injector.assign_resources(this, used_zmm_idx, reg_ret, k2);
        injector.assign_reg_elt_constp(reg_elt_constv);
      }

      xor_(reg_iterrow, reg_iterrow);
      L(".rowloop");
      xor_(reg_itercol, reg_itercol);
      mov(reg_tmpsrc, reg_srcptr);
      mov(reg_tmpdst, reg_dstptr);

      L(".colloop");
      mov(reg_tmp, reg_colsize);
      sub(reg_tmp, reg_itercol);
      cmp(reg_tmp, ColUnroll * VBytes);
      jl(".maskproc", T_NEAR);
      push(reg_tmp);
      mov(reg_elt_constv, ptr[parambase + OFFSET(elt_const_v)]);
      if (unrollk > 1) {
        for (int j = 0; j < unrollk; j++) {
          for (int i = 0; i < ColUnroll; i++) {
            if (j == 3) {
              vmovups(Xbyak::Zmm(i + j * ColUnroll), ptr[reg_tmpsrc + reg_tmp1 + i * VBytes]);
              for (int k = 0; k < injectors.size(); k++)
                injectors[k].vector_compute(Xbyak::Zmm(i + j * ColUnroll), k * 3 * sizeof(float));
              vmovups(ptr[reg_tmpdst + reg_tmp2 + i * VBytes], Xbyak::Zmm(i + j * ColUnroll));
            } else {
              vmovups(Xbyak::Zmm(i + j * ColUnroll), ptr[reg_tmpsrc + reg_srcstride * j + i * VBytes]);
              for (int k = 0; k < injectors.size(); k++)
                injectors[k].vector_compute(Xbyak::Zmm(i + j * ColUnroll), k * 3 * sizeof(float));
              vmovups(ptr[reg_tmpdst + reg_dststride * j + i * VBytes], Xbyak::Zmm(i + j * ColUnroll));
            }
          }
        }
      } else {
        for (int i = 0; i < ColUnroll; i++) {
          vmovups(Xbyak::Zmm(i), ptr[reg_tmpsrc + i * VBytes]);
          for (int k = 0; k < injectors.size(); k++) injectors[k].vector_compute(Xbyak::Zmm(i), k * 3 * sizeof(float));
          vmovups(ptr[reg_tmpdst + i * VBytes], Xbyak::Zmm(i));
        }
      }
      pop(reg_tmp);
      add(reg_tmpsrc, ColUnroll * VBytes);
      add(reg_tmpdst, ColUnroll * VBytes);
      add(reg_itercol, ColUnroll * VBytes);
      jmp(".colend", T_NEAR);
      L(".maskproc");
      push(reg_tmp1);
      generate_Nbitsmask(k1, reg_itercol, reg_colsize, reg_tmp, reg_tmp1, VBytes);
      pop(reg_tmp1);
      push(reg_tmp);
      mov(reg_elt_constv, ptr[parambase + OFFSET(elt_const_v)]);
      if (unrollk > 1) {
        for (int j = 0; j < unrollk; j++) {
          if (j == 3) {
            vmovdqu8(Xbyak::Zmm(0) | k1, ptr[reg_tmpsrc + reg_tmp1]);
            for (int k = 0; k < injectors.size(); k++)
              injectors[k].vector_compute(Xbyak::Zmm(0), k * 3 * sizeof(float));
            vmovdqu8(ptr[reg_tmpdst + reg_tmp2], Xbyak::Zmm(0) | k1);
          } else {
            vmovdqu8(Xbyak::Zmm(0) | k1, ptr[reg_tmpsrc + reg_srcstride * j]);
            for (int k = 0; k < injectors.size(); k++)
              injectors[k].vector_compute(Xbyak::Zmm(0), k * 3 * sizeof(float));
            vmovdqu8(ptr[reg_tmpdst + reg_dststride * j], Xbyak::Zmm(0) | k1);
          }
        }
      } else {
        vmovdqu8(Xbyak::Zmm(0) | k1, ptr[reg_tmpsrc]);
        for (int k = 0; k < injectors.size(); k++) injectors[k].vector_compute(Xbyak::Zmm(0), k * 3 * sizeof(float));
        vmovdqu8(ptr[reg_tmpdst], Xbyak::Zmm(0) | k1);
      }
      pop(reg_tmp);
      add(reg_tmpsrc, VBytes);
      add(reg_tmpdst, VBytes);
      add(reg_itercol, VBytes);
      L(".colend");
      cmp(reg_itercol, reg_colsize);
      jb(".colloop");
      add(reg_iterrow, unrollk);
      lea(reg_srcptr, ptr[reg_srcptr + reg_srcstride * unrollk]);
      lea(reg_dstptr, ptr[reg_dstptr + reg_dststride * unrollk]);
      cmp(reg_iterrow, reg_rowsize);
      jb(".rowloop");

      mov(reg_ret, 0);
      vreg_pop(rsp);
    }
    outLocalLabel();  // end of local label
    for (auto&& injector : injectors) injector.prepare_table();
    this->ready();
    mKernel = this->getCode<func_t>();
  }

  func_t mKernel = nullptr;
  std::set<int> used_zmm_idx;
};

class CustomMemCpy : public JitMemcpy2DAvx512f {
 public:
  CustomMemCpy(int unroll_row, std::vector<kernel::jit_injector::eltwise_injector>& injectors)
      : JitMemcpy2DAvx512f(unroll_row, injectors) {}
  template <JBLAS_ELTWISEOP _OP>
  static JBLAS_CODE forward(const float* srcptr, float* dstptr, int row, int col, int srcstride, int dststride,
                            void* elt_const_v) {
    static std::vector<kernel::jit_injector::eltwise_injector> p = {_OP};
    static CustomMemCpy instance_withops(1, p);
    static CustomMemCpy instance4_withops(4, p);
    auto param = params{const_cast<float*>(srcptr), dstptr, elt_const_v, row, col, srcstride, dststride};
    int row4 = utils::padto_le(row, 4);
    if (row4) {
      param.row = row4;
      instance4_withops.mKernel(&param);
    }
    int rowtail = row - row4;
    if (rowtail) {
      param.srcptr = (char*)param.srcptr + row4 * srcstride;
      param.dstptr = (char*)param.dstptr + row4 * dststride;
      param.row = rowtail;
      instance_withops.mKernel(&param);
    }
    return JblasSuccess;
  }
};

static inline Xbyak::Zmm unpack_4bit(Xbyak::CodeGenerator* jit, Xbyak::Ymm v4bits, Xbyak::Zmm zmm, Xbyak::Zmm zmm1,
                                     Xbyak::Zmm vmask, Xbyak::Opmask unpack_mask) {
  Xbyak::Ymm ymm1(zmm1.getIdx());
  jit->vpmovsxbw(zmm, v4bits);
  jit->vpslld(ymm1, v4bits, 4);
  jit->vpmovsxbw(zmm1, ymm1);
  jit->vpsllw(zmm, zmm, 8);
  jit->vmovdqu8(zmm1 | unpack_mask, zmm);
  jit->vpandd(zmm1, vmask, zmm1);
  return zmm1;
}

static inline Xbyak::Zmm unpack_4bit_2regs(Xbyak::CodeGenerator* jit, Xbyak::Ymm v4bits, Xbyak::Zmm tmp,
                                           Xbyak::Zmm vmask, Xbyak::Opmask unpack_mask) {
  Xbyak::Zmm dst(v4bits.getIdx());
  jit->vpmovsxbw(tmp, v4bits);
  jit->vpslld(v4bits, v4bits, 4);
  jit->vpmovsxbw(dst, v4bits);
  jit->vpsllw(tmp, tmp, 8);
  jit->vmovdqu8(dst | unpack_mask, tmp);
  jit->vpandd(dst, vmask, dst);
  return dst;
}

class DecompressS4S8_AVX512F : protected jblas::xbyak::JitAvx512f {
 public:
  struct params {
    void *srcptr, *dstptr;
    size_t size;
  };
  typedef long long (*func_t)(params*);

 public:
  static int constexpr VBytes = 64;
  DecompressS4S8_AVX512F() {
    inLocalLabel();  // use local label for multiple instance
    int SF_TmpSize = 64;
    Xbyak::util::StackFrame st(this, 1, 13, 16 * 10 + SF_TmpSize);
    const Xbyak::Reg64& parambase = st.p[0];
    const Xbyak::Reg64& reg_srcptr = st.t[0];
    const Xbyak::Reg64& reg_dstptr = st.t[1];
    const Xbyak::Reg64& reg_size = st.t[5];
    const Xbyak::Reg64& reg_iterrow = st.t[6];
    const Xbyak::Reg64& reg_itercol = st.t[7];
    const Xbyak::Reg64& reg_tmp = st.t[8];
    const Xbyak::Reg64& reg_tmp1 = st.t[12];
    const Xbyak::Reg64& reg_ret = rax;

    vreg_push(rsp);

    mov(reg_srcptr, ptr[parambase + OFFSET(srcptr)]);
    mov(reg_dstptr, ptr[parambase + OFFSET(dstptr)]);
    mov(reg_size, ptr[parambase + OFFSET(size)]);
    Xbyak::Opmask unpack_mask(4);
    Xbyak::Zmm zmm_mask(31);
    mov(reg_tmp.cvt32(), uint32_t(0xf0f0f0f0));
    vpbroadcastd(zmm_mask, reg_tmp.cvt32());
    mov(reg_tmp, 0xaaaaaaaaaaaaaaaa);
    kmovq(unpack_mask, reg_tmp);
    int const ColUnroll = 4;
    xor_(reg_iterrow, reg_iterrow);
    xor_(reg_itercol, reg_itercol);
    L(".colloop");
    mov(reg_tmp, reg_size);
    sub(reg_tmp, reg_itercol);
    cmp(reg_tmp, ColUnroll * VBytes);
    jl(".maskproc", T_NEAR);
    mov(reg_tmp, reg_itercol);
    shr(reg_tmp, 1);
    for (int i = 0; i < ColUnroll; i++) {
      vmovups(Xbyak::Ymm(i), ptr[reg_srcptr + reg_tmp + i * VBytes / 2]);
      unpack_4bit_2regs(this, Xbyak::Ymm(i), Xbyak::Zmm(ColUnroll), zmm_mask, unpack_mask);
      vmovups(ptr[reg_dstptr + reg_itercol + i * VBytes], Xbyak::Zmm(i));
    }
    add(reg_itercol, ColUnroll * VBytes);
    jmp(".colend");
    L(".maskproc");
    generate_Nbitsmask(k1, reg_itercol, reg_size, reg_tmp, reg_tmp1, VBytes);
    mov(reg_tmp, reg_itercol);
    shr(reg_tmp, 1);
    vmovdqu8(Xbyak::Zmm(0) | k1, ptr[reg_srcptr + reg_tmp]);
    unpack_4bit_2regs(this, Xbyak::Ymm(0), Xbyak::Zmm(ColUnroll), zmm_mask, unpack_mask);
    vmovdqu8(ptr[reg_dstptr + reg_itercol], Xbyak::Zmm(0) | k1);
    add(reg_itercol, VBytes);
    L(".colend");
    cmp(reg_itercol, reg_size);
    jb(".colloop");

    mov(reg_ret, 0);
    vreg_pop(rsp);
    outLocalLabel();  // end of local label

    this->ready();
    mKernel = this->getCode<func_t>();
  }

  static JBLAS_CODE forward(void* srcptr, void* dstptr, size_t size) {
    static DecompressS4S8_AVX512F instance;
    auto param = params{srcptr, dstptr, size};
    instance.mKernel(&param);
    return JblasSuccess;
  }

 private:
  func_t mKernel = nullptr;
};

static inline JBLAS_CODE decompress_s4_s8(utils::int4x2* srcptr, int8_t* dstptr, int row, int col, int ld_src,
                                          int ld_dst) {
  if (col != ld_src) {  // memory is not continueous
    return JblasNotSupport;
  }
  DecompressS4S8_AVX512F::forward(srcptr, dstptr, (size_t)row * col);
  return JblasSuccess;
}

// src: row x col => dst: ⌈col/n_tile⌉ x ⌈row/row_pack⌉ x n_tile x row_pack (zeor-padded)
// Extra padding can be applied with memset calls in `static void forward(...)`
class PaddingInterleaveCvt : protected xbyak::JitAvx512f {
 public:
  struct params {
    const void* srcptr;
    void* dstptr;
    int row, col;
    int srcstride, dststride;  // dst = dst_base + dststride * n_idx, where n_idx % n_tile == 0
  };
  typedef void (*func_t)(params* p);
  void operator()(params* p) const { mKernel(p); }

 private:
  static inline const uint16_t idx_interleave_self[32] = {
      0,  16, 1,  17, 2,  18, 3,  19,  //
      4,  20, 5,  21, 6,  22, 7,  23,  //
      8,  24, 9,  25, 10, 26, 11, 27,  //
      12, 28, 13, 29, 14, 30, 15, 31,  //
  };

  PaddingInterleaveCvt(int n_tile, JBLAS_DTYPE dst_t) : PaddingInterleaveCvt(n_tile, dst_t, dst_t) {}
  PaddingInterleaveCvt(int n_tile, JBLAS_DTYPE dst_t, JBLAS_DTYPE src_t, int row_pack = 0) : xbyak::JitAvx512f() {
    inLocalLabel();  // use local label for multiple instance
    const auto src_bytes = static_cast<int>(utils::jblas_dtype_size(src_t));
    const auto dst_bytes = static_cast<int>(utils::jblas_dtype_size(dst_t));
    if (row_pack == 0) row_pack = 4 / dst_bytes;  // default value
    const auto ne_zmm = 64 / std::max(src_bytes, dst_bytes);
    const auto src_bytes_vmm = ne_zmm * src_bytes;

    assert(n_tile % ne_zmm == 0);
    assert(row_pack > 0 && row_pack < 3);  // TODO(yi): int8 interleave not implemented

    int SF_TmpSize = 64;
    Xbyak::Label l_idx_interleave_self;
    std::shared_ptr<void> epilogue{
        // generate code at the very end
        nullptr, [&](void*) {
          align(64);
          L(l_idx_interleave_self);
          db(reinterpret_cast<const uint8_t*>(idx_interleave_self), sizeof(idx_interleave_self));
          outLocalLabel();  // end of local label

          this->ready();
          this->mKernel = this->getCode<func_t>();
        }};
    Xbyak::util::StackFrame st(this, 1, 13, 16 * 10 + SF_TmpSize);
    const Xbyak::Reg64& parambase = st.p[0];
    const Xbyak::Reg64& reg_srcptr = st.t[0];
    const Xbyak::Reg64& reg_dstptr = st.t[1];
    const Xbyak::Reg64& reg_srcstride = st.t[2];
    const Xbyak::Reg64& reg_dststride = st.t[3];
    const Xbyak::Reg64& reg_colsize = st.t[5];
    const Xbyak::Reg64& reg_iterrow = st.t[6];
    const Xbyak::Reg64& reg_itercol = st.t[7];
    const Xbyak::Reg64& reg_tmp = st.t[8];
    const Xbyak::Reg64& reg_tmp1 = st.t[9];
    const Xbyak::Reg64& reg_tmp2 = st.t[12];
    const Xbyak::Reg64& reg_tmp3 = st.t[10];

    const Xbyak::Reg64& reg_ret = rax;
    auto& mask_rd = k1;
    const Xbyak::Zmm& vreg_idx0 = zmm31;

    vreg_push(rsp);
    vmovups(vreg_idx0, zword[rip + l_idx_interleave_self]);
    mov(reg_srcptr, ptr[parambase + OFFSET(srcptr)]);
    mov(reg_dstptr, ptr[parambase + OFFSET(dstptr)]);
    mov(reg_srcstride.cvt32(), ptr[parambase + OFFSET(srcstride)]);
    mov(reg_dststride.cvt32(), ptr[parambase + OFFSET(dststride)]);
    mov(reg_colsize.cvt32(), ptr[parambase + OFFSET(col)]);

    std::vector<Xbyak::Zmm> reg_srcs(row_pack), reg_tmps(row_pack);
    const int ZIDX_TranSrc = 0;
    const int ZIDX_TransTmp = row_pack;
    for (int i = 0; i < row_pack; i++) reg_srcs[i] = Xbyak::Zmm(ZIDX_TranSrc + i);
    for (int i = 0; i < row_pack; i++) reg_tmps[i] = Xbyak::Zmm(ZIDX_TransTmp + i);

    xor_(reg_iterrow, reg_iterrow);
    L(".rowloop");
    xor_(reg_itercol, reg_itercol);
    mov(reg_tmp2.cvt32(), ptr[parambase + OFFSET(row)]);
    sub(reg_tmp2, reg_iterrow);
    cmp(reg_tmp2, row_pack);
    jb(".tailrowloop", T_NEAR);

    L(".colloop");
    mov(reg_tmp1, reg_itercol);
    imul(reg_tmp1, reg_dststride);
    lea(reg_tmp, ptr[reg_dstptr + reg_tmp1]);
    lea(reg_tmp1, ptr[reg_srcptr + reg_itercol * src_bytes]);
    for (int jj = 0; jj < n_tile; jj += ne_zmm) {
      generate_Nbitsmask(mask_rd, reg_itercol, ptr[reg_colsize - jj], reg_tmp2, reg_tmp3, ne_zmm);
      for (int ii = 0; ii < row_pack; ii++) {
        const Xbyak::Xmm reg_srcs_ii = src_bytes_vmm == 64   ? Xbyak::Zmm(reg_srcs[ii].getIdx())
                                       : src_bytes_vmm == 32 ? Xbyak::Ymm(reg_srcs[ii].getIdx())
                                       : src_bytes_vmm == 16 ? Xbyak::Xmm(reg_srcs[ii].getIdx())
                                                             : (assert(false), reg_srcs[ii]);
        if (src_bytes == 1) {
          vmovdqu8(reg_srcs_ii | mask_rd | T_z, ptr[reg_tmp1 + ii * reg_srcstride + jj * src_bytes]);
        } else if (src_bytes == 2) {
          vmovdqu16(reg_srcs_ii | mask_rd | T_z, ptr[reg_tmp1 + ii * reg_srcstride + jj * src_bytes]);
        } else if (src_bytes == 4) {
          vmovdqu32(reg_srcs_ii | mask_rd | T_z, ptr[reg_tmp1 + ii * reg_srcstride + jj * src_bytes]);
        }
      }
      if (src_t == JblasF32 && dst_t == JblasBF16) {
        vcvtne2ps2bf16(reg_tmps[0], reg_srcs[1], reg_srcs[0]);
        vpermt2w(reg_tmps[0], vreg_idx0, reg_tmps[0]);
        vmovups(ptr[reg_tmp + jj * row_pack * dst_bytes], reg_tmps[0]);
      } else {
        // interleave_2rows_4regs(reg_srcs.data(), reg_tmps.data());
        assert(false);  // Not implemented
      }
    }
    add(reg_itercol, n_tile);
    cmp(reg_itercol.cvt32(), ptr[parambase + OFFSET(col)]);
    jb(".colloop");
    lea(reg_srcptr, ptr[reg_srcptr + row_pack * reg_srcstride]);
    lea(reg_dstptr, ptr[reg_dstptr + row_pack * n_tile * dst_bytes]);

    add(reg_iterrow, row_pack);
    cmp(reg_iterrow.cvt32(), ptr[parambase + OFFSET(row)]);
    jb(".rowloop");
    jmp(".aftercolloop", T_NEAR);

    L(".tailrowloop");
    L(".tailcolloop");
    mov(reg_tmp1, reg_itercol);
    imul(reg_tmp1, reg_dststride);
    lea(reg_tmp, ptr[reg_dstptr + reg_tmp1]);
    lea(reg_tmp1, ptr[reg_srcptr + reg_itercol * src_bytes]);
    for (int jj = 0; jj < n_tile; jj += ne_zmm) {
      generate_Nbitsmask(mask_rd, reg_itercol, ptr[reg_colsize - jj], reg_tmp2, reg_tmp3, ne_zmm);
      if (row_pack == 2) {
        const Xbyak::Xmm reg_srcs_0 = src_bytes_vmm == 64   ? Xbyak::Zmm(reg_srcs[0].getIdx())
                                      : src_bytes_vmm == 32 ? Xbyak::Ymm(reg_srcs[0].getIdx())
                                      : src_bytes_vmm == 16 ? Xbyak::Xmm(reg_srcs[0].getIdx())
                                                            : (assert(false), reg_srcs[0]);
        if (src_bytes == 1) {
          vmovdqu8(reg_srcs_0 | mask_rd | T_z, ptr[reg_tmp1 + jj * src_bytes]);
        } else if (src_bytes == 2) {
          vmovdqu16(reg_srcs_0 | mask_rd | T_z, ptr[reg_tmp1 + jj * src_bytes]);
        } else if (src_bytes == 4) {
          vmovdqu32(reg_srcs_0 | mask_rd | T_z, ptr[reg_tmp1 + jj * src_bytes]);
        }
        vxorps(reg_srcs[1], reg_srcs[1]);
      } else {
        assert(false);
      }
      if (src_t == JblasF32 && dst_t == JblasBF16) {
        vcvtne2ps2bf16(reg_tmps[0], reg_srcs[1], reg_srcs[0]);
        vpermt2w(reg_tmps[0], vreg_idx0, reg_tmps[0]);
        vmovups(ptr[reg_tmp + jj * row_pack * dst_bytes], reg_tmps[0]);
      } else {
        assert(false);
      }
    }
    add(reg_itercol, n_tile);
    cmp(reg_itercol.cvt32(), ptr[parambase + OFFSET(col)]);
    jb(".tailcolloop");
    L(".aftercolloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);
  }

  func_t mKernel = nullptr;

 public:
  template <int NTile, typename T_SRC, typename T_DST = T_SRC, int RowPack = 4 / sizeof(T_DST)>
  static void forward(const T_SRC* src, T_DST* dst, int row, int col, int row_pad, int col_pad, int src_step,
                      int dst_step) {
    const auto kern_col_pad = utils::padto(col, NTile);
    const auto kern_row_pad = utils::padto(row, RowPack);
    assert(kern_col_pad <= col_pad && col_pad % NTile == 0);
    assert(kern_row_pad <= row_pad && row_pad % RowPack == 0);
    const auto src_stride = static_cast<int>(sizeof(T_SRC)) * src_step;
    const auto dst_stride = static_cast<int>(sizeof(T_DST)) * dst_step;
    params param = {src, dst, row, col, src_stride, dst_stride};
    static const PaddingInterleaveCvt kern(NTile, utils::jblas_dtype<T_DST>, utils::jblas_dtype<T_SRC>, RowPack);
    kern(&param);

    // extra row and col pad
    const auto row_pad_size_memset = sizeof(T_DST) * (row_pad - kern_row_pad) * NTile;
    if (row_pad_size_memset) {
      for (int j = 0; j < kern_col_pad; j += NTile)
        memset(dst + j * dst_step + kern_row_pad * NTile, 0, row_pad_size_memset);
    }
    for (int j = kern_col_pad; j < col_pad; j += NTile)  //
      memset(dst + j * dst_step, 0, sizeof(T_DST) * NTile * row_pad);
  }

  template <int NTile, typename T_SRC, typename T_DST = T_SRC, int RowPack = 4 / sizeof(T_DST)>
  static void reference(const T_SRC* src, T_DST* dst, int row, int col, int row_pad, int col_pad, int src_step,
                        int dst_step) {
    assert(utils::padto(col, NTile) <= col_pad && col_pad % NTile == 0);
    assert(utils::padto(row, RowPack) <= row_pad && row_pad % RowPack == 0);
    for (int i = 0; i < row_pad; i += RowPack)
      for (int j = 0; j < col_pad; j += NTile)
        for (int ii = 0; ii < RowPack; ++ii)
          for (int jj = 0; jj < NTile; ++jj)
            dst[i * NTile + j * dst_step + ii + jj * RowPack] =
                static_cast<T_DST>((i + ii < row && j + jj < col) ? src[(i + ii) * src_step + j + jj] : 0);
  }
};

// src: row x col => dst: ⌈row/m_tile⌉ x ⌈col/(trans_cell*col_pack==64/sizeof(t_dst))⌉ x m_tile x col_pack (zeor-padded)
// Note1: the extra padding on the dimension of col due to the implementation limitation
// Note2: dst will only be zero-padded to a multiple of trans_cell in the dimension of m_tile
// Extra padding can be applied with memset calls in `static void forward(...)`
class PaddingTransInterleaveCvt : protected xbyak::JitAvx512f {
 public:
  struct params {
    const void* srcptr;
    void* dstptr;
    int row, col;
    int srcstride;  // src = src_base + srcstride * m_idx
    int dststride;  // dst = dst_base + dststride * m_idx, where m_idx % m_tile == 0
  };
  typedef void (*func_t)(params* p);
  void operator()(params* p) const { mKernel(p); }
  const int trans_cell;  // transpose matrices of size trans_cellxtrans_cell (in terms of #elements or #packs)

 private:
  PaddingTransInterleaveCvt(int m_tile, JBLAS_DTYPE dst_t) : PaddingTransInterleaveCvt(m_tile, dst_t, dst_t) {}
  PaddingTransInterleaveCvt(int m_tile, JBLAS_DTYPE dst_t, JBLAS_DTYPE src_t, int col_pack = 0)
      : xbyak::JitAvx512f(), trans_cell(64 / col_pack / int(utils::jblas_dtype_size(dst_t))) {
    const auto src_bytes = static_cast<int>(utils::jblas_dtype_size(src_t));
    const auto dst_bytes = static_cast<int>(utils::jblas_dtype_size(dst_t));
    if (col_pack == 0) col_pack = 4 / dst_bytes;  // default value
    // const auto src_bytes_vmm = ne_zmm * src_bytes;
    // const auto dst_bytes_vmm = ne_zmm * dst_bytes;

    assert(m_tile % trans_cell == 0);
    assert(col_pack > 0 && col_pack < 3);  // TODO(yi): int8 interleave not implemented

    inLocalLabel();                // use local label for multiple instance
    std::shared_ptr<void> epilogue{// generate code at the very end
                                   nullptr, [&](void*) {
                                     outLocalLabel();  // end of local label

                                     this->ready();
                                     this->mKernel = this->getCode<func_t>();
                                   }};
    Xbyak::util::StackFrame st(this, 1, 11 | Xbyak::util::UseRDX, 16 * 10);
    const Xbyak::Reg64& parambase = st.p[0];
    const Xbyak::Reg64& reg_srcptr = st.t[0];
    const Xbyak::Reg64& reg_dstptr = st.t[1];
    const Xbyak::Reg64& reg_srcstride = st.t[2];
    const Xbyak::Reg64& reg_dststride = st.t[3];
    const Xbyak::Reg64& reg_colsize = st.t[4];
    const Xbyak::Reg64& reg_iterrow = st.t[5];
    const Xbyak::Reg64& reg_itercol = st.t[6];
    const Xbyak::Reg64& reg_tmp = st.t[7];
    const Xbyak::Reg64& reg_tmp2 = st.t[9];
    const Xbyak::Reg64& reg_tmp3 = st.t[10];

    const Xbyak::Reg64& reg_ret = rax;
    const auto& mask_rd = k1;
    const auto& mask_rd2 = k2;

    vreg_push(rsp);
    mov(reg_srcptr, ptr[parambase + OFFSET(srcptr)]);
    mov(reg_srcstride.cvt32(), ptr[parambase + OFFSET(srcstride)]);
    mov(reg_dststride.cvt32(), ptr[parambase + OFFSET(dststride)]);
    mov(reg_colsize.cvt32(), ptr[parambase + OFFSET(col)]);

    std::vector<Xbyak::Zmm> reg_srcs(trans_cell), reg_tmps(trans_cell);
    const int ZIDX_TranSrc = 0;
    const int ZIDX_TransTmp = trans_cell;
    for (int i = 0; i < trans_cell; i++) reg_srcs[i] = Xbyak::Zmm(ZIDX_TranSrc + i);
    for (int i = 0; i < trans_cell; i++) reg_tmps[i] = Xbyak::Zmm(ZIDX_TransTmp + i);

    xor_(reg_iterrow, reg_iterrow);
    L(".rowloop");
    xor_(rdx, rdx);
    mov(rax, reg_iterrow);
    mov(reg_tmp, m_tile);
    div(reg_tmp);                                 // reg_iterrow `div` m_tile
    imul(reg_dstptr, rdx, col_pack * dst_bytes);  // ii * col_pack
    add(reg_dstptr, ptr[parambase + OFFSET(dstptr)]);
    imul(reg_tmp, rax, m_tile);
    imul(reg_tmp, reg_dststride);
    lea(reg_dstptr, ptr[reg_dstptr + reg_tmp]);  // dst = dst_base + i * dst_step + ii * col_pack
    xor_(reg_itercol, reg_itercol);

    mov(reg_tmp2.cvt32(), ptr[parambase + OFFSET(row)]);
    sub(reg_tmp2, reg_iterrow);
    cmp(reg_tmp2, trans_cell);
    jb(".tailrowloop", T_NEAR);

    L(".colloop");
    generate_Nbitsmask(mask_rd, reg_itercol, ptr[reg_colsize], reg_tmp2, reg_tmp3, 64 / dst_bytes);
    if (src_t == JblasF32 && dst_t == JblasBF16) {
      kshiftrq(mask_rd2, mask_rd, 16);
      assert(trans_cell == 16);
      for (int ii = 0; ii < trans_cell; ++ii) {
        lea(reg_tmp, (ii == 0) ? ptr[reg_srcptr + reg_itercol * src_bytes] : ptr[reg_tmp + reg_srcstride]);
        vmovups(reg_srcs[ii] | mask_rd | T_z, zword[reg_tmp]);
        vmovups(reg_tmps[ii] | mask_rd2 | T_z, zword[reg_tmp + 64]);
        vcvtne2ps2bf16(reg_srcs[ii], reg_tmps[ii], reg_srcs[ii]);
      }
      transpose16x16_4B(reg_srcs.data(), reg_tmps.data());
      for (int jj = 0; jj < trans_cell; ++jj) {
        vmovups(ptr[reg_dstptr + jj * m_tile * col_pack * dst_bytes], reg_srcs[jj]);
      }
    } else {
      assert(false);  // Not implemented
    }
    lea(reg_dstptr, ptr[reg_dstptr + col_pack * trans_cell * dst_bytes * m_tile]);
    lea(reg_itercol, ptr[reg_itercol + col_pack * trans_cell]);
    cmp(reg_itercol.cvt32(), ptr[parambase + OFFSET(col)]);
    jb(".colloop");

    imul(reg_tmp, reg_srcstride, trans_cell);
    lea(reg_srcptr, ptr[reg_srcptr + reg_tmp]);  // srcptr += trans_cell * srcstride
    lea(reg_iterrow, ptr[reg_iterrow + trans_cell]);
    cmp(reg_iterrow.cvt32(), ptr[parambase + OFFSET(row)]);
    jb(".rowloop");
    jmp(".aftercolloop", T_NEAR);

    L(".tailrowloop");
    // reg_itercol, reg_dstptr should have been set in the non-tail section
    Xbyak::Label l_tail_tbl;
    std::vector<Xbyak::Label> l_tail_case(trans_cell);
    mov(reg_tmp, l_tail_tbl);                              // TODO(Yi): rip + l + offset?
    jmp(ptr[reg_tmp + reg_tmp2 * sizeof(void*)], T_NEAR);  // switch(rows-iterrow) ...
    align(sizeof(intptr_t));
    L(l_tail_tbl);
    db(reinterpret_cast<uintptr_t>(nullptr), sizeof(intptr_t));  // case 0 should never occour
    for (int i = 1; i < trans_cell; ++i) putL(l_tail_case[i]);

    for (int m_tail = 1; m_tail < trans_cell; ++m_tail) {  // case (m_tail):
      auto& tailcolloop = l_tail_case[m_tail];
      L(tailcolloop);
      generate_Nbitsmask(mask_rd, reg_itercol, ptr[reg_colsize], reg_tmp2, reg_tmp3, 64 / dst_bytes);
      if (src_t == JblasF32 && dst_t == JblasBF16) {
        kshiftrq(mask_rd2, mask_rd, 16);
        assert(trans_cell == 16);
        for (int ii = 0; ii < trans_cell; ++ii) {
          if (ii < m_tail) {
            lea(reg_tmp, (ii == 0) ? ptr[reg_srcptr + reg_itercol * src_bytes] : ptr[reg_tmp + reg_srcstride]);
            vmovups(reg_srcs[ii] | mask_rd | T_z, zword[reg_tmp]);
            vmovups(reg_tmps[ii] | mask_rd2 | T_z, zword[reg_tmp + 64]);
            vcvtne2ps2bf16(reg_srcs[ii], reg_tmps[ii], reg_srcs[ii]);
          } else if (ii == m_tail) {
            vxorps(reg_srcs[ii], reg_srcs[ii], reg_srcs[ii]);
          } else {
            vmovaps(reg_srcs[ii], reg_srcs[m_tail]);
          }
        }
        transpose16x16_4B(reg_srcs.data(), reg_tmps.data());
        for (int jj = 0; jj < trans_cell; ++jj) {
          vmovups(ptr[reg_dstptr + jj * m_tile * col_pack * dst_bytes], reg_srcs[jj]);
        }
      } else {
        assert(false);  // Not implemented
      }
      lea(reg_dstptr, ptr[reg_dstptr + col_pack * trans_cell * dst_bytes * m_tile]);
      lea(reg_itercol, ptr[reg_itercol + col_pack * trans_cell]);
      cmp(reg_itercol.cvt32(), ptr[parambase + OFFSET(col)]);
      jb(tailcolloop);
      jmp(".aftercolloop", T_NEAR);
    }

    L(".aftercolloop");
    mov(reg_ret, 0);
    vreg_pop(rsp);
  }

  func_t mKernel = nullptr;

 public:
  template <int MTile, typename T_SRC, typename T_DST = T_SRC, int ColPack = 4 / sizeof(T_DST)>
  static void forward(const T_SRC* src, T_DST* dst, int row, int col, int row_pad, int col_pad, int src_step,
                      int dst_step) {
    assert(utils::padto(row, MTile) <= row_pad && row_pad % MTile == 0);
    assert(utils::padto(col, ColPack) <= col_pad && col_pad % ColPack == 0);
    static const PaddingTransInterleaveCvt kern(MTile, utils::jblas_dtype<T_DST>, utils::jblas_dtype<T_SRC>, ColPack);
    // 0-padded guarantee by jit kern
    const auto kern_row_pad = utils::padto(row, kern.trans_cell),
               kern_col_pad = utils::padto(col, kern.trans_cell * ColPack);
    assert(kern_row_pad <= row_pad && row_pad % MTile == 0);
    assert(kern_col_pad <= col_pad && col_pad % ColPack == 0);
    const auto src_stride = static_cast<int>(sizeof(T_SRC)) * src_step;
    const auto dst_stride = static_cast<int>(sizeof(T_DST)) * dst_step;
    params param = {src, dst, row, col, src_stride, dst_stride};
    kern(&param);

    // extra row and col pad
    const auto col_pad_size_memset = sizeof(T_DST) * (col_pad - kern_col_pad) * MTile;
    if (col_pad_size_memset) {
      for (int i = 0; i < kern_row_pad; i += MTile)
        memset(dst + i * dst_step + kern_col_pad * MTile, 0, col_pad_size_memset);
    }
    const auto row_tail_pad_size_memset = sizeof(T_DST) * (utils::padto(row, MTile) - kern_row_pad) * ColPack;
    if (row_tail_pad_size_memset) {  // row tail due to kernel limitation: kern_row_pad < next_multiple_of_MTile
      const auto kern_row_pad_le_mtile = utils::padto_le(kern_row_pad, MTile);
      const auto tail_dst_base = dst + kern_row_pad_le_mtile * dst_step + kern_row_pad % MTile * ColPack;
      for (int j = 0; j < kern_col_pad; j += ColPack) memset(tail_dst_base + j * MTile, 0, row_tail_pad_size_memset);
    }
    for (int j = utils::padto(row, MTile); j < row_pad; j += MTile)
      memset(dst + kern_row_pad * dst_step, 0, sizeof(T_DST) * MTile * col_pad);
  }

  template <int MTile, typename T_SRC, typename T_DST = T_SRC, int ColPack = 4 / sizeof(T_DST)>
  static void reference(const T_SRC* src, T_DST* dst, int row, int col, int row_pad, int col_pad, int src_step,
                        int dst_step) {
    assert(utils::padto(row, MTile) <= row_pad && row_pad % MTile == 0);
    assert(utils::padto(col, ColPack) <= col_pad && col_pad % ColPack == 0);
    for (int i = 0; i < row_pad; i += MTile)
      for (int j = 0; j < col_pad; j += ColPack)
        for (int ii = 0; ii < MTile; ++ii)
          for (int jj = 0; jj < ColPack; ++jj)
            dst[j * MTile + i * dst_step + jj + ii * ColPack] =
                static_cast<T_DST>((j + jj < col && i + ii < row) ? src[(i + ii) * src_step + j + jj] : 0);
  }
};

}  // namespace jit
}  // namespace kernel
}  // namespace jblas
