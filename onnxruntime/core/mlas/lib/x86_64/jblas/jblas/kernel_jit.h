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
#include <functional>
#include <unordered_map>

#include "jit_base.hpp"
#include "jit_blas_utils.h"
#include "kernel_jit_injector.h"
struct decompress_block_s4_f32_codegen_param {
  int row;
  int col;
  int kblock;
  int ld_src;
  int ld_dst;
  bool operator==(const decompress_block_s4_f32_codegen_param& other) const {
    return row == other.row && col == other.col && kblock == other.kblock && ld_src == other.ld_src &&
           ld_dst == other.ld_dst;
  }
};

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
    int irow = 0;
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
      int SF_TmpPos = 16 * 10;
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
    int irow = 0;
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

class DequanKBlockS4F32 {
 public:
  static inline void decompress_load_scale(Xbyak::CodeGenerator* jit, int zmm_scale_num,
                                           const std::vector<Xbyak::Zmm>& scale_zmms, const Xbyak::Reg64& reg_scale) {
    for (int i = 0; i < zmm_scale_num; i++) jit->vmovups(scale_zmms[i], jit->zword[reg_scale + i * 16 * sizeof(float)]);
  }

  struct convert_s4_s8_param {
    Xbyak::CodeGenerator* jit;
    Xbyak::RegExp src_addr;
    Xbyak::RegExp dst_addr;
    Xbyak::Zmm vmask;  // TODO: contain one tmp_zmm.
    Xbyak::Opmask load_mask;
    Xbyak::Opmask store_mask;
    Xbyak::Opmask unpack_mask;
    int free_zmm_idx;
  };

  template <int N>
  static inline void convert_s4_s8(convert_s4_s8_param p) {
    Xbyak::Ymm ymm(p.free_zmm_idx);
    if (N == 48) {
      p.jit->vmovdqu64(Xbyak::Ymm(p.free_zmm_idx) | p.load_mask, p.jit->yword[p.src_addr]);
      auto zmm = unpack_4bit(p.jit, ymm, Xbyak::Zmm(p.free_zmm_idx + 1), Xbyak::Zmm(p.free_zmm_idx + 2), p.vmask,
                             p.unpack_mask);
      p.jit->vmovdqu64(p.jit->ptr[p.dst_addr] | p.store_mask, zmm);
    }
    if (N == 64) {
      p.jit->vmovdqu(ymm, p.jit->yword[p.src_addr]);
      auto zmm = unpack_4bit(p.jit, ymm, Xbyak::Zmm(p.free_zmm_idx + 1), Xbyak::Zmm(p.free_zmm_idx + 2), p.vmask,
                             p.unpack_mask);
      p.jit->vmovdqu32(p.jit->ptr[p.dst_addr], zmm);
    }
  }

  template <int N>
  static inline void dequant_s8_N(Xbyak::CodeGenerator* jit, Xbyak::RegExp dst_addr, Xbyak::RegExp src_addr,
                                  const std::vector<Xbyak::Zmm>& zmms, int zmm_begin_idx,
                                  const std::vector<Xbyak::Zmm>& scales) {
    int constexpr VLoop = N / 16;
    for (int iv = 0; iv < VLoop; iv += 1) {
      jit->movdqu(Xbyak::Xmm(zmms[zmm_begin_idx + iv].getIdx()), jit->xword[src_addr + iv * 16]);
      jit->vpmovsxbd(zmms[iv], Xbyak::Xmm(zmms[zmm_begin_idx + iv].getIdx()));
      jit->vcvtdq2ps(zmms[iv], zmms[iv]);
      jit->vmulps(zmms[iv], zmms[iv], scales[iv]);
      jit->vmovups(jit->zword[dst_addr + iv * 16 * sizeof(float)], zmms[iv]);
    }
  }

  class decompress_block_s4_f32 : protected Xbyak::CodeGenerator {
    struct params {
      float* scale_addr;
      void* s4;
      void* s8;
      float* f32;
    };
    typedef void (*ker_t)(params* p);

   public:
    decompress_block_s4_f32(decompress_block_s4_f32_codegen_param p) : Xbyak::CodeGenerator(128 * 1024) {
      assert(p.col == 48);
      int zmm_scale_num = p.col / 16;
      Xbyak::Zmm zmm_mask(31);
      int data_zmm_num = 16;
      int data_idx = 0;
      int scale_idx = 16;
      std::vector<Xbyak::Zmm> scale_zmms(zmm_scale_num);
      std::vector<Xbyak::Zmm> data_zmms(data_zmm_num);
      std::transform(scale_zmms.begin(), scale_zmms.end(), scale_zmms.begin(),
                     [&](Xbyak::Zmm zmm) { return Xbyak::Zmm(scale_idx++); });
      std::transform(data_zmms.begin(), data_zmms.end(), data_zmms.begin(),
                     [&](Xbyak::Zmm zmm) { return Xbyak::Zmm(data_idx++); });
      inLocalLabel();
      Xbyak::Label const_v;
      {
        Xbyak::util::StackFrame sf(this, 1, 13, 64);
        auto& reg_param = sf.p[0];
        auto& reg_scale = sf.t[1];
        auto& reg_s4 = sf.t[2];
        auto& reg_s8 = sf.t[3];
        auto& reg_f32 = sf.t[4];
        auto& reg_tmp = sf.t[5];
        auto& reg_kblock_loop = sf.t[6];
        Xbyak::Opmask load_mask(2);
        Xbyak::Opmask store_mask(3);
        Xbyak::Opmask unpack_mask(4);
        vbroadcastss(zmm_mask, dword[rip + const_v]);
        mov(reg_tmp, 0xaaaaaaaaaaaaaaaa);
        kmovq(unpack_mask, reg_tmp);
        if (p.col == 48) {
          mov(reg_tmp.cvt32(), 0x7);
          kmovd(load_mask, reg_tmp.cvt32());
          mov(reg_tmp.cvt32(), 0x3f);
          kmovd(store_mask, reg_tmp.cvt32());

          mov(reg_scale, ptr[reg_param + OFFSET(scale_addr)]);
          mov(reg_s4, ptr[reg_param + OFFSET(s4)]);
          mov(reg_s8, ptr[reg_param + OFFSET(s8)]);
          mov(reg_f32, ptr[reg_param + OFFSET(f32)]);
          decompress_load_scale(this, zmm_scale_num, scale_zmms, reg_scale);
          auto max_row_unroll = data_zmm_num / zmm_scale_num;
          if (p.row != p.kblock) {
            int i = 0;
            for (int i = 0; i < p.row; i++) {
              convert_s4_s8_param cvt_p = {this,
                                           reg_s4 + i * p.ld_src,
                                           reg_s8 + i * 64,  // avoid leap the cache-line.
                                           zmm_mask,
                                           load_mask,
                                           store_mask,
                                           unpack_mask,
                                           data_zmms[0].getIdx()};
              convert_s4_s8<48>(cvt_p);
            }
            auto max_unroll_num = p.row / max_row_unroll;
            auto tail_row_unroll = p.row % max_row_unroll;
            for (; i < max_unroll_num; i++)
              for (int j = 0; j < max_row_unroll; j++)
                dequant_s8_N<48>(this, reg_f32 + (i * max_row_unroll + j) * p.ld_dst * sizeof(float),
                                 reg_s8 + (i * max_row_unroll + j) * 64, data_zmms, j * zmm_scale_num, scale_zmms);
            for (int k = 0; k < tail_row_unroll; k++)
              dequant_s8_N<48>(this, reg_f32 + (i * max_row_unroll + k) * p.ld_dst * sizeof(float),
                               reg_s8 + (i * max_row_unroll + k) * 64, data_zmms, k * zmm_scale_num, scale_zmms);
          } else {
            auto fin_row_unroll = max_row_unroll;
            while (p.col * fin_row_unroll % 64 != 0) fin_row_unroll -= 1;
            int Loop64 = p.col * fin_row_unroll / 64;
            xor_(reg_kblock_loop, reg_kblock_loop);
            L(".kblock_loop");
            imul(reg_tmp, reg_kblock_loop, p.ld_src);
            for (int j = 0; j < Loop64; j++) {
              convert_s4_s8_param cvt_p = {
                  this,        reg_s4 + reg_tmp + 32 * j, reg_s8 + j * 64, zmm_mask, load_mask, store_mask,
                  unpack_mask, data_zmms[0].getIdx()};
              convert_s4_s8<64>(cvt_p);
            }
            imul(reg_tmp, reg_kblock_loop, p.ld_dst * sizeof(float));
            for (int k = 0; k < fin_row_unroll; k++)
              dequant_s8_N<48>(this, reg_f32 + reg_tmp + k * p.ld_dst * sizeof(float), reg_s8 + k * 48, data_zmms,
                               k * zmm_scale_num, scale_zmms);
            add(reg_kblock_loop, fin_row_unroll);
            cmp(reg_kblock_loop, p.kblock);
            jl(".kblock_loop");
          }
        }
      }
      outLocalLabel();
      L(const_v);
      uint32_t const_value[] = {0xf0f0f0f0};
      db(reinterpret_cast<uint8_t*>(const_value), sizeof(const_value));
      this->ready();
      ker_ = this->getCode<ker_t>();
    }

    void fwd(float* scale, void* s4, void* tmp_buf, float* f32) {
      params p{scale, s4, tmp_buf, f32};
      ker_(&p);
    }

   private:
    ker_t ker_;
    int dump_idx = 0;
  };
};

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
    int SF_TmpPos = 16 * 10;
    Xbyak::util::StackFrame st(this, 1, 13, 16 * 10 + SF_TmpSize);
    const Xbyak::Reg64& parambase = st.p[0];
    const Xbyak::Reg64& reg_srcptr = st.t[0];
    const Xbyak::Reg64& reg_dstptr = st.t[1];
    const Xbyak::Reg64& reg_srcstride = st.t[2];
    const Xbyak::Reg64& reg_dststride = st.t[3];
    const Xbyak::Reg64& reg_rowsize = st.t[4];
    const Xbyak::Reg64& reg_size = st.t[5];
    const Xbyak::Reg64& reg_iterrow = st.t[6];
    const Xbyak::Reg64& reg_itercol = st.t[7];
    const Xbyak::Reg64& reg_tmp = st.t[8];
    const Xbyak::Reg64& reg_tmpsrc = st.t[9];
    const Xbyak::Reg64& reg_tmpdst = st.t[10];
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
}  // namespace jit
}  // namespace kernel
}  // namespace jblas
