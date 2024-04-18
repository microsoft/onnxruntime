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
#include <array>
#include <cassert>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "bestla.h"
#include "bestla_device.h"
#include "bestla_utils.h"
#include "bestla_jit.h"
#include "kernel_jit_injector.h"

namespace bestla {
namespace kernel {
namespace jit {

class DequanS8FP {
 public:
  class MicroKernelAVX512F : protected xbyak::JitAvx512f {
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
    MicroKernelAVX512F(BTLA_DTYPE dst_dt, bool is_sym_, int pack_row) {
      assert(dst_dt == BTLA_DTYPE::F32 || dst_dt == BTLA_DTYPE::BF16);
      is_sym = is_sym_;
      generate(dst_dt, pack_row);
      this->ready();
      mKernel = this->getCode<func_t>();
    }

    void generate(BTLA_DTYPE dst_dt, int pack_row) {
      assert(pack_row == 1 || pack_row == 2 || pack_row == 4);
      int zmm_scale_step = 64 / pack_row;
      Xbyak::Label data_label;
      inLocalLabel();  // use local label for multiple instance
      {
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

        auto get_dst_step = [&] {
          if (dst_dt == BTLA_DTYPE::BF16) return 2;
          return 4;  // f32 case.
        };

        auto generateNTile = [&](int N, BTLA_DTYPE dst_dt, int zmm_scale_step, std::string row_label) {
          if (pack_row == 2) {
            vmovups(Xbyak::Zmm(RegTmp), ptr[rip + data_label + 8]);
          } else if (pack_row == 4) {
            vmovups(Xbyak::Zmm(RegTmp), ptr[rip + data_label + 72]);
          }
          for (int i = 0; i < N; i++) {
            vmovups(Xbyak::Zmm(RegScale + i), ptr[reg_scaleptr + i * zmm_scale_step]);
            if (pack_row == 2 || pack_row == 4) {
              vpermd(Xbyak::Zmm(RegScale + i), Xbyak::Zmm(RegTmp), Xbyak::Zmm(RegScale + i));
            }
            if (!is_sym) {
              vpmovsxbd(Xbyak::Zmm(RegZP + i),
                        ptr[reg_zpptr + i * zmm_scale_step / sizeof(float)]);  // revert to zp_step.
              if (pack_row == 2 || pack_row == 4) {
                vpermd(Xbyak::Zmm(RegZP + i), Xbyak::Zmm(RegTmp), Xbyak::Zmm(RegZP + i));
              }
            }
          }
          xor_(reg_iterrow, reg_iterrow);
          mov(reg_tmp, reg_srcptr);
          mov(reg_tmp1, reg_dstptr);
          L(row_label);
          for (int i = 0; i < N; i++) {
            vpmovsxbd(Xbyak::Zmm(RegTmp), ptr[reg_tmp + i * 16]);
            if (!is_sym) {
              vpsubd(Xbyak::Zmm(RegTmp), Xbyak::Zmm(RegTmp), Xbyak::Zmm(RegZP + i));
            }
            vcvtdq2ps(Xbyak::Zmm(RegTmp), Xbyak::Zmm(RegTmp));
            vmulps(Xbyak::Zmm(RegTmp), Xbyak::Zmm(RegScale + i));
            if (dst_dt == BTLA_DTYPE::F32) {
              vmovups(ptr[reg_tmp1 + i * 64], Xbyak::Zmm(RegTmp));
            }
            if (dst_dt == BTLA_DTYPE::BF16) {
              Xbyak::Ymm ymm_v = Xbyak::Ymm(RegTmp);
              Xbyak::Zmm zmm_v = Xbyak::Zmm(RegTmp);
              if (device::CpuDevice::getInstance()->AVX512_BF16()) {
                vcvtneps2bf16(ymm_v, zmm_v);
              } else {
                vmovups(Xbyak::Zmm(31), zmm_v);
                vpsrldq(zmm_v, zmm_v, 2);
                vpandd(zmm_v, zmm_v, zword_b[rip + data_label]);
                vpaddd(zmm_v, zmm_v, zword_b[rip + data_label + 4]);
                vpaddd(zmm_v, zmm_v, Xbyak::Zmm(31));
                vpsrld(zmm_v, zmm_v, 16);
                vpmovdw(ymm_v, zmm_v);
              }
              vmovups(ptr[reg_tmp1 + i * 32], ymm_v);
            }
          }
          add(reg_tmp, reg_srcstride);
          add(reg_tmp1, reg_dststride);
          add(reg_iterrow, 1);
          cmp(reg_iterrow, reg_rowsize);
          jb(row_label);
        };

        L(".colloop");
        mov(reg_tmp, reg_colsize);
        sub(reg_tmp, reg_itercol);
        cmp(reg_tmp, 64);
        jl(".proc48", T_NEAR);
        generateNTile(4, dst_dt, zmm_scale_step, ".rowloop1");
        add(reg_itercol, 64);
        add(reg_srcptr, 1 * 64);
        add(reg_dstptr, get_dst_step() * 64);
        add(reg_scaleptr, 4 * 64 / pack_row);
        if (!is_sym) add(reg_zpptr, 1 * 64 / pack_row);
        jmp(".colend", T_NEAR);

        L(".proc48");
        cmp(reg_tmp, 48);
        jl(".proc32", T_NEAR);
        generateNTile(3, dst_dt, zmm_scale_step, ".rowloop2");
        add(reg_itercol, 48);
        add(reg_srcptr, 1 * 48);
        add(reg_dstptr, get_dst_step() * 48);
        add(reg_scaleptr, 4 * 48 / pack_row);
        if (!is_sym) add(reg_zpptr, 1 * 48 / pack_row);
        jmp(".colend", T_NEAR);

        L(".proc32");
        generateNTile(2, dst_dt, zmm_scale_step, ".rowloop3");
        add(reg_itercol, 32);
        add(reg_srcptr, 1 * 32);
        add(reg_dstptr, get_dst_step() * 32);
        add(reg_scaleptr, 4 * 32 / pack_row);
        if (!is_sym) add(reg_zpptr, 1 * 32 / pack_row);

        L(".colend");
        cmp(reg_itercol, reg_colsize);
        jb(".colloop");

        mov(reg_ret, 0);
        vreg_pop(rsp);
      }
      outLocalLabel();  // end of local label
      L(data_label);
      uint32_t bf16_cvt_magic_num[2] = {0x00000001, 0X00007FFF};
      uint32_t packrow2_permute_idx[16] = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7};
      uint32_t packrow4_permute_idx[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
      db(reinterpret_cast<uint8_t*>(bf16_cvt_magic_num), sizeof(bf16_cvt_magic_num));
      db(reinterpret_cast<uint8_t*>(packrow2_permute_idx), sizeof(packrow2_permute_idx));
      db(reinterpret_cast<uint8_t*>(packrow4_permute_idx), sizeof(packrow4_permute_idx));
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
  template <int PACK_ROW, typename _DST_T>
  static void forward_avx512f(int8_t* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst, float* scales,
                              int8_t* zero_points) {
    static MicroKernelAVX512F mAVX512FSym(utils::bestla_dtype<_DST_T>, true, PACK_ROW);
    static MicroKernelAVX512F mAVX512FASym(utils::bestla_dtype<_DST_T>, false, PACK_ROW);
    auto param = MicroKernelAVX512F::params{srcptr,
                                            dstptr,
                                            row,
                                            col,
                                            static_cast<int>(ld_src * sizeof(int8_t)),
                                            static_cast<int>(ld_dst * sizeof(_DST_T)),
                                            scales,
                                            zero_points};
    if (zero_points == nullptr) {
      mAVX512FSym.mKernel(&param);
    } else {
      mAVX512FASym.mKernel(&param);
    }
  }
};

class DequanKBlockS8Fp {
 public:
  template <int _PACK_ROW, typename _ST, typename _DST_T>
  static inline BTLA_CODE forward_avx512f(int8_t* srcptr, _DST_T* dstptr, int row, int col, int ld_src, int ld_dst,
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
      DequanS8FP::forward_avx512f<_PACK_ROW>(srcptr, dstptr, row0, col, ld_src, ld_dst, sptr, zptr);
      srcptr += row0 * ld_src;
      dstptr += row0 * ld_dst;
      sptr += NPad;
      if (zero_points != nullptr) zptr += NPad;
    }
    for (int i = 0; i < row1_blk; i += kblock) {
      DequanS8FP::forward_avx512f<_PACK_ROW>(srcptr, dstptr, kblock, col, ld_src, ld_dst, sptr, zptr);
      srcptr += kblock * ld_src;
      dstptr += kblock * ld_dst;
      sptr += NPad;
      if (zero_points != nullptr) zptr += NPad;
    }
    if (row2 > 0) {
      DequanS8FP::forward_avx512f<_PACK_ROW>(srcptr, dstptr, row2, col, ld_src, ld_dst, sptr, zptr);
    }
    return BTLA_CODE::Success;
  }
};

struct DataConvertConfig {
  enum class cvt_direct {
    NO_CVT,
    BF16_TO_FP32,
    FP32_TO_BF16,
    F16_TO_FP32,
    FP32_TO_F16,
  };

  DataConvertConfig(BTLA_DTYPE src_t, BTLA_DTYPE dst_t, std::vector<kernel::jit_injector::eltwise_injector> injectors) {
    input_dt = src_t;
    output_dt = dst_t;
    if (injectors.size() != 0) {
      assert(src_t == BTLA_DTYPE::F32 || src_t == BTLA_DTYPE::BF16 || src_t == BTLA_DTYPE::F16);
      if (src_t == BTLA_DTYPE::BF16) before_postop = DataConvertConfig::cvt_direct::BF16_TO_FP32;
      if (src_t == BTLA_DTYPE::F16) before_postop = DataConvertConfig::cvt_direct::F16_TO_FP32;
    }
    // once contain postop, data_type before store will be fp32.
    if (injectors.size() != 0 || src_t == BTLA_DTYPE::F32) {
      if (dst_t == BTLA_DTYPE::BF16) before_store = DataConvertConfig::cvt_direct::FP32_TO_BF16;
      if (dst_t == BTLA_DTYPE::F16) {
        if (!device::CpuDevice::getInstance()->AVX512_FP16()) assert(0);
        before_store = DataConvertConfig::cvt_direct::FP32_TO_F16;
      }
    } else if (src_t == BTLA_DTYPE::BF16 && dst_t == BTLA_DTYPE::F32) {
      before_store = DataConvertConfig::cvt_direct::BF16_TO_FP32;
    } else if (src_t == BTLA_DTYPE::F16 && dst_t == BTLA_DTYPE::F32) {
      assert(device::CpuDevice::getInstance()->AVX512_FP16());
      before_store = DataConvertConfig::cvt_direct::F16_TO_FP32;
    }
  }

  int get_store_process_bytes(int VBytes) {
    if (before_store == DataConvertConfig::cvt_direct::BF16_TO_FP32 ||
        before_store == DataConvertConfig::cvt_direct::F16_TO_FP32)
      return 2 * VBytes;
    if (before_store == DataConvertConfig::cvt_direct::FP32_TO_BF16 ||
        before_store == DataConvertConfig::cvt_direct::FP32_TO_F16)
      return VBytes / 2;
    return VBytes;
  }

  cvt_direct before_postop = cvt_direct::NO_CVT;
  cvt_direct before_store = cvt_direct::NO_CVT;
  BTLA_DTYPE input_dt, output_dt;
};

template <typename SIMD_REG>
struct MemcpyStoreParam {
  SIMD_REG vmm_v;
  Xbyak::RegExp store_addr;
  bool tail;
  Xbyak::Opmask store_mask = Xbyak::util::k1;
};

class JitMemcpy2DAvx2 : protected xbyak::JitAvx2 {
 public:
  struct params {
    void *srcptr, *dstptr, *elt_const_v;
    int col;
  };
  typedef long long (*func_t)(params*);

 public:
  static int constexpr VBytes = 32;
  JitMemcpy2DAvx2(int unroll_row, BTLA_DTYPE src_t, BTLA_DTYPE dst_t,
                  std::vector<kernel::jit_injector::eltwise_injector> injectors = {}) {
    DataConvertConfig dt_cvt_cfg(src_t, dst_t, injectors);
    generate(unroll_row, injectors, dt_cvt_cfg);
  }

  template <typename _SRC_T, typename _DST_T>
  static BTLA_CODE forward(const _SRC_T* srcptr, _DST_T* dstptr, int row, int col, int srcstep, int dststep,
                           void* elt_const_v = nullptr) {
    static JitMemcpy2DAvx2 instance_withops(1, utils::bestla_dtype<_SRC_T>, utils::bestla_dtype<_DST_T>);
    for (int i = 0; i < row; i++) {
      auto param = params{reinterpret_cast<char*>(const_cast<_SRC_T*>(srcptr)) + i * srcstep * sizeof(_SRC_T),
                          reinterpret_cast<char*>(dstptr) + i * dststep * sizeof(_DST_T), elt_const_v,
                          static_cast<int>(col * sizeof(_SRC_T))};
      instance_withops.mKernel(&param);
    }
    return BTLA_CODE::Success;
  }

  template <typename _SRC_T, typename _DST_T, BTLA_ELTWISEOP Op>
  static BTLA_CODE forward1(const _SRC_T* srcptr, _DST_T* dstptr, int row, int col, int srcstep, int dststep,
                            void* elt_const_v = nullptr) {
    static JitMemcpy2DAvx2 instance_withops(1, utils::bestla_dtype<_SRC_T>, utils::bestla_dtype<_DST_T>,
                                            {kernel::jit_injector::eltwise_injector(Op)});
    for (int i = 0; i < row; i++) {
      auto param = params{reinterpret_cast<char*>(const_cast<_SRC_T*>(srcptr)) + i * srcstep * sizeof(_SRC_T),
                          reinterpret_cast<char*>(dstptr) + i * dststep * sizeof(_DST_T), elt_const_v,
                          static_cast<int>(col * sizeof(_SRC_T))};
      instance_withops.mKernel(&param);
    }
    return BTLA_CODE::Success;
  }

 protected:
  void generate(int unrollk, std::vector<kernel::jit_injector::eltwise_injector>& injectors,
                DataConvertConfig dt_cvt_cfg) {
    // unrollK=[1,2]
    if (unrollk != 1 && unrollk != 2) {
      assert(false);
      return;
    }
    Xbyak::Label data_label;
    inLocalLabel();  // use local label for multiple instance
    {
      int SF_TmpSize = 64;
      int SF_TmpPos = 16 * 10;
      Xbyak::util::StackFrame st(this, 1, 13, 16 * 10 + SF_TmpSize);
      const Xbyak::Reg64& parambase = st.p[0];
      const Xbyak::Reg64& reg_srcptr = st.t[0];
      const Xbyak::Reg64& reg_dstptr = st.t[1];
      const Xbyak::Reg64& reg_colsize = st.t[2];
      const Xbyak::Reg64& reg_itercol = st.t[3];
      const Xbyak::Reg64& reg_tmp = st.t[4];
      const Xbyak::Reg64& reg_elt_constv = st.t[5];  // alias of reg_tmp.
      const Xbyak::Reg64& reg_tmp1 = st.t[6];
      const Xbyak::Reg64& reg_tmp2 = st.t[7];
      const Xbyak::Reg64& reg_ret = rax;

      vreg_push(rsp);

      mov(reg_srcptr, ptr[parambase + OFFSET(srcptr)]);
      mov(reg_dstptr, ptr[parambase + OFFSET(dstptr)]);
      mov(reg_elt_constv, ptr[parambase + OFFSET(elt_const_v)]);

      load32(reg_colsize, ptr[parambase + OFFSET(col)]);
      int const ColUnroll = 4;
      int const ymm_tmp_num = 2;
      std::array<Xbyak::Ymm, ymm_tmp_num> ymm_tmps = {Xbyak::Ymm(unrollk * ColUnroll),
                                                      Xbyak::Ymm(unrollk * ColUnroll + 1)};
      for (int i = 0; i < unrollk * ColUnroll; i++) used_ymm_idx.insert(i);
      for (auto&& injector : injectors) {
        injector.assign_resources(this, used_ymm_idx, reg_ret);
        injector.assign_reg_elt_constp(reg_elt_constv);
      }

      auto store_ymm_v = [&](MemcpyStoreParam<Xbyak::Ymm> p) { vmovups(ptr[p.store_addr], p.vmm_v); };

      auto unpack_ymm_16bit_withfunc = [&](MemcpyStoreParam<Xbyak::Ymm> p,
                                           std::function<void(MemcpyStoreParam<Xbyak::Ymm>)> func,
                                           BTLA_DTYPE BIT16_DT) {
        vmovups(ymm_tmps[0], p.vmm_v);
        Xbyak::Ymm ymm_v = Xbyak::Ymm(p.vmm_v.getIdx());
        if (BIT16_DT == BTLA_DTYPE::BF16) {
          vpmovzxwd(p.vmm_v, ymm_v);
          vpslld(p.vmm_v, p.vmm_v, 16);
        }
        func(p);
        vextractf128(Xbyak::Xmm(ymm_tmps[0].getIdx()), ymm_tmps[0], 1);
        if (BIT16_DT == BTLA_DTYPE::BF16) {
          vpmovzxwd(ymm_tmps[0], Xbyak::Ymm(ymm_tmps[0].getIdx()));
          vpslld(ymm_tmps[0], ymm_tmps[0], 16);
        }
        p.vmm_v = ymm_tmps[0];
        p.store_addr = p.store_addr + VBytes;
        func(p);
      };

      auto apply_postop_and_store = [&](MemcpyStoreParam<Xbyak::Ymm> p) {
        for (int k = 0; k < injectors.size(); k++) injectors[k].vector_compute(p.vmm_v, k * 3 * sizeof(float));
        if (dt_cvt_cfg.before_store == DataConvertConfig::cvt_direct::NO_CVT) {
          store_ymm_v(p);
        } else if (dt_cvt_cfg.before_store == DataConvertConfig::cvt_direct::BF16_TO_FP32) {
          unpack_ymm_16bit_withfunc(p, store_ymm_v, BTLA_DTYPE::BF16);
        } else if (dt_cvt_cfg.before_store == DataConvertConfig::cvt_direct::FP32_TO_BF16) {
          Xbyak::Xmm xmm_v = Xbyak::Xmm(p.vmm_v.getIdx());
          Xbyak::Xmm xmm_tmp = Xbyak::Xmm(ymm_tmps[1].getIdx());
          vmovups(ymm_tmps[0], p.vmm_v);
          vpsrldq(p.vmm_v, p.vmm_v, 2);
          mov(reg_tmp.cvt32(), 0x00000001);
          vmovd(xmm_tmp, reg_tmp.cvt32());
          vpbroadcastd(ymm_tmps[1], xmm_tmp);
          vpand(p.vmm_v, p.vmm_v, ymm_tmps[1]);
          mov(reg_tmp.cvt32(), 0x00007FFF);
          vmovd(xmm_tmp, reg_tmp.cvt32());
          vpbroadcastd(ymm_tmps[1], xmm_tmp);
          vpaddd(p.vmm_v, p.vmm_v, ymm_tmps[1]);
          vpaddd(p.vmm_v, p.vmm_v, ymm_tmps[0]);
          vpshufb(p.vmm_v, p.vmm_v, ptr[rip + data_label + 32]);
          vpermq(p.vmm_v, p.vmm_v, 0x58);
          vmovups(ptr[p.store_addr], xmm_v);
        } else {
          assert(0);
        }
      };

      auto load_store_value = [&](Xbyak::Ymm ymm_v, Xbyak::RegExp load_addr, Xbyak::RegExp store_addr) {
        vmovups(ymm_v, ptr[load_addr]);
        if (dt_cvt_cfg.before_postop == DataConvertConfig::cvt_direct::NO_CVT) {
          apply_postop_and_store({ymm_v, store_addr});
        } else if (dt_cvt_cfg.before_postop == DataConvertConfig::cvt_direct::BF16_TO_FP32) {
          unpack_ymm_16bit_withfunc({ymm_v, store_addr}, apply_postop_and_store, BTLA_DTYPE::BF16);
        } else {
          assert(0);
        }
      };

      xor_(reg_itercol, reg_itercol);

      L(".colloop");
      mov(reg_tmp, reg_colsize);
      sub(reg_tmp, reg_itercol);
      cmp(reg_tmp, ColUnroll * VBytes);
      jl(".maskproc", T_NEAR);

      for (int i = 0; i < ColUnroll; i++)
        load_store_value(Xbyak::Ymm(i), reg_srcptr + i * VBytes,
                         reg_dstptr + i * dt_cvt_cfg.get_store_process_bytes(VBytes));

      add(reg_srcptr, ColUnroll * VBytes);
      add(reg_dstptr, ColUnroll * dt_cvt_cfg.get_store_process_bytes(VBytes));
      add(reg_itercol, ColUnroll * VBytes);
      jmp(".colend", T_NEAR);
      L(".maskproc");
      mov(reg_tmp2, reg_colsize);
      sub(reg_tmp2, reg_itercol);
      cmp(reg_tmp2, VBytes);
      jb(".maskflag", T_NEAR);
      cmp(reg_tmp2, 0);
      jl(".maskend", T_NEAR);
      load_store_value(Xbyak::Ymm(0), reg_srcptr, reg_dstptr);
      jmp(".maskend", T_NEAR);
      L(".maskflag");
      // 0<tail<8
      mov(reg_tmp1.cvt32(), 1);
      shlx(reg_tmp1.cvt32(), reg_tmp1.cvt32(), reg_tmp2.cvt32());
      sub(reg_tmp1.cvt32(), 1);
      vmovd(Xbyak::Xmm(1), reg_tmp1.cvt32());
      vpbroadcastd(Xbyak::Ymm(1), Xbyak::Xmm(1));
      vpsllvd(Xbyak::Ymm(1), Xbyak::Ymm(1), ptr[rip + data_label]);
      mov(reg_elt_constv, ptr[parambase + OFFSET(elt_const_v)]);
      vpmaskmovd(Xbyak::Ymm(0), Xbyak::Ymm(1), ptr[reg_srcptr]);
      for (int k = 0; k < injectors.size(); k++) injectors[k].vector_compute(Xbyak::Ymm(0), k * 3 * sizeof(float));
      vpmaskmovd(ptr[reg_dstptr], Xbyak::Ymm(1), Xbyak::Ymm(0));
      L(".maskend");
      add(reg_srcptr, VBytes);
      add(reg_dstptr, dt_cvt_cfg.get_store_process_bytes(VBytes));
      add(reg_itercol, VBytes);
      L(".colend");
      cmp(reg_itercol, reg_colsize);
      jb(".colloop");
      mov(reg_ret, 0);
      vreg_pop(rsp);
    }
    outLocalLabel();  // end of local label
    L(data_label);
    uint32_t mask_bias[8] = {28, 24, 20, 16, 12, 8, 4, 0};
    const uint8_t avx2_bf16_convert_maigc_num[32] = {0x02, 0x03, 0x06, 0x07, 0x0a, 0x0b, 0x0e, 0x0f, 0x80, 0x80, 0x80,
                                                     0x80, 0x80, 0x80, 0x80, 0x80, 0x02, 0x03, 0x06, 0x07, 0x0a, 0x0b,
                                                     0x0e, 0x0f, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80};
    db(reinterpret_cast<uint8_t*>(mask_bias), sizeof(mask_bias));
    db(avx2_bf16_convert_maigc_num, sizeof(avx2_bf16_convert_maigc_num));
    for (auto&& injector : injectors) injector.prepare_table();
    this->ready();
    mKernel = this->getCode<func_t>();
  }

  func_t mKernel = nullptr;
  std::set<int> used_ymm_idx;
};

class JitMemcpy2DAvx512f : protected xbyak::JitAvx512f {
 public:
  struct params {
    void *srcptr, *dstptr, *elt_const_v;
    int col;
  };
  typedef long long (*func_t)(params*);

 public:
  static int constexpr VBytes = 64;
  JitMemcpy2DAvx512f(int unroll_row, BTLA_DTYPE src_t, BTLA_DTYPE dst_t,
                     std::vector<kernel::jit_injector::eltwise_injector> injectors = {}) {
    DataConvertConfig dt_cvt_cfg(src_t, dst_t, injectors);
    generate(unroll_row, injectors, dt_cvt_cfg);
  }

  template <typename _SRC_T, typename _DST_T>
  static BTLA_CODE forward(const _SRC_T* srcptr, _DST_T* dstptr, int row, int col, int srcstep, int dststep,
                           void* elt_const_v = nullptr) {
    static JitMemcpy2DAvx512f instance_withops(1, utils::bestla_dtype<_SRC_T>, utils::bestla_dtype<_DST_T>);

    for (int i = 0; i < row; i++) {
      auto param = params{reinterpret_cast<char*>(const_cast<_SRC_T*>(srcptr)) + i * srcstep * sizeof(_SRC_T),
                          reinterpret_cast<char*>(dstptr) + i * dststep * sizeof(_DST_T), elt_const_v,
                          static_cast<int>(col * sizeof(_SRC_T))};
      instance_withops.mKernel(&param);
    }
    return BTLA_CODE::Success;
  }

  template <typename _SRC_T, typename _DST_T, BTLA_ELTWISEOP Op>
  static BTLA_CODE forward1(const _SRC_T* srcptr, _DST_T* dstptr, int row, int col, int srcstep, int dststep,
                            void* elt_const_v = nullptr) {
    static JitMemcpy2DAvx512f instance_withops(1, utils::bestla_dtype<_SRC_T>, utils::bestla_dtype<_DST_T>,
                                               {kernel::jit_injector::eltwise_injector(Op)});
    for (int i = 0; i < row; i++) {
      auto param = params{reinterpret_cast<char*>(const_cast<_SRC_T*>(srcptr)) + i * srcstep * sizeof(_SRC_T),
                          reinterpret_cast<char*>(dstptr) + i * dststep * sizeof(_DST_T), elt_const_v,
                          static_cast<int>(col * sizeof(_SRC_T))};
      instance_withops.mKernel(&param);
    }
    return BTLA_CODE::Success;
  }

 protected:
  void generate(int unrollk, std::vector<kernel::jit_injector::eltwise_injector>& injectors,
                DataConvertConfig dt_cvt_cfg) {
    inLocalLabel();  // use local label for multiple instance
    Xbyak::Label data_label;
    {
      int SF_TmpSize = 64;
      Xbyak::util::StackFrame st(this, 1, 13, 10 * 16 + SF_TmpSize);
      const Xbyak::Reg64& parambase = st.p[0];
      const Xbyak::Reg64& reg_src = st.t[0];
      const Xbyak::Reg64& reg_dst = st.t[1];
      const Xbyak::Reg64& reg_size = st.t[2];
      const Xbyak::Reg64& reg_iter = st.t[3];
      const Xbyak::Reg64& reg_tmp = st.t[4];
      const Xbyak::Reg64& reg_tmp2 = st.t[5];
      const Xbyak::Reg64& reg_elt_constv = st.t[6];
      const Xbyak::Reg64& reg_ret = rax;

      vreg_push(rsp);

      int const ColUnroll = 4;
      int const zmm_tmp_num = 2;
      std::array<Xbyak::Zmm, zmm_tmp_num> zmm_tmps = {Xbyak::Zmm(unrollk * ColUnroll),
                                                      Xbyak::Zmm(unrollk * ColUnroll + 1)};
      for (int i = 0; i < unrollk * ColUnroll; i++) used_zmm_idx.insert(i);
      for (int i = 0; i < zmm_tmp_num; i++) used_zmm_idx.insert(i + unrollk * ColUnroll);
      for (auto&& injector : injectors) {
        injector.assign_resources(this, used_zmm_idx, reg_ret, k2);
        injector.assign_reg_elt_constp(reg_elt_constv);
      }

      auto store_zmm_v = [&](MemcpyStoreParam<Xbyak::Zmm> p) {
        if (p.tail) {
          vmovdqu8(ptr[p.store_addr], p.vmm_v | p.store_mask);
        } else {
          vmovups(ptr[p.store_addr], p.vmm_v);
        }
      };

      auto unpack_zmm_16bit_withfunc = [&](MemcpyStoreParam<Xbyak::Zmm> p,
                                           std::function<void(MemcpyStoreParam<Xbyak::Zmm>)> func,
                                           BTLA_DTYPE BIT16_DT) {
        vmovups(zmm_tmps[0], p.vmm_v);
        Xbyak::Ymm ymm_v = Xbyak::Ymm(p.vmm_v.getIdx());
        if (BIT16_DT == BTLA_DTYPE::BF16) {
          vpmovzxwd(p.vmm_v, ymm_v);
          vpslld(p.vmm_v, p.vmm_v, 16);
        }
        if (BIT16_DT == BTLA_DTYPE::F16) vcvtph2psx(p.vmm_v, ymm_v);
        p.store_mask = k3;
        func(p);
        vextractf32x8(Xbyak::Ymm(zmm_tmps[0].getIdx()), zmm_tmps[0], 1);
        if (BIT16_DT == BTLA_DTYPE::BF16) {
          vpmovzxwd(zmm_tmps[0], Xbyak::Ymm(zmm_tmps[0].getIdx()));
          vpslld(zmm_tmps[0], zmm_tmps[0], 16);
        }
        if (BIT16_DT == BTLA_DTYPE::F16) vcvtph2psx(zmm_tmps[0], Xbyak::Ymm(zmm_tmps[0].getIdx()));
        p.vmm_v = zmm_tmps[0];
        p.store_addr = p.store_addr + VBytes;
        p.store_mask = k4;
        func(p);
      };

      auto apply_postop_and_store = [&](MemcpyStoreParam<Xbyak::Zmm> p) {
        for (int k = 0; k < injectors.size(); k++) injectors[k].vector_compute(p.vmm_v, k * 3 * sizeof(float));
        if (dt_cvt_cfg.before_store == DataConvertConfig::cvt_direct::NO_CVT) {
          store_zmm_v(p);
        } else if (dt_cvt_cfg.before_store == DataConvertConfig::cvt_direct::BF16_TO_FP32) {
          unpack_zmm_16bit_withfunc(p, store_zmm_v, BTLA_DTYPE::BF16);
        } else if (dt_cvt_cfg.before_store == DataConvertConfig::cvt_direct::F16_TO_FP32) {
          unpack_zmm_16bit_withfunc(p, store_zmm_v, BTLA_DTYPE::F16);
        } else if (dt_cvt_cfg.before_store == DataConvertConfig::cvt_direct::FP32_TO_BF16) {
          Xbyak::Ymm ymm_v = Xbyak::Ymm(p.vmm_v.getIdx());
          if (device::CpuDevice::getInstance()->AVX512_BF16()) {
            vcvtneps2bf16(ymm_v, p.vmm_v);
          } else {
            vmovups(zmm_tmps[1], p.vmm_v);
            vpsrldq(p.vmm_v, p.vmm_v, 2);
            vpandd(p.vmm_v, p.vmm_v, zword_b[rip + data_label]);
            vpaddd(p.vmm_v, p.vmm_v, zword_b[rip + data_label + 4]);
            vpaddd(p.vmm_v, p.vmm_v, zmm_tmps[1]);
            vpsrld(p.vmm_v, p.vmm_v, 16);
            vpmovdw(ymm_v, p.vmm_v);
          }
          if (p.tail) {
            vmovdqu8(ptr[p.store_addr], ymm_v | k3);
          } else {
            vmovups(ptr[p.store_addr], ymm_v);
          }
        } else if (dt_cvt_cfg.before_store == DataConvertConfig::cvt_direct::FP32_TO_F16) {
          Xbyak::Ymm ymm_v = Xbyak::Ymm(p.vmm_v.getIdx());
          vcvtps2phx(ymm_v, p.vmm_v);
          if (p.tail) {
            vmovdqu8(ptr[p.store_addr], ymm_v | k3);
          } else {
            vmovups(ptr[p.store_addr], ymm_v);
          }
        } else {
          assert(0);
        }
      };

      auto load_store_value = [&](Xbyak::Zmm zmm_v, Xbyak::RegExp load_addr, Xbyak::RegExp store_addr,
                                  bool tail = false) {
        if (tail) {
          vmovdqu8(zmm_v | k1, ptr[load_addr]);
        } else {
          vmovups(zmm_v, ptr[load_addr]);
        }
        if (dt_cvt_cfg.before_postop == DataConvertConfig::cvt_direct::NO_CVT) {
          apply_postop_and_store({zmm_v, store_addr, tail});
        } else if (dt_cvt_cfg.before_postop == DataConvertConfig::cvt_direct::BF16_TO_FP32) {
          unpack_zmm_16bit_withfunc({zmm_v, store_addr, tail}, apply_postop_and_store, BTLA_DTYPE::BF16);
        } else if (dt_cvt_cfg.before_postop == DataConvertConfig::cvt_direct::F16_TO_FP32) {
          unpack_zmm_16bit_withfunc({zmm_v, store_addr, tail}, apply_postop_and_store, BTLA_DTYPE::F16);
        } else {
          assert(0);
        }
      };

      mov(reg_elt_constv, ptr[parambase + OFFSET(elt_const_v)]);
      mov(reg_src, ptr[parambase + OFFSET(srcptr)]);
      mov(reg_dst, ptr[parambase + OFFSET(dstptr)]);
      load32(reg_size, ptr[parambase + OFFSET(col)]);
      xor_(reg_iter, reg_iter);
      L(".colloop");
      mov(reg_tmp, reg_size);
      sub(reg_tmp, reg_iter);
      cmp(reg_tmp, ColUnroll * VBytes);
      jl(".maskproc", T_NEAR);
      for (int i = 0; i < ColUnroll; i++)
        load_store_value(Xbyak::Zmm(i), reg_src + i * VBytes, reg_dst + i * dt_cvt_cfg.get_store_process_bytes(VBytes));
      add(reg_src, ColUnroll * VBytes);
      add(reg_dst, ColUnroll * dt_cvt_cfg.get_store_process_bytes(VBytes));
      add(reg_iter, ColUnroll * VBytes);
      jmp(".colend", T_NEAR);
      L(".maskproc");
      generate_Nbitsmask(k1, reg_iter, reg_size, reg_tmp, reg_tmp2, VBytes);

      if (dt_cvt_cfg.before_store == DataConvertConfig::cvt_direct::FP32_TO_BF16) {
        push(reg_iter);
        push(reg_size);
        int vbytes = VBytes;
        // consider a case that input==bf16 but apply postop, betore sotre will be fp32_to_bf16 but need to normal gen
        // mask.
        if (dt_cvt_cfg.input_dt == BTLA_DTYPE::F32) {
          shr(reg_iter, 1);
          shr(reg_size, 1);
          vbytes /= 2;
        }
        generate_Nbitsmask(k3, reg_iter, reg_size, reg_tmp, reg_tmp2, vbytes);
        pop(reg_size);
        pop(reg_iter);
      }
      // once enable postop the data-type before store will not be bf16.
      if (dt_cvt_cfg.before_store == DataConvertConfig::cvt_direct::BF16_TO_FP32 ||
          dt_cvt_cfg.before_store == DataConvertConfig::cvt_direct::F16_TO_FP32) {
        push(reg_iter);
        push(reg_size);
        shl(reg_iter, 1);
        shl(reg_size, 1);
        generate_Nbitsmask(k3, reg_iter, reg_size, reg_tmp, reg_tmp2, VBytes);
        add(reg_iter, VBytes);
        generate_Nbitsmask(k4, reg_iter, reg_size, reg_tmp, reg_tmp2, VBytes);
        pop(reg_size);
        pop(reg_iter);
      }
      load_store_value(Xbyak::Zmm(0), reg_src, reg_dst, true);
      add(reg_src, VBytes);
      add(reg_dst, dt_cvt_cfg.get_store_process_bytes(VBytes));
      add(reg_iter, VBytes);
      L(".colend");
      cmp(reg_iter, reg_size);
      jb(".colloop");
      mov(reg_ret, 0);
      vreg_pop(rsp);
    }
    outLocalLabel();  // end of local label
    L(data_label);
    uint32_t bf16_cvt_magic_num[2] = {0x00000001, 0X00007FFF};
    db(reinterpret_cast<uint8_t*>(bf16_cvt_magic_num), sizeof(bf16_cvt_magic_num));
    for (auto&& injector : injectors) injector.prepare_table();
    this->ready();
    mKernel = this->getCode<func_t>();
  }

  func_t mKernel = nullptr;
  std::set<int> used_zmm_idx;
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

class DecompressS4S8_AVX512F : protected xbyak::JitAvx512f {
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

  static BTLA_CODE forward(void* srcptr, void* dstptr, size_t size) {
    static DecompressS4S8_AVX512F instance;
    auto param = params{srcptr, dstptr, size};
    instance.mKernel(&param);
    return BTLA_CODE::Success;
  }

 private:
  func_t mKernel = nullptr;
};

static inline BTLA_CODE decompress_s4_s8(utils::int4x2* srcptr, int8_t* dstptr, int row, int col, int ld_src,
                                         int ld_dst) {
  if (col != ld_src) {  // memory is not continuous
    return BTLA_CODE::NotSupport;
  }
  DecompressS4S8_AVX512F::forward(srcptr, dstptr, (size_t)row * col);
  return BTLA_CODE::Success;
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

  PaddingInterleaveCvt(int n_tile, BTLA_DTYPE dst_t) : PaddingInterleaveCvt(n_tile, dst_t, dst_t) {}
  PaddingInterleaveCvt(int n_tile, BTLA_DTYPE dst_t, BTLA_DTYPE src_t, int row_pack = 0) : xbyak::JitAvx512f() {
    inLocalLabel();  // use local label for multiple instance
    const auto src_bytes = static_cast<int>(utils::bestla_dtype_size(src_t));
    const auto dst_bytes = static_cast<int>(utils::bestla_dtype_size(dst_t));
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
      if (src_t == BTLA_DTYPE::F32 && dst_t == BTLA_DTYPE::BF16) {
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
      if (src_t == BTLA_DTYPE::F32 && dst_t == BTLA_DTYPE::BF16) {
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
    static const PaddingInterleaveCvt kern(NTile, utils::bestla_dtype<T_DST>, utils::bestla_dtype<T_SRC>, RowPack);
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
  PaddingTransInterleaveCvt(int m_tile, BTLA_DTYPE dst_t) : PaddingTransInterleaveCvt(m_tile, dst_t, dst_t) {}
  PaddingTransInterleaveCvt(int m_tile, BTLA_DTYPE dst_t, BTLA_DTYPE src_t, int col_pack = 0)
      : xbyak::JitAvx512f(), trans_cell(64 / col_pack / int(utils::bestla_dtype_size(dst_t))) {
    const auto src_bytes = static_cast<int>(utils::bestla_dtype_size(src_t));
    const auto dst_bytes = static_cast<int>(utils::bestla_dtype_size(dst_t));
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
    if (src_t == BTLA_DTYPE::F32 && dst_t == BTLA_DTYPE::BF16) {
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
    db(reinterpret_cast<uintptr_t>(nullptr), sizeof(intptr_t));  // case 0 should never occur
    for (int i = 1; i < trans_cell; ++i) putL(l_tail_case[i]);

    for (int m_tail = 1; m_tail < trans_cell; ++m_tail) {  // case (m_tail):
      auto& tailcolloop = l_tail_case[m_tail];
      L(tailcolloop);
      generate_Nbitsmask(mask_rd, reg_itercol, ptr[reg_colsize], reg_tmp2, reg_tmp3, 64 / dst_bytes);
      if (src_t == BTLA_DTYPE::F32 && dst_t == BTLA_DTYPE::BF16) {
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
    static const PaddingTransInterleaveCvt kern(MTile, utils::bestla_dtype<T_DST>, utils::bestla_dtype<T_SRC>, ColPack);
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

// Complex number matrix(interleaved) - vector(as diagonal matrix) multiplication; Typically used for
// shift-RoPE
//
// vector: fp16 values; view every adjacent 2 values on colunm as a complex num
// src: bf16 ⌈row/row_pack⌉ x n_tile x row_pack; view every adjacent 2 values on colunm as a complex num
// dst: same as src
class CScaleInterleavedBF16FP16 : protected xbyak::JitAvx512_fp16 {
 public:
  struct params {
    void* srcptr;
    const void* scaleptr;
    int row;
  };
  typedef void (*func_t)(params* p);
  void operator()(params* p) const { mKernel(p); }

 private:
  explicit CScaleInterleavedBF16FP16(int n_tile, int n_off, int row_pack = 2, int unroll = 2)
      : xbyak::JitAvx512_fp16() {
    inLocalLabel();  // use local label for multiple instance
    assert(("n_tile must be a multiple of 16", n_tile % 16 == 0));
    assert(row_pack > 0 && row_pack < 3);  // TODO(yi): int8 interleave not implemented
    int SF_TmpSize = 64;
    std::shared_ptr<void> epilogue{// generate code at the very end
                                   nullptr, [&](void*) {
                                     outLocalLabel();  // end of local label
                                     this->ready();
                                     this->mKernel = this->getCode<func_t>();
                                   }};
    Xbyak::util::StackFrame st(this, 1, 4, 16 * 10 + SF_TmpSize);
    const Xbyak::Reg64& parambase = st.p[0];
    const Xbyak::Reg64& reg_src = st.t[0];
    const Xbyak::Reg64& reg_scale = st.t[1];
    const Xbyak::Reg64& reg_rowsize = st.t[2];
    const Xbyak::Reg64& reg_iterrow = st.t[3];
    const Xbyak::Zmm& vreg_scale = zmm31;
    const auto& mask = k1;
    const auto masked_off = n_off % 16;
    if (masked_off != 0) {
      mov(reg_src, ((1ULL << (16 - masked_off)) - 1) << masked_off);
      kmovw(mask, reg_src.cvt32());
    }

    vreg_push(rsp);
    mov(reg_rowsize.cvt32(), ptr[parambase + OFFSET(row)]);
    mov(reg_src, qword[parambase + OFFSET(srcptr)]);
    mov(reg_scale, qword[parambase + OFFSET(scaleptr)]);

    std::vector<Xbyak::Zmm> vreg_src(4 * n_tile / 16);
    const int ZIDX_TranSrc = 0;
    for (int i = 0; i < 4 * n_tile / 16; i++) vreg_src[i] = Xbyak::Zmm(ZIDX_TranSrc + i);

    xor_(reg_iterrow, reg_iterrow);
    Xbyak::Label rowloop;
    L(rowloop);
    {
      assert(("only implement for pack2 bf16", row_pack == 2));
      for (int i = 0; i < unroll * row_pack; i += row_pack) {
        vpbroadcastd(vreg_scale, dword[reg_scale + reg_iterrow * sizeof(utils::fp16) + i * sizeof(utils::fp16)]);

        if (masked_off != 0) {
          int j = utils::padto_le(n_off, 16);

          const auto& vreg0 = vreg_src[j / 16 * 4 + 0];
          const auto& vreg1 = vreg_src[j / 16 * 4 + 1];
          const auto& vreg2 = vreg_src[j / 16 * 4 + 2];
          const auto& vreg3 = vreg_src[j / 16 * 4 + 3];
          vpmovzxwd(vreg0, yword[reg_src + (i * n_tile + j * row_pack) * sizeof(utils::bf16) + 0]);
          vpmovzxwd(vreg1, yword[reg_src + (i * n_tile + j * row_pack) * sizeof(utils::bf16) + 32]);
          vpslldq(vreg0, vreg0, 2);
          vpslldq(vreg1, vreg1, 2);
          vcvtps2phx(Xbyak::Ymm(vreg0.getIdx()), vreg0);
          vcvtps2phx(Xbyak::Ymm(vreg1.getIdx()), vreg1);
          // #UD If (dest_reg == src1_reg) or (dest_reg == src2_reg)
          vfmulcph(Xbyak::Ymm(vreg2.getIdx()), Xbyak::Ymm(vreg0.getIdx()), Xbyak::Ymm(vreg_scale.getIdx()));
          vfmulcph(Xbyak::Ymm(vreg3.getIdx()), Xbyak::Ymm(vreg1.getIdx()), Xbyak::Ymm(vreg_scale.getIdx()));
          vcvtph2psx(vreg0, Xbyak::Ymm(vreg2.getIdx()));
          vcvtph2psx(vreg1, Xbyak::Ymm(vreg3.getIdx()));
          vcvtne2ps2bf16(vreg0, vreg1, vreg0);
          vmovups(zword[reg_src + (i * n_tile + j * row_pack) * sizeof(utils::bf16)] | mask, vreg0);
        }

        for (int j = utils::padto(n_off, 16); j < n_tile; j += 16) {
          const auto& vreg0 = vreg_src[j / 16 * 4 + 0];
          const auto& vreg1 = vreg_src[j / 16 * 4 + 1];
          const auto& vreg2 = vreg_src[j / 16 * 4 + 2];
          const auto& vreg3 = vreg_src[j / 16 * 4 + 3];
          vpmovzxwd(vreg0, yword[reg_src + (i * n_tile + j * row_pack) * sizeof(utils::bf16) + 0]);
          vpmovzxwd(vreg1, yword[reg_src + (i * n_tile + j * row_pack) * sizeof(utils::bf16) + 32]);
          vpslldq(vreg0, vreg0, 2);
          vpslldq(vreg1, vreg1, 2);
          vcvtps2phx(Xbyak::Ymm(vreg0.getIdx()), vreg0);
          vcvtps2phx(Xbyak::Ymm(vreg1.getIdx()), vreg1);
          // #UD If (dest_reg == src1_reg) or (dest_reg == src2_reg)
          vfmulcph(Xbyak::Ymm(vreg2.getIdx()), Xbyak::Ymm(vreg0.getIdx()), Xbyak::Ymm(vreg_scale.getIdx()));
          vfmulcph(Xbyak::Ymm(vreg3.getIdx()), Xbyak::Ymm(vreg1.getIdx()), Xbyak::Ymm(vreg_scale.getIdx()));
          vcvtph2psx(vreg0, Xbyak::Ymm(vreg2.getIdx()));
          vcvtph2psx(vreg1, Xbyak::Ymm(vreg3.getIdx()));
          vcvtne2ps2bf16(vreg0, vreg1, vreg0);
          vmovups(zword[reg_src + (i * n_tile + j * row_pack) * sizeof(utils::bf16)], vreg0);
        }
      }
    }
    lea(reg_iterrow, ptr[reg_iterrow + unroll * row_pack]);
    lea(reg_src, ptr[reg_src + unroll * row_pack * n_tile * sizeof(utils::bf16)]);
    cmp(reg_iterrow, reg_rowsize);
    jb(rowloop);

    vreg_pop(rsp);
  }

  func_t mKernel = nullptr;

 public:
  template <int NTile, int RowPack = 2>
  static void forward(utils::bf16* src, const utils::fp16* scale, int row, int col, int src_step, int n_offset) {
    static_assert(RowPack == 2, "Only implement rowpack2 bf16");
    static_assert(NTile % 16 == 0, "NTile must be a multiple of 16");
    constexpr auto unroll = 2;
    assert(("row should be paded", row % (RowPack * unroll) == 0));
    assert(("cow should be paded", col % NTile == 0));
    assert(("can not skip more than col", n_offset < col));
    int j = utils::padto_le(n_offset, NTile);
    if (n_offset % NTile != 0) {
      static const CScaleInterleavedBF16FP16 kern_off(NTile, n_offset % NTile, RowPack, unroll);
      params param = {src + j * src_step, scale, row};
      kern_off(&param);
      j += NTile;
    }

    for (; j < col; j += NTile) {
      static const CScaleInterleavedBF16FP16 kern(NTile, 0, RowPack, unroll);
      params param = {src + j * src_step, scale, row};
      kern(&param);
    }
  }

  template <int NTile, int RowPack = 2>
  static void reference(utils::bf16* src, const utils::fp16* scale, int row, int col, int src_step, int n_offset) {
    static_assert(RowPack == 2, "Only implement rowpack2 bf16");
    static_assert(NTile % 16 == 0, "NTile must be a multiple of 16");
    assert(("row should be paded", row % RowPack == 0));
    assert(("cow should be paded", col % NTile == 0));
    assert(("can not skip more than col", n_offset < col));
    for (int j = 0; j < col; j += NTile) {
      for (int i = 0; i < row; i += RowPack) {
        for (int jj = 0; jj < NTile; ++jj) {
          if (j + jj < n_offset) continue;
          auto& rel = (src + j * src_step)[i * NTile + jj * RowPack + 0];
          auto& img = (src + j * src_step)[i * NTile + jj * RowPack + 1];
          const auto rel_f32 = static_cast<float>(rel);
          const auto img_f32 = static_cast<float>(img);
          const auto rel_scale = static_cast<float>(scale[i + 0]);
          const auto img_scale = static_cast<float>(scale[i + 1]);
          rel = static_cast<utils::bf16>(rel_f32 * rel_scale - img_f32 * img_scale);
          img = static_cast<utils::bf16>(rel_f32 * img_scale + img_f32 * rel_scale);
        }
      }
    }
  }
};

}  // namespace jit
}  // namespace kernel
}  // namespace bestla
