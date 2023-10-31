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
#include <stdint.h>

#include <cstddef>
#include <type_traits>
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

#define OFFSET(field) offsetof(params, field)

namespace jblas {

namespace xbyak {
class JitBase : protected Xbyak::CodeGenerator {
 protected:
  JitBase(size_t size = 16 * 1024) : CodeGenerator(size) {}

  void load32(const Xbyak::Reg64& reg, const Xbyak::Address& addr) {
    xor_(reg, reg);
    mov(reg.cvt32(), addr);
  }

  void vreg_push(const Xbyak::Reg64& baseaddr) {
#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(xword[baseaddr + i * 16], Xbyak::Xmm(6 + i));
    }
#endif
  }

  void vreg_pop(const Xbyak::Reg64& baseaddr) {
#ifdef _WIN32
    for (int i = 0; i < 10; i++) {
      movaps(Xbyak::Xmm(6 + i), xword[baseaddr + i * 16]);
    }
#endif
  }

  void generate_Nbitsmask(const Xbyak::Opmask& _msk, const Xbyak::Reg64& _pos, const Xbyak::Address& _total,
                          const Xbyak::Reg64& _tmp, const Xbyak::Reg64& _tmp1, int N) {
    inLocalLabel();
    lea(_tmp, _total);
    sub(_tmp, _pos);
    cmp(_tmp, N);
    jb(".maskflag");
    cmp(_tmp, 0);
    jl(".zeroflag");
    uint64_t allmask = ((uint64_t)1 << N) - 1;
    if (N == 64) {
      allmask = (uint64_t)-1;
    }
    mov(_tmp, allmask);
    kmovq(_msk, _tmp);
    jmp(".maskend");
    L(".maskflag");
    mov(_tmp1, 1);
    shlx(_tmp1, _tmp1, _tmp);
    sub(_tmp1, 1);
    kmovq(_msk, _tmp1);
    jmp(".maskend");
    L(".zeroflag");
    mov(_tmp1, 0);
    kmovq(_msk, _tmp1);
    L(".maskend");
    outLocalLabel();
  }
  void generate_Nbitsmask(const Xbyak::Opmask& _msk, const Xbyak::Reg64& _pos, const Xbyak::Reg64& _total,
                          const Xbyak::Reg64& _tmp, const Xbyak::Reg64& _tmp1, int N) {
    generate_Nbitsmask(_msk, _pos, ptr[_total], _tmp, _tmp1, N);
  }
};

class JitAvx : protected JitBase {
 protected:
  static int constexpr VBits = 256;
  typedef Xbyak::Ymm vreg_t;
};

class JitAvx2 : protected JitAvx {
 protected:
  static int constexpr VBits = 256;
  typedef Xbyak::Ymm vreg_t;

  void loadbf16_f32(const Xbyak::Ymm& dst, const Xbyak::Address& addr) {
    vpmovzxwd(dst, addr);
    vpslld(dst, dst, 16);
  }
};

class JitAvx512f : protected JitAvx2 {
 protected:
  static int constexpr VBits = 512;
  typedef Xbyak::Zmm vreg_t;

  void interleave_2rows_4regs(Xbyak::Zmm* src_2regs, Xbyak::Zmm* tmp_2reg) {
    vpunpcklwd(tmp_2reg[0], src_2regs[0], src_2regs[1]);
    vpunpckhwd(tmp_2reg[1], src_2regs[0], src_2regs[1]);
    vshuff32x4(src_2regs[0], tmp_2reg[0], tmp_2reg[1], 0 | (1 << 2) | (0 << 4) | (1 << 6));
    vshuff32x4(src_2regs[0], src_2regs[0], src_2regs[0], 0 | (2 << 2) | (1 << 4) | (3 << 6));
    vshuff32x4(src_2regs[1], tmp_2reg[0], tmp_2reg[1], 2 | (3 << 2) | (2 << 4) | (3 << 6));
    vshuff32x4(src_2regs[1], src_2regs[1], src_2regs[1], 0 | (2 << 2) | (1 << 4) | (3 << 6));
  }

  void transpose16x16_4B(Xbyak::Zmm* src, Xbyak::Zmm* tmp, const int N = 16) {
    for (int i = 0; i < 8; ++i) {
      vpunpckldq(tmp[2 * i + 0], src[2 * i], src[2 * i + 1]);
      vpunpckhdq(tmp[2 * i + 1], src[2 * i], src[2 * i + 1]);
    }

    for (int i = 0; i < 4; ++i) {
      vpunpcklqdq(src[4 * i + 0], tmp[4 * i + 0], tmp[4 * i + 2]);
      vpunpckhqdq(src[4 * i + 1], tmp[4 * i + 0], tmp[4 * i + 2]);
      vpunpcklqdq(src[4 * i + 2], tmp[4 * i + 1], tmp[4 * i + 3]);
      vpunpckhqdq(src[4 * i + 3], tmp[4 * i + 1], tmp[4 * i + 3]);
    }

    for (int i = 0; i < 2; ++i) {
      vshufi32x4(tmp[8 * i + 0], src[8 * i + 0], src[8 * i + 4], 0x88);
      vshufi32x4(tmp[8 * i + 1], src[8 * i + 1], src[8 * i + 5], 0x88);
      vshufi32x4(tmp[8 * i + 2], src[8 * i + 2], src[8 * i + 6], 0x88);
      vshufi32x4(tmp[8 * i + 3], src[8 * i + 3], src[8 * i + 7], 0x88);
      vshufi32x4(tmp[8 * i + 4], src[8 * i + 0], src[8 * i + 4], 0xdd);
      vshufi32x4(tmp[8 * i + 5], src[8 * i + 1], src[8 * i + 5], 0xdd);
      vshufi32x4(tmp[8 * i + 6], src[8 * i + 2], src[8 * i + 6], 0xdd);
      vshufi32x4(tmp[8 * i + 7], src[8 * i + 3], src[8 * i + 7], 0xdd);
    }

    // last step and move out
    for (int i = 0; i < N; ++i) {
      vshufi32x4(src[i], tmp[i % 8], tmp[8 + i % 8], i < 8 ? 0x88 : 0xdd);
    }
  }

  void interleave_4rows_6regs(Xbyak::Zmm* src_4regs, Xbyak::Zmm* tmp_regs, const Xbyak::Opmask* masks) {
    vpunpcklbw(tmp_regs[0], src_4regs[0], src_4regs[1]);
    vpunpckhbw(tmp_regs[1], src_4regs[0], src_4regs[1]);
    vpunpcklbw(tmp_regs[2], src_4regs[2], src_4regs[3]);
    vpunpckhbw(tmp_regs[3], src_4regs[2], src_4regs[3]);

    vpunpcklwd(tmp_regs[4], tmp_regs[0], tmp_regs[2]);
    vpunpckhwd(tmp_regs[5], tmp_regs[0], tmp_regs[2]);
    vpunpcklwd(tmp_regs[0], tmp_regs[1], tmp_regs[3]);
    vpunpckhwd(tmp_regs[2], tmp_regs[1], tmp_regs[3]);
    vshuff32x4(tmp_regs[1], tmp_regs[4], tmp_regs[0], (4 << 4) | 4);
    vshuff32x4(tmp_regs[3], tmp_regs[5], tmp_regs[2], (4 << 4) | 4);
    vmovups(src_4regs[0], tmp_regs[1]);
    vshuff32x4(src_4regs[0] | masks[0], tmp_regs[3], tmp_regs[3], 0 | (0 << 2) | (0 << 4) | (2 << 6));
    vmovups(src_4regs[1], tmp_regs[3]);
    vshuff32x4(src_4regs[1] | masks[1], tmp_regs[1], tmp_regs[1], 1 | (0 << 2) | (3 << 4) | (0 << 6));
    vshuff32x4(tmp_regs[1], tmp_regs[4], tmp_regs[0], (14 << 4) | 14);
    vshuff32x4(tmp_regs[3], tmp_regs[5], tmp_regs[2], (14 << 4) | 14);
    vmovups(src_4regs[2], tmp_regs[1]);
    vshuff32x4(src_4regs[2] | masks[0], tmp_regs[3], tmp_regs[3], 0 | (0 << 2) | (0 << 4) | (2 << 6));
    vmovups(src_4regs[3], tmp_regs[3]);
    vshuff32x4(src_4regs[3] | masks[1], tmp_regs[1], tmp_regs[1], 1 | (0 << 2) | (3 << 4) | (0 << 6));
  }

  void cvt_fp32_bf16(const Xbyak::Ymm& _bf16, const Xbyak::Zmm& _fp32) {
    vpsrld(_fp32, _fp32, 16);
    vpmovdw(_bf16, _fp32);
  }

  void loadbf16_f32(const Xbyak::Zmm& dst, const Xbyak::Address& addr) {
    vpmovzxwd(dst, addr);
    vpslld(dst, dst, 16);
  }

  void broadcastbf16_f32(const Xbyak::Zmm& dst, const Xbyak::Reg64& tmp, const Xbyak::Address& addr) {
    mov(tmp.cvt16(), addr);
    shl(tmp.cvt32(), 16);
    vpbroadcastd(dst, tmp.cvt32());
  }

  void store_fp32_bf16(const Xbyak::Zmm& _fp32, const Xbyak::Address& _add) {
    auto bf16 = Xbyak::Ymm(_fp32.getIdx());
    cvt_fp32_bf16(bf16, _fp32);
    vmovups(_add, bf16);
  }
};

class JitAvx512_fp16 : protected JitAvx512f {};

class JitAvx512vnni : protected JitAvx512f {
 protected:
  void vpdpbusds_evex(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2, const Xbyak::Operand& op) {
    vpdpbusds(x1, x2, op, Xbyak::EvexEncoding);
  }
};

class JitAvxvnni : protected JitAvx2 {
 protected:
  void vpdpbusds_vex(const Xbyak::Xmm& x1, const Xbyak::Xmm& x2, const Xbyak::Operand& op) {
    vpdpbusds(x1, x2, op, Xbyak::VexEncoding);
  }
};

class JitAmxtile : protected JitAvx512f {
 public:
  struct alignas(64) tileconfig_t {
    uint8_t palette_id;
    uint8_t reserved[15];
    uint16_t colb[16];
    uint8_t rows[16];
  };

  static void configure_tiles(tileconfig_t& tc, int TILE_M, int TILE_N, int TILE_K, int elesize, int ANum, int BNum,
                              int CNum) {
    // Filling tile configure structure. Could be done offline.
    tc.palette_id = 1;
    // Configure C tiles
    int t = 0;
    for (; t < CNum; ++t) {
      tc.rows[t] = uint8_t(TILE_M);
      tc.colb[t] = uint16_t(TILE_N * 4);
    }
    // Configure A tiles
    for (; t < CNum + ANum; ++t) {
      tc.rows[t] = uint8_t(TILE_M);
      tc.colb[t] = uint16_t(TILE_K * elesize);
    }
    // Configure B tile. B effectively has 64 rows and 16 columns.
    int kpack = 4 / elesize;
    for (; t < CNum + ANum + BNum; ++t) {
      tc.rows[t] = uint8_t(TILE_K / kpack);
      tc.colb[t] = uint16_t(TILE_N * 4);
    }
  }
};

class JitAmxbf16 : protected JitAmxtile {
 protected:
  void cvt_fp32_bf16(const Xbyak::Ymm& _bf16, const Xbyak::Zmm& _fp32) { vcvtneps2bf16(_bf16, _fp32); }
};

class JitAmxint8 : protected JitAmxtile {
 protected:
  template <class, class>
  void _tdpb(const Xbyak::Tmm& x1, const Xbyak::Tmm& x2, const Xbyak::Tmm& x3);
};
template <>
inline void JitAmxint8::_tdpb<int8_t, int8_t>(const Xbyak::Tmm& x1, const Xbyak::Tmm& x2, const Xbyak::Tmm& x3) {
  tdpbssd(x1, x2, x3);
}
template <>
inline void JitAmxint8::_tdpb<int8_t, uint8_t>(const Xbyak::Tmm& x1, const Xbyak::Tmm& x2, const Xbyak::Tmm& x3) {
  tdpbsud(x1, x2, x3);
}
template <>
inline void JitAmxint8::_tdpb<uint8_t, int8_t>(const Xbyak::Tmm& x1, const Xbyak::Tmm& x2, const Xbyak::Tmm& x3) {
  tdpbusd(x1, x2, x3);
}
template <>
inline void JitAmxint8::_tdpb<uint8_t, uint8_t>(const Xbyak::Tmm& x1, const Xbyak::Tmm& x2, const Xbyak::Tmm& x3) {
  tdpbuud(x1, x2, x3);
}
}  // namespace xbyak
}  // namespace jblas
