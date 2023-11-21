//  Copyright (c) 2022 Intel Corporation
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

#include <utility>
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <set>
#include <array>

#include "jit_blas.h"
#include "jit_blas_utils.h"
#include "xbyak/xbyak.h"

namespace jblas {
namespace kernel {
namespace jit_injector {
using Zmm = Xbyak::Zmm;
using Ymm = Xbyak::Ymm;
using Xmm = Xbyak::Xmm;
class eltwise_injector {
 public:
  eltwise_injector(JBLAS_ELTWISEOP eltwiseop) : elt_op(eltwiseop) { reigster_table_entries(); }
  virtual ~eltwise_injector() {}

  void assign_resources(Xbyak::CodeGenerator* ptr, const std::set<int>& used_zmm_idx, const Xbyak::Reg64& table_reg,
                        const Xbyak::Opmask& mask_reg) {
    h = ptr;
    k_mask = mask_reg;
    p_table = table_reg;
    assert(used_zmm_idx.size() <= 26);
    assign_zmm(used_zmm_idx, &zmm_mask);
    assign_zmm(used_zmm_idx, &zmm_aux0);
    assign_zmm(used_zmm_idx, &zmm_aux1);
    assign_zmm(used_zmm_idx, &zmm_aux2);
    assign_zmm(used_zmm_idx, &zmm_aux3);
    assign_zmm(used_zmm_idx, &zmm_aux4);
  }
  void assign_resources(Xbyak::CodeGenerator* ptr, const std::set<int>& used_ymm_idx, const Xbyak::Reg64& table_reg) {
    h = ptr;
    p_table = table_reg;
    assert(used_ymm_idx.size() <= 10);
    assign_ymm(used_ymm_idx, &ymm_mask);
    assign_ymm(used_ymm_idx, &ymm_aux0);
    assign_ymm(used_ymm_idx, &ymm_aux1);
    assign_ymm(used_ymm_idx, &ymm_aux2);
    assign_ymm(used_ymm_idx, &ymm_aux3);
    assign_ymm(used_ymm_idx, &ymm_aux4);
  }
  void assign_reg_elt_constp(const Xbyak::Reg64& reg) { reg_rt_const_p = reg; }
  void vector_compute(const Xbyak::Zmm& zmm_src, int const_p_offset = 0) {
    load_table_addr();
    switch (elt_op) {
      case EXP:
        exp_compute_vector_fwd(zmm_src);
        break;
      case TANH:
        tanh_compute_vector_fwd(zmm_src);
        break;
      case GELU:
        gelu_compute_vector_fwd(zmm_src);
        break;
      case RELU:
        relu_compute_vector_fwd(zmm_src, const_p_offset);
        break;
      case LINEAR:
        linear_compute_vector_fwd(zmm_src, const_p_offset);
        break;
      case LOW_PRECISION_EXP:
        low_precision_exp_compute_vector_fwd(zmm_src);
        break;
      case SWISH:
        swish_compute_vector_fwd(zmm_src, const_p_offset);
        break;
      default:
        assert(false);
        break;
    }
  }
  void vector_compute(const Xbyak::Ymm& ymm_src, int const_p_offset = 0) {
    load_table_addr();
    switch (elt_op) {
      case EXP:
        exp_compute_vector_fwd(ymm_src);
        break;
      case TANH:
        tanh_compute_vector_fwd(ymm_src);
        break;
      case GELU:
        gelu_compute_vector_fwd(ymm_src);
        break;
      case LOW_PRECISION_EXP:
        low_precision_exp_compute_vector_fwd(ymm_src);
        break;
      case SWISH:
        swish_compute_vector_fwd(ymm_src, const_p_offset);
        break;
      default:
        assert(false);
        break;
    }
  }
  void prepare_table() {
    h->align(64);
    h->L(l_table);
    assert(sizeof(table_entry_val_t) == 4);  // sizeof(table_entry_val_t) should be 4
    for (auto it = entry_map.begin(); it != entry_map.end(); it++) {
      const auto& te = (*it).second;
      const auto len = te.bcast ? 64u : sizeof(table_entry_val_t);
      for (size_t d = 0; d < len; d += sizeof(table_entry_val_t)) h->dd(te.val);
    }
  }

 private:
  void reigster_table_entries() {
    static const table_t common_values{
        {zero, {0x00000000, true}},      {half, {0x3f000000, true}},          {one, {0x3f800000, true}},
        {two, {0x40000000, true}},       {minus_one, {0xbf800000, true}},     {minus_two, {0xc0000000, true}},
        {ln2f, {0x3f317218, true}},      {one_epi32, {0x00000001, true}},     {positive_mask, {0x7fffffff, true}},
        {sign_mask, {0x80000000, true}}, {exponent_bias, {0x0000007f, true}},
    };

    static constexpr std::array<float, 3> exp_approx_f32_coeff{0.35815147f, 0.96963238f, 1.f};
    static const table_t low_precision_exp_consts{
        {low_precision_exp_const_v0, {jblas::utils::bit_cast<uint32_t>(exp_approx_f32_coeff[0]), true}},
        {low_precision_exp_const_v1, {jblas::utils::bit_cast<uint32_t>(exp_approx_f32_coeff[1]), true}},
        {low_precision_exp_const_v2, {jblas::utils::bit_cast<uint32_t>(exp_approx_f32_coeff[2]), true}},
    };

    static const table_t exp_consts{{exp_log2ef, {0x3fb8aa3b, true}},
                                    {exp_ln_flt_max_f, {0x42b17218, true}},
                                    {exp_ln_flt_min_f, {0xc2aeac50, true}}};

    static const table_t exp_polynomial{
        // p0 = 1.0f
        {exp_pol, {0x3f7ffffb, true}},  // p1 = 0.999999701f
        {exp_pol, {0x3efffee3, true}},  // p2 = 0.499991506f
        {exp_pol, {0x3e2aad40, true}},  // p3 = 0.166676521f
        {exp_pol, {0x3d2b9d0d, true}},  // p4 = 0.0418978221f
        {exp_pol, {0x3c07cfce, true}}   // p5 = 0.00828929059f
    };

    static const table_t gelu_tanh_const{{gelu_tanh_fitting_const, {0x3d372713, true}},
                                         {gelu_tanh_fitting_const_times_three, {0x3e095d4f, true}},
                                         {gelu_tanh_sqrt_two_over_pi, {0x3f4c422a, true}},
                                         {gelu_tanh_flt_max_x, {0x4154C480, true}},
                                         {gelu_tanh_flt_min_x, {0xC154C480, true}}};

    // tanh(x) constants for four interval approximation
    static const table_t tanh_consts{{tanh_idx_bias, {0x39800000, true}},
                                     {tanh_idx_mask, {0xffc00000, true}},
                                     {tanh_linear_ubound, {0x39ddb3d7, true}},
                                     {tanh_saturation_lbound, {0x41102cb3, true}}};

    // tanh(x) polynomial approximation
    // For each coefficient, there is 32 entries
    static const table_t tanh_polynomial_table{
        // coefficients of degree 0
        {tanh_pol_table, {0x00000000, false}},
        {tanh_pol_table, {0x39bfffff, false}},
        {tanh_pol_table, {0x39ffffff, false}},
        {tanh_pol_table, {0x3a3ffffe, false}},
        {tanh_pol_table, {0x3a7ffffb, false}},
        {tanh_pol_table, {0x3abffff7, false}},
        {tanh_pol_table, {0x3affffeb, false}},
        {tanh_pol_table, {0x3b3fffdc, false}},
        {tanh_pol_table, {0x3b7fffab, false}},
        {tanh_pol_table, {0x3bbfff70, false}},
        {tanh_pol_table, {0x3bfffeab, false}},
        {tanh_pol_table, {0x3c3ffdc0, false}},
        {tanh_pol_table, {0x3c7ffaab, false}},
        {tanh_pol_table, {0x3cbff701, false}},
        {tanh_pol_table, {0x3cffeaad, false}},
        {tanh_pol_table, {0x3d3fdc08, false}},
        {tanh_pol_table, {0x3d7faacd, false}},
        {tanh_pol_table, {0x3dbf7081, false}},
        {tanh_pol_table, {0x3dfeacc9, false}},
        {tanh_pol_table, {0x3e3dc7fd, false}},
        {tanh_pol_table, {0x3e7acbf5, false}},
        {tanh_pol_table, {0x3eb77a9f, false}},
        {tanh_pol_table, {0x3eec9a9f, false}},
        {tanh_pol_table, {0x3f22991f, false}},
        {tanh_pol_table, {0x3f42f7d6, false}},
        {tanh_pol_table, {0x3f67b7cc, false}},
        {tanh_pol_table, {0x3f76ca83, false}},
        {tanh_pol_table, {0x3f7ebbe9, false}},
        {tanh_pol_table, {0x3f7fd40c, false}},
        {tanh_pol_table, {0x3f7fff32, false}},
        {tanh_pol_table, {0x3f7ffffc, false}},
        {tanh_pol_table, {0x3f800000, false}},
        // coefficients of degree 1
        {tanh_pol_table, {0x3f800000, false}},
        {tanh_pol_table, {0x3f800018, false}},
        {tanh_pol_table, {0x3f7fffe8, false}},
        {tanh_pol_table, {0x3f7fffda, false}},
        {tanh_pol_table, {0x3f7fffdc, false}},
        {tanh_pol_table, {0x3f7fffdc, false}},
        {tanh_pol_table, {0x3f7fffac, false}},
        {tanh_pol_table, {0x3f7fff70, false}},
        {tanh_pol_table, {0x3f7ffeec, false}},
        {tanh_pol_table, {0x3f7ffdc0, false}},
        {tanh_pol_table, {0x3f7ffbed, false}},
        {tanh_pol_table, {0x3f7ff704, false}},
        {tanh_pol_table, {0x3f7feff5, false}},
        {tanh_pol_table, {0x3f7fdbca, false}},
        {tanh_pol_table, {0x3f7fbfff, false}},
        {tanh_pol_table, {0x3f7f7041, false}},
        {tanh_pol_table, {0x3f7f009b, false}},
        {tanh_pol_table, {0x3f7dc36c, false}},
        {tanh_pol_table, {0x3f7c0aa8, false}},
        {tanh_pol_table, {0x3f7734b8, false}},
        {tanh_pol_table, {0x3f70a4de, false}},
        {tanh_pol_table, {0x3f5f1fd8, false}},
        {tanh_pol_table, {0x3f495493, false}},
        {tanh_pol_table, {0x3f18b9ec, false}},
        {tanh_pol_table, {0x3ed706cb, false}},
        {tanh_pol_table, {0x3e390b06, false}},
        {tanh_pol_table, {0x3d90b11f, false}},
        {tanh_pol_table, {0x3c21a053, false}},
        {tanh_pol_table, {0x3aaf7fdb, false}},
        {tanh_pol_table, {0x37ccc1a3, false}},
        {tanh_pol_table, {0x355c6733, false}},
        {tanh_pol_table, {0x00000000, false}},
        // coefficients of degree 2
        {tanh_pol_table, {0x00000000, false}},
        {tanh_pol_table, {0xbe4e0ff1, false}},
        {tanh_pol_table, {0x3d25b1b1, false}},
        {tanh_pol_table, {0x3d6b6dab, false}},
        {tanh_pol_table, {0x3c9fb1d5, false}},
        {tanh_pol_table, {0xbabff06f, false}},
        {tanh_pol_table, {0x3c07b3f6, false}},
        {tanh_pol_table, {0xbb3fc1bc, false}},
        {tanh_pol_table, {0x3a9f5921, false}},
        {tanh_pol_table, {0xbbbf06f2, false}},
        {tanh_pol_table, {0xbbb0f402, false}},
        {tanh_pol_table, {0xbc47db9e, false}},
        {tanh_pol_table, {0xbc73d5e7, false}},
        {tanh_pol_table, {0xbca25bda, false}},
        {tanh_pol_table, {0xbcfca780, false}},
        {tanh_pol_table, {0xbd40e07c, false}},
        {tanh_pol_table, {0xbd7dab03, false}},
        {tanh_pol_table, {0xbdbe4a0f, false}},
        {tanh_pol_table, {0xbdfb14a5, false}},
        {tanh_pol_table, {0xbe36cc8d, false}},
        {tanh_pol_table, {0xbe6bd102, false}},
        {tanh_pol_table, {0xbe9fe7c5, false}},
        {tanh_pol_table, {0xbeba0f10, false}},
        {tanh_pol_table, {0xbec206a8, false}},
        {tanh_pol_table, {0xbea3c388, false}},
        {tanh_pol_table, {0xbe277d62, false}},
        {tanh_pol_table, {0xbd8b7960, false}},
        {tanh_pol_table, {0xbc209f49, false}},
        {tanh_pol_table, {0xbaad44ca, false}},
        {tanh_pol_table, {0xb7c6eeac, false}},
        {tanh_pol_table, {0xb663aa41, false}},
        {tanh_pol_table, {0x00000000, false}},
        // coefficients of degree 3
        {tanh_pol_table, {0x00000000, false}},
        {tanh_pol_table, {0x45b3ae96, false}},
        {tanh_pol_table, {0xc414eb20, false}},
        {tanh_pol_table, {0xc450e02e, false}},
        {tanh_pol_table, {0xc3152b4e, false}},
        {tanh_pol_table, {0xbead2f56, false}},
        {tanh_pol_table, {0xc2162e02, false}},
        {tanh_pol_table, {0xbeb4bd5a, false}},
        {tanh_pol_table, {0xc11a59a4, false}},
        {tanh_pol_table, {0xbed2f507, false}},
        {tanh_pol_table, {0xc020d32c, false}},
        {tanh_pol_table, {0x3dd0f506, false}},
        {tanh_pol_table, {0xbf2a75e2, false}},
        {tanh_pol_table, {0xbff950e3, false}},
        {tanh_pol_table, {0xbed47334, false}},
        {tanh_pol_table, {0xbe809b8c, false}},
        {tanh_pol_table, {0xbeb64532, false}},
        {tanh_pol_table, {0xbe961a5b, false}},
        {tanh_pol_table, {0xbe9b63ac, false}},
        {tanh_pol_table, {0xbea0d4b2, false}},
        {tanh_pol_table, {0xbe828a77, false}},
        {tanh_pol_table, {0xbe378612, false}},
        {tanh_pol_table, {0xbdc20908, false}},
        {tanh_pol_table, {0x3d2d3957, false}},
        {tanh_pol_table, {0x3dd46e89, false}},
        {tanh_pol_table, {0x3db3f629, false}},
        {tanh_pol_table, {0x3d2c5e7b, false}},
        {tanh_pol_table, {0x3bd20403, false}},
        {tanh_pol_table, {0x3a59dfae, false}},
        {tanh_pol_table, {0x3770af45, false}},
        {tanh_pol_table, {0x372cc014, false}},
        {tanh_pol_table, {0x00000000, false}},
        // coefficients of degree 4
        {tanh_pol_table, {0x00000000, false}},
        {tanh_pol_table, {0xcc981a1b, false}},
        {tanh_pol_table, {0x4a7edd3d, false}},
        {tanh_pol_table, {0x4ab1007c, false}},
        {tanh_pol_table, {0x48fedd9c, false}},
        {tanh_pol_table, {0x41a557b5, false}},
        {tanh_pol_table, {0x477ee32a, false}},
        {tanh_pol_table, {0x422557f5, false}},
        {tanh_pol_table, {0x45ff3ce4, false}},
        {tanh_pol_table, {0x42a55641, false}},
        {tanh_pol_table, {0x446e0867, false}},
        {tanh_pol_table, {0xc33dc19a, false}},
        {tanh_pol_table, {0x42915214, false}},
        {tanh_pol_table, {0x43af4fad, false}},
        {tanh_pol_table, {0x4110fe88, false}},
        {tanh_pol_table, {0xc1099b75, false}},
        {tanh_pol_table, {0x3fc8a8dc, false}},
        {tanh_pol_table, {0xbfbeaef5, false}},
        {tanh_pol_table, {0xbe365aad, false}},
        {tanh_pol_table, {0x3f4d9652, false}},
        {tanh_pol_table, {0x3ddfa08f, false}},
        {tanh_pol_table, {0x3e34e9b8, false}},
        {tanh_pol_table, {0x3e2d07a6, false}},
        {tanh_pol_table, {0x3dc63567, false}},
        {tanh_pol_table, {0x3cdaeb78, false}},
        {tanh_pol_table, {0xbcd17537, false}},
        {tanh_pol_table, {0xbc92829c, false}},
        {tanh_pol_table, {0xbb43ab99, false}},
        {tanh_pol_table, {0xb9b471dd, false}},
        {tanh_pol_table, {0xb6baad5a, false}},
        {tanh_pol_table, {0xb78bafc7, false}},
        {tanh_pol_table, {0x00000000, false}},
        // coefficients of degree 5
        {tanh_pol_table, {0x00000000, false}},
        {tanh_pol_table, {0x52f688d5, false}},
        {tanh_pol_table, {0xd0505c72, false}},
        {tanh_pol_table, {0xd08f98e3, false}},
        {tanh_pol_table, {0xce505cc9, false}},
        {tanh_pol_table, {0xc7162b8a, false}},
        {tanh_pol_table, {0xcc5061d6, false}},
        {tanh_pol_table, {0xc7162bdf, false}},
        {tanh_pol_table, {0xca50b37f, false}},
        {tanh_pol_table, {0xc7162a3a, false}},
        {tanh_pol_table, {0xc8422086, false}},
        {tanh_pol_table, {0x471a714e, false}},
        {tanh_pol_table, {0xc5ece1f1, false}},
        {tanh_pol_table, {0xc70e3d90, false}},
        {tanh_pol_table, {0xc3eba94a, false}},
        {tanh_pol_table, {0x43e0c424, false}},
        {tanh_pol_table, {0xc21f4552, false}},
        {tanh_pol_table, {0x42217cc8, false}},
        {tanh_pol_table, {0x405e7dc4, false}},
        {tanh_pol_table, {0xc10dd401, false}},
        {tanh_pol_table, {0x3e96b602, false}},
        {tanh_pol_table, {0xbd1a6d2f, false}},
        {tanh_pol_table, {0xbd393883, false}},
        {tanh_pol_table, {0xbd674682, false}},
        {tanh_pol_table, {0xbd310016, false}},
        {tanh_pol_table, {0xb961e269, false}},
        {tanh_pol_table, {0x3ba32495, false}},
        {tanh_pol_table, {0x3a7680d5, false}},
        {tanh_pol_table, {0x38b3173c, false}},
        {tanh_pol_table, {0x35a9deea, false}},
        {tanh_pol_table, {0x375c3f2a, false}},
        {tanh_pol_table, {0x00000000, false}},
        // coefficients of degree 6
        {tanh_pol_table, {0x00000000, false}},
        {tanh_pol_table, {0xd8995ed1, false}},
        {tanh_pol_table, {0x558285ea, false}},
        {tanh_pol_table, {0x55b2cd69, false}},
        {tanh_pol_table, {0x53028625, false}},
        {tanh_pol_table, {0x4bc9991f, false}},
        {tanh_pol_table, {0x5082898a, false}},
        {tanh_pol_table, {0x4b4999b3, false}},
        {tanh_pol_table, {0x4e02c07c, false}},
        {tanh_pol_table, {0x4ac99764, false}},
        {tanh_pol_table, {0x4b72c822, false}},
        {tanh_pol_table, {0xca40c0e1, false}},
        {tanh_pol_table, {0x489413e4, false}},
        {tanh_pol_table, {0x49b12224, false}},
        {tanh_pol_table, {0x46134c4e, false}},
        {tanh_pol_table, {0xc60c2d57, false}},
        {tanh_pol_table, {0x43c83910, false}},
        {tanh_pol_table, {0xc3c872d1, false}},
        {tanh_pol_table, {0xc186bc9e, false}},
        {tanh_pol_table, {0x42325bc3, false}},
        {tanh_pol_table, {0xbf2ffa4a, false}},
        {tanh_pol_table, {0x3d9a203c, false}},
        {tanh_pol_table, {0xbc545a43, false}},
        {tanh_pol_table, {0xbae08fee, false}},
        {tanh_pol_table, {0x3c80225d, false}},
        {tanh_pol_table, {0x3b1fd1df, false}},
        {tanh_pol_table, {0xba36b9d1, false}},
        {tanh_pol_table, {0xb91de544, false}},
        {tanh_pol_table, {0xb71f100f, false}},
        {tanh_pol_table, {0xb408e2ed, false}},
        {tanh_pol_table, {0xb685fec8, false}},
        {tanh_pol_table, {0x00000000, false}},
    };

    auto push_arg_entry_of = [&](const key_t key, const table_entry_val_t val, const bool broadcast) {
      mapped_table_entry_t te{0, val, broadcast};
      entry_map.insert(std::make_pair(key, te));
    };

    auto push_entries_of = [&](const table_t& t) {
      for (auto it = t.begin(); it != t.end(); it++) {
        auto key = it->first;
        auto te = it->second;
        push_arg_entry_of(key, te.val, te.bcast);
      }
    };

    auto set_table_term_offset = [&]() {
      size_t off = 0;
      for (auto it = entry_map.begin(); it != entry_map.end(); it++) {
        auto& te = (*it).second;
        te.off = off;
        off += te.bcast ? 64u : sizeof(table_entry_val_t);
      }
    };

    struct need_t {
      explicit need_t(JBLAS_ELTWISEOP& op) {
        if (op == EXP) exp_ = true;
        if (op == TANH) tanh_ = true;
        if (op == GELU) gelu_ = true;
        if (op == SWISH) swish_ = true;
        if (op == LOW_PRECISION_EXP) low_precision_exp_ = true;
      }
      bool bf16_ = false;
      bool exp_ = false;
      bool tanh_ = false;
      bool gelu_ = false;
      bool low_precision_exp_ = false;
      bool swish_ = false;

      bool bf16() const { return bf16_; }
      bool exp() const { return exp_; }
      bool tanh() const { return tanh_; }
      bool gelu() const { return gelu_; }
      bool low_precision_exp() { return low_precision_exp_; }
      bool swish() const { return swish_; }
    };

    need_t need(elt_op);
    push_entries_of(common_values);
    if (need.exp()) {
      push_entries_of(exp_consts);
      push_entries_of(exp_polynomial);
    }
    if (need.low_precision_exp() || need.swish()) {
      push_entries_of(exp_polynomial);
      push_entries_of(exp_consts);
      push_entries_of(low_precision_exp_consts);
    }
    if (need.tanh() || need.gelu()) {
      push_entries_of(tanh_consts);
      push_entries_of(tanh_polynomial_table);
    }
    if (need.gelu()) push_entries_of(gelu_tanh_const);

    set_table_term_offset();
  }
  void exp_compute_vector_fwd(const Xbyak::Ymm& ymm_src) {
    /* exp code */
    h->vcmpps(ymm_mask, ymm_src, table_val(exp_ln_flt_min_f), _cmp_lt_os);
    h->vminps(ymm_src, ymm_src, table_val(exp_ln_flt_max_f));
    h->vmaxps(ymm_src, ymm_src, table_val(exp_ln_flt_min_f));
    h->vmovups(ymm_aux1, ymm_src);
    h->vmulps(ymm_src, ymm_src, table_val(exp_log2ef));
    h->vaddps(ymm_src, ymm_src, table_val(half));
    h->vroundps(ymm_aux2, ymm_src, _op_floor);

    // keep ymm_src = fx for further computations
    h->vmovups(ymm_src, ymm_aux2);

    // x = x - fx * ln2
    h->vfnmadd231ps(ymm_aux1, ymm_aux2, table_val(ln2f));

    // We do not count 2^n here, because n can reach 128 and 2^128 is not
    // representable by fp32, so to get around this problem, instead of
    // computing 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
    // and 2 are numbers representable in fp32.

    // compute 2^(n-1)
    h->vsubps(ymm_src, ymm_src, table_val(one));
    h->vcvtps2dq(ymm_aux2, ymm_src);
    h->vpaddd(ymm_aux2, ymm_aux2, table_val(exponent_bias));
    h->vpslld(ymm_aux2, ymm_aux2, n_mantissa_bits);

    // use ymm_src as tmp ymm_zero when applying mask
    h->vxorps(ymm_src, ymm_src, ymm_src);

    // set zeroes at those points which were < log(FLT_MIN)
    h->vblendvps(ymm_aux2, ymm_aux2, ymm_src, ymm_mask);

    // compute polynomial
    h->vmovups(ymm_src, table_val(exp_pol, 4));
    h->vfmadd213ps(ymm_src, ymm_aux1, table_val(exp_pol, 3));
    h->vfmadd213ps(ymm_src, ymm_aux1, table_val(exp_pol, 2));
    h->vfmadd213ps(ymm_src, ymm_aux1, table_val(exp_pol, 1));
    h->vfmadd213ps(ymm_src, ymm_aux1, table_val(exp_pol, 0));
    h->vfmadd213ps(ymm_src, ymm_aux1, table_val(one));

    // y = y * 2^n

    h->vmulps(ymm_src, ymm_src, ymm_aux2);
    h->vmulps(ymm_src, ymm_src, table_val(two));
  }
  void exp_compute_vector_fwd(const Xbyak::Zmm& zmm_src) {
    /* exp code */
    h->vcmpps(k_mask, zmm_src, table_val(exp_ln_flt_min_f), _cmp_lt_os);
    h->vminps(zmm_src, zmm_src, table_val(exp_ln_flt_max_f));
    h->vmaxps(zmm_src, zmm_src, table_val(exp_ln_flt_min_f));
    h->vmovups(zmm_aux1, zmm_src);
    h->vmulps(zmm_src, zmm_src, table_val(exp_log2ef));
    h->vaddps(zmm_src, zmm_src, table_val(half));
    h->vrndscaleps(zmm_aux2, zmm_src, _op_floor & 0x3);

    // keep zmm_src = fx for further computations
    h->vmovups(zmm_src, zmm_aux2);

    // x = x - fx * ln2
    h->vfnmadd231ps(zmm_aux1, zmm_aux2, table_val(ln2f));

    // We do not count 2^n here, because n can reach 128 and 2^128 is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
    // and 2 are numbers representable in fp32.

    // compute 2^(n-1)
    h->vsubps(zmm_src, zmm_src, table_val(one));
    h->vcvtps2dq(zmm_aux2, zmm_src);
    h->vpaddd(zmm_aux2, zmm_aux2, table_val(exponent_bias));
    h->vpslld(zmm_aux2, zmm_aux2, n_mantissa_bits);

    // use zmm_src as tmp zmm_zero when applying mask
    h->vxorps(zmm_src, zmm_src, zmm_src);

    // set zeroes at those points which were < log(FLT_MIN)
    h->vblendmps(zmm_aux2 | k_mask, zmm_aux2, zmm_src);

    // compute polynomial
    h->vmovups(zmm_src, table_val(exp_pol, 4));
    h->vfmadd213ps(zmm_src, zmm_aux1, table_val(exp_pol, 3));
    h->vfmadd213ps(zmm_src, zmm_aux1, table_val(exp_pol, 2));
    h->vfmadd213ps(zmm_src, zmm_aux1, table_val(exp_pol, 1));
    h->vfmadd213ps(zmm_src, zmm_aux1, table_val(exp_pol, 0));
    h->vfmadd213ps(zmm_src, zmm_aux1, table_val(one));

    // y = y * 2^n

    h->vmulps(zmm_src, zmm_src, zmm_aux2);
    h->vmulps(zmm_src, zmm_src, table_val(two));
  }
  void low_precision_exp_compute_vector_fwd(const Xbyak::Ymm& ymm_src) {
    // support abs(x)<23
    auto code = [&](Xbyak::CodeGenerator* h, const Ymm& dst, const Ymm& src, const Xbyak::Operand& log2e,
                    const Xbyak::Operand& ln2, const Xbyak::Operand& coeff0, const Xbyak::Operand& coeff1,
                    const Xbyak::Operand& coeff2, const std::array<Ymm, 4>& tmp) {
      h->vmulps(tmp[0], src, log2e);      // x / ln2
      h->vroundps(tmp[0], tmp[0], 0x0A);  // round up
      const auto& z = tmp[0];
      h->vmulps(tmp[1], tmp[0], ln2);
      h->vsubps(tmp[1], src, tmp[1]);  // x mod ln2 (can we use fmsub?)
      h->vmovaps(dst, coeff1);
      h->vfmadd231ps(dst, tmp[1], coeff0);  // dst = f * c0 + c1
      h->vfmadd213ps(dst, tmp[1], coeff2);  // dst = (f * c0 + c1) * f + c2

      const auto& z_sign = tmp[2];
      const auto& z_abs = tmp[3];
      h->vcmpps(z_sign, z, table_val(zero), _cmp_lt_os);
      h->vcvtps2dq(z, z);
      h->vpabsd(z_abs, z);
      h->vmovdqu(tmp[1], table_val(one_epi32));
      h->vpsllvd(z_abs, tmp[1], z_abs);  // 2^z
      h->vcvtdq2ps(z_abs, z_abs);
      h->vrcpps(z, z_abs);
      h->vblendvps(z, z_abs, z, z_sign);
      h->vmulps(dst, dst, z);  // dst = exp(f) * 2^z
    };
    code(h, ymm_src, ymm_src, table_val(exp_log2ef), table_val(ln2f),  //
         table_val(low_precision_exp_const_v0), table_val(low_precision_exp_const_v1),
         table_val(low_precision_exp_const_v2), {ymm_aux1, ymm_aux2, ymm_aux3, ymm_aux4});
  }
  void low_precision_exp_compute_vector_fwd(const Xbyak::Zmm& zmm_src) {
    auto code = [&](Xbyak::CodeGenerator* h, const Zmm& dst, const Zmm& src, const Xbyak::Operand& log2e,
                    const Xbyak::Operand& ln2, const Xbyak::Operand& coeff0, const Xbyak::Operand& coeff1,
                    const Xbyak::Operand& coeff2, const std::array<Zmm, 2>& tmp) {
      h->vmovups(tmp[0], log2e);
      h->vmulps(tmp[0] | h->T_ru_sae, src, tmp[0]);  // round up(x / ln2)
      const auto& z = tmp[0];
      h->vmulps(tmp[1], tmp[0], ln2);
      h->vsubps(tmp[1], src, tmp[1]);  // x mod ln2 (can we use fmsub?)
      h->vmovaps(dst, coeff1);
      h->vfmadd231ps(dst, tmp[1], coeff0);  // dst = f * c0 + c1
      h->vfmadd213ps(dst, tmp[1], coeff2);  // dst = (f * c0 + c1) * f + c2
      h->vscalefps(dst, dst, z);            // dst = exp(f) * 2^z
    };
    code(h, zmm_src, zmm_src, table_val(exp_log2ef), table_val(ln2f),  //
         table_val(low_precision_exp_const_v0), table_val(low_precision_exp_const_v1),
         table_val(low_precision_exp_const_v2), {zmm_aux1, zmm_aux2});
  }
  void swish_compute_vector_fwd(const Xbyak::Ymm& ymm_src, int const_p_offset) {
    h->vbroadcastss(ymm_aux0, h->ptr[reg_rt_const_p + const_p_offset]);
    h->vmulps(ymm_aux0, ymm_aux0, ymm_src);
    exp_compute_vector_fwd(ymm_aux0);
    h->vaddps(ymm_aux0, ymm_aux0, table_val(one));
    h->vrcpps(ymm_aux0, ymm_aux0);
    h->vmulps(ymm_src, ymm_src, ymm_aux0);
  }
  void swish_compute_vector_fwd(const Xbyak::Zmm& zmm_src, int const_p_offset) {
    h->vmovups(zmm_aux0, zmm_src);
    h->vmulps(zmm_aux0, zmm_aux0, h->zword_b[reg_rt_const_p + const_p_offset]);
    low_precision_exp_compute_vector_fwd(zmm_aux0);
    h->vaddps(zmm_aux0, zmm_aux0, table_val(one));
    h->vrcp14ps(zmm_aux0, zmm_aux0);
    h->vmulps(zmm_src, zmm_src, zmm_aux0);
  }
  void tanh_compute_vector_fwd(const Xbyak::Ymm& ymm_src) {
    // register mapping
    Ymm ymm_dst = ymm_aux1, ymm_src_shift = ymm_aux1, ymm_coeff = ymm_aux1, ymm_pol = ymm_aux2, ymm_indices = ymm_aux3,
        ymm_src_original = ymm_aux4, ymm_sign = ymm_aux4;

    const int tanh_n_polynomials = 32;

    // We split the positive domain in 33 intervals:
    // a) [0; linear_ubound]: in this interval tanh(x) = x
    // b) [linear_ubound; 0x1.8p-12]: This interval spans part of a
    //    half binade
    // c) [0x1.8p-12; 0x1.0p-11], ..., [0x1.8p2; 0x1.0p3]:
    //    one interval for each half binade, there are 29 of those
    // d) [0x1.0p3; saturation_ubound]:
    //    This interval spans part of a half binade
    // e) [0x1.205966p3; saturation_ubound]: in this interval, tanh(x) = 1
    // For b-d, we need 31 polynomials and will do a table lookup for those.
    // To simplify the logic, we will also put a) in the table.
    auto coeffs_address = [&](int coeff_off, int off = 0) {
      return table_val(tanh_pol_table, coeff_off * tanh_n_polynomials + off);
    };
    auto gather_coefficient = [&](Ymm vmm_coeff, int coeff_idx, Ymm vmm_pol_idx) {
      Ymm ymm_coeff(vmm_coeff.getIdx());
      Ymm ymm_pol_idx(vmm_pol_idx.getIdx());
      Xbyak::Address idx_addr =
          h->ptr[p_table + table_off(tanh_pol_table, coeff_idx * tanh_n_polynomials) + ymm_pol_idx * sizeof(float)];
      h->vcmpps(ymm_mask, ymm_mask, ymm_mask, _cmp_eq_oq);
      h->vgatherdps(vmm_coeff, idx_addr, ymm_mask);
    };

    // because tanh(x) = -tanh(-x), we extract sign to make x positive
    // and reapply sign at the end
    h->vmovups(ymm_src_original, ymm_src);
    h->vandps(ymm_src, ymm_src, table_val(positive_mask));

    // We compute the indices for the table lookup
    h->vmovups(ymm_indices, ymm_src);
    h->vpsubd(ymm_indices, ymm_indices, table_val(tanh_idx_bias));
    h->vandps(ymm_indices, ymm_indices, table_val(tanh_idx_mask));
    h->vpsrld(ymm_indices, ymm_indices, 22);

    // we do the argument reduction
    h->vmovups(ymm_src_shift, ymm_src);
    h->vandps(ymm_src_shift, ymm_src_shift, table_val(tanh_idx_mask));
    h->vsubps(ymm_src, ymm_src, ymm_src_shift);

    // we gather and evaluate the polynonials
    gather_coefficient(ymm_pol, 6, ymm_indices);
    for (int deg = 5; deg >= 0; --deg) {
      gather_coefficient(ymm_coeff, deg, ymm_indices);
      h->vfmadd213ps(ymm_pol, ymm_src, ymm_coeff);
    }

    // we restore src with cleared sign, and keep sign
    h->vmovups(ymm_src, ymm_src_original);
    h->vandps(ymm_sign, ymm_sign, table_val(sign_mask));
    h->vandps(ymm_src, ymm_src, table_val(positive_mask));

    // Now we blend the results
    // [saturation_ubound; +inf[ : we return +/- 1
    h->vmovups(ymm_dst, table_val(one));
    // [linear_ubound; saturation_lbound] : we return +/- P(x)
    h->vmovups(ymm_mask, table_val(tanh_saturation_lbound));
    h->vcmpps(ymm_mask, ymm_mask, ymm_src, _cmp_nle_us);
    h->vblendvps(ymm_dst, ymm_dst, ymm_pol, ymm_mask);
    // [0; linear_ubound]  : we return x
    h->vmovups(ymm_mask, table_val(tanh_linear_ubound));
    h->vcmpps(ymm_mask, ymm_mask, ymm_src, _cmp_nle_us);
    h->vblendvps(ymm_dst, ymm_dst, ymm_src, ymm_mask);

    // We reapply the sign and return
    h->vxorps(ymm_dst, ymm_dst, ymm_sign);
    h->vmovups(ymm_src, ymm_dst);
  }
  void tanh_compute_vector_fwd(const Xbyak::Zmm& zmm_src) {
    // register mapping
    Zmm zmm_dst = zmm_aux1, zmm_src_shift = zmm_aux1, zmm_coeff = zmm_aux1, zmm_pol = zmm_aux2, zmm_indices = zmm_aux3,
        zmm_src_original = zmm_aux4, zmm_sign = zmm_aux4;

    const int tanh_n_polynomials = 32;

    // We split the positive domain in 33 intervals:
    // a) [0; linear_ubound]: in this interval tanh(x) = x
    // b) [linear_ubound; 0x1.8p-12]: This interval spans part of a
    //    half binade
    // c) [0x1.8p-12; 0x1.0p-11], ..., [0x1.8p2; 0x1.0p3]:
    //    one interval for each half binade, there are 29 of those
    // d) [0x1.0p3; saturation_ubound]:
    //    This interval spans part of a half binade
    // e) [0x1.205966p3; saturation_ubound]: in this interval, tanh(x) = 1
    // For b-d, we need 31 polynomials and will do a table lookup for those.
    // To simplify the logic, we will also put a) in the table.
    auto coeffs_address = [&](int coeff_off, int off = 0) {
      return table_val(tanh_pol_table, (size_t)coeff_off * tanh_n_polynomials + off);
    };
    auto gather_coefficient = [&](Zmm vmm_coeff, int coeff_idx, Zmm vmm_pol_idx) {
      Zmm zmm_coeff(vmm_coeff.getIdx());
      Zmm zmm_pol_idx(vmm_pol_idx.getIdx());
      h->vmovups(zmm_coeff, coeffs_address(coeff_idx, 0));
      h->vpermt2ps(zmm_coeff, zmm_pol_idx, coeffs_address(coeff_idx, 16));
    };

    // because tanh(x) = -tanh(-x), we extract sign to make x positive
    // and reapply sign at the end
    h->vmovups(zmm_src_original, zmm_src);
    h->vpandd(zmm_src, zmm_src, table_val(positive_mask));

    // We compute the indices for the table lookup
    h->vmovups(zmm_indices, zmm_src);
    h->vpsubd(zmm_indices, zmm_indices, table_val(tanh_idx_bias));
    h->vpandd(zmm_indices, zmm_indices, table_val(tanh_idx_mask));
    h->vpsrld(zmm_indices, zmm_indices, 22);

    // we do the argument reduction
    h->vmovups(zmm_src_shift, zmm_src);
    h->vpandd(zmm_src_shift, zmm_src_shift, table_val(tanh_idx_mask));
    h->vsubps(zmm_src, zmm_src, zmm_src_shift);

    // we gather and evaluate the polynonials
    gather_coefficient(zmm_pol, 6, zmm_indices);
    for (int deg = 5; deg >= 0; --deg) {
      gather_coefficient(zmm_coeff, deg, zmm_indices);
      h->vfmadd213ps(zmm_pol, zmm_src, zmm_coeff);
    }

    // we restore src with cleared sign, and keep sign
    h->vmovups(zmm_src, zmm_src_original);
    h->vpandd(zmm_sign, zmm_sign, table_val(sign_mask));
    h->vpandd(zmm_src, zmm_src, table_val(positive_mask));

    // Now we blend the results
    // [saturation_ubound; +inf[ : we return +/- 1
    h->vmovups(zmm_dst, table_val(one));
    // [linear_ubound; saturation_lbound] : we return +/- P(x)
    h->vmovups(zmm_mask, table_val(tanh_saturation_lbound));
    h->vcmpps(k_mask, zmm_mask, zmm_src, _cmp_nle_us);
    h->vblendmps(zmm_dst | k_mask, zmm_dst, zmm_pol);
    // [0; linear_ubound]  : we return x
    h->vmovups(zmm_mask, table_val(tanh_linear_ubound));
    h->vcmpps(k_mask, zmm_mask, zmm_src, _cmp_nle_us);
    h->vblendmps(zmm_dst | k_mask, zmm_dst, zmm_src);

    // We reapply the sign and return
    h->vpxord(zmm_dst, zmm_dst, zmm_sign);
    h->vmovups(zmm_src, zmm_dst);
  }
  void gelu_compute_vector_fwd(const Xbyak::Ymm& ymm_src) {
    h->vmovups(ymm_aux0, ymm_src);
    // compute G(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x * x)
    h->vmulps(ymm_src, ymm_src, ymm_src);
    h->vmovups(ymm_aux1, table_val(gelu_tanh_fitting_const));
    h->vfmadd213ps(ymm_src, ymm_aux1, table_val(one));
    h->vmulps(ymm_src, ymm_src, ymm_aux0);
    h->vmulps(ymm_src, ymm_src, table_val(gelu_tanh_sqrt_two_over_pi));

    // compute tanh(G(x))
    tanh_compute_vector_fwd(ymm_src);

    // compute 0.5 * x * (1 + tanh(G(x)))
    h->vaddps(ymm_src, ymm_src, table_val(one));
    h->vmulps(ymm_src, ymm_src, table_val(half));
    h->vmulps(ymm_src, ymm_src, ymm_aux0);
  }
  void gelu_compute_vector_fwd(const Xbyak::Zmm& zmm_src) {
    h->vmovups(zmm_aux0, zmm_src);
    // compute G(x) = sqrt_root_two_over_pi * x * (1 + fitting_const * x * x)
    h->vmulps(zmm_src, zmm_src, zmm_src);
    h->vmovups(zmm_aux1, table_val(gelu_tanh_fitting_const));
    h->vfmadd213ps(zmm_src, zmm_aux1, table_val(one));
    h->vmulps(zmm_src, zmm_src, zmm_aux0);
    h->vmulps(zmm_src, zmm_src, table_val(gelu_tanh_sqrt_two_over_pi));

    // compute tanh(G(x))
    tanh_compute_vector_fwd(zmm_src);

    // compute 0.5 * x * (1 + tanh(G(x)))
    h->vaddps(zmm_src, zmm_src, table_val(one));
    h->vmulps(zmm_src, zmm_src, table_val(half));
    h->vmulps(zmm_src, zmm_src, zmm_aux0);
  }
  void relu_compute_vector_fwd(const Xbyak::Zmm& zmm_src, int const_p_offset) {
    h->vmovups(zmm_aux1, zmm_src);
    h->vcmpps(k_mask, zmm_src, table_val(zero), _cmp_nle_us);
    h->vmulps(zmm_src, zmm_src, h->zword_b[reg_rt_const_p + const_p_offset]);
    h->vblendmps(zmm_src | k_mask, zmm_src, zmm_aux1);
  }
  void linear_compute_vector_fwd(const Xbyak::Zmm& zmm_src, int const_p_offset) {
    h->vbroadcastss(zmm_aux0, h->dword[reg_rt_const_p + const_p_offset]);
    h->vfmadd213ps(zmm_src, zmm_aux0, h->zword_b[reg_rt_const_p + const_p_offset + 1 * sizeof(float)]);
  }
  void load_table_addr() { h->mov(p_table, l_table); }
  void assign_zmm(const std::set<int>& used_zmm_idx, Zmm* zmm) {
    constexpr int max_zmm_idx = 32;
    for (int idx = 0; idx < max_zmm_idx; idx++) {
      if (used_zmm_idx.count(idx) == 0 && assign_vmm_idx.count(idx) == 0) {
        *zmm = Zmm(idx);
        assign_vmm_idx.insert(idx);
        break;
      }
    }
  }
  void assign_ymm(const std::set<int>& used_ymm_idx, Ymm* ymm) {
    constexpr int max_ymm_idx = 16;
    for (int idx = 0; idx < max_ymm_idx; idx++) {
      if (used_ymm_idx.count(idx) == 0 && assign_vmm_idx.count(idx) == 0) {
        *ymm = Ymm(idx);
        assign_vmm_idx.insert(idx);
        break;
      }
    }
  }

 private:
  JBLAS_ELTWISEOP elt_op;
  Xbyak::CodeGenerator* h = nullptr;

  /*labels*/
  Xbyak::Label l_table;

  /*register for fwd*/
  Xbyak::Reg64 p_table;
  Xbyak::Reg64 reg_rt_const_p;
  std::set<int> assign_vmm_idx;  // use for zmm (in avx512) or ymm (in avx2)
  Zmm zmm_mask, zmm_aux0, zmm_aux1, zmm_aux2, zmm_aux3, zmm_aux4;
  Ymm ymm_mask, ymm_aux0, ymm_aux1, ymm_aux2, ymm_aux3, ymm_aux4;
  Xbyak::Opmask k_mask;
  static constexpr int n_mantissa_bits = 23;

  enum {
    _cmp_eq_oq = 0u,
    _cmp_lt_os = 1u,
    _cmp_le_os = 2u,
    _cmp_neq_uq = 4u,
    _cmp_nlt_us = 5u,
    _cmp_nle_us = 6u,

    _op_floor = 1u,
    _op_mxcsr = 4u,
  };

  enum key_t {
    zero = 0,                             // 0.f
    half,                                 // 0.5f
    one,                                  // 1.f  or  mask for exponent bits
    two,                                  // 2.f
    three,                                // 3.f
    six,                                  // 6.f
    minus_one,                            // -1.f  or  changes sign to opposite
    minus_two,                            // -2.f
    minus_three,                          // -3.f
    ln2f,                                 // 0.69314718f
    one_epi32,                            // 1 in int32
    positive_mask,                        // changes sign to positive
    sign_mask,                            // gets sign value
    exponent_bias,                        // (127 = 2^7 - 1), gets exponent bits
    exp_log2ef,                           // 1.44269502f - formula-based for approx
    exp_ln_flt_max_f,                     // logf(FLT_MAX) - max normal value
    exp_ln_flt_min_f,                     // logf(FLT_MIN) - min normal value
    exp_pol,                              // see correspondent table for float values
    gelu_tanh_fitting_const,              // 0.044715f
    gelu_tanh_fitting_const_times_three,  // 0.134145f
    gelu_tanh_sqrt_two_over_pi,           // sqrtf(2.f/pi) = 0.797884f
    gelu_tanh_flt_max_x,
    gelu_tanh_flt_min_x,
    tanh_idx_bias,
    tanh_idx_mask,
    tanh_linear_ubound,
    tanh_saturation_lbound,
    tanh_pol_table,
    low_precision_exp_const_v0,
    low_precision_exp_const_v1,
    low_precision_exp_const_v2,
    undef_key,
  };

  size_t table_off(key_t key, size_t key_off_val_shift = 0) {
    const auto it = entry_map.find(key);
    assert(it != entry_map.end());  // "key is not in entry_map"
    const auto& te = (*it).second;
    const auto scale = te.bcast ? 64u : sizeof(table_entry_val_t);
    return te.off + key_off_val_shift * scale;
  }
  Xbyak::Address table_val(key_t key, size_t key_off_val_shift = 0) {
    auto off = table_off(key, key_off_val_shift);
    return h->ptr[p_table + off];
  }
  using table_entry_val_t = uint32_t;
  using table_entry_offset_t = size_t;  // offsets are in bytes wrt p_table
  using table_entry_bcast_t = bool;

  struct table_entry_t {
    table_entry_val_t val;
    table_entry_bcast_t bcast;
  };
  struct mapped_table_entry_t {
    table_entry_offset_t off;
    table_entry_val_t val;
    table_entry_bcast_t bcast;
  };
  using table_t = std::multimap<key_t, table_entry_t>;
  using mapped_table_t = std::multimap<key_t, mapped_table_entry_t>;
  mapped_table_t entry_map = {};
};
}  // namespace jit_injector
}  // namespace kernel
}  // namespace jblas
