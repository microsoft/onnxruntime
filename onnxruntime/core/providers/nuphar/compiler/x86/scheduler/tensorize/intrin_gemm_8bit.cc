// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/providers/nuphar/common/nuphar_settings.h"
#include "core/providers/nuphar/compiler/x86/scheduler/tensorize/intrin_gemm_8bit.h"
#include "core/providers/nuphar/compiler/x86/scheduler/tensorize/tensorize_utilities.h"
#include <tvm/buffer.h>
#include <tvm/codegen.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>

namespace onnxruntime {
namespace nuphar {

TensorizeIntGemm8bit::TensorizeIntGemm8bit(const std::string& name, const std::vector<int32_t>& vshape, const std::string& target)
    : TensorizeBase(name, "TensorizeIntGemm8bit_Parameter", vshape), tensorize_target_(target) {}

void TensorizeIntGemm8bit::InsertTensorizeDimInfo(std::string name, TensorizeDimMeta dim_meta) {
  tensorize_dims_.emplace(name, dim_meta);
}

tvm::Expr TensorizeIntGemm8bit::CreatePredicateMask(int tail_size) {
  int mask_lanes;
  if (tensorize_target_ == "avx512-skylake") {
    mask_lanes = 16;
  } else if (tensorize_target_ == "avx2") {
    mask_lanes = 8;
  } else if (tensorize_target_ == "avx") {
    mask_lanes = 4;
  } else {
    ORT_NOT_IMPLEMENTED("Tensorization only support avx2/avx512-skylake currently!");
  }

  auto u1x1 = HalideIR::UInt(1, 1);
  auto u1v = tensorize_targets_meta_.at(tensorize_target_).u1v;

  if (tail_size < mask_lanes / 2) {
    tvm::Expr mask = tvm::make_const(u1v, 0);
    for (int i = 0; i < tail_size; i++) {
      mask = tvm_codegen::InsertElement(mask, tvm::make_const(u1x1, 1), i);
    }
    return mask;
  } else {
    tvm::Expr mask = tvm::make_const(u1v, 1);
    for (int i = tail_size; i < mask_lanes; i++) {
      mask = tvm_codegen::InsertElement(mask, tvm::make_const(u1x1, 0), i);
    }
    return mask;
  }
}

tvm::Expr TensorizeIntGemm8bit::ExpandScalarUInt8(tvm::Expr& v) {
  auto re_uint32 = tvm_codegen::Reinterpret(HalideIR::UInt(32), v);
  auto i32v = tensorize_targets_meta_.at(tensorize_target_).i32v;
  auto vec_uint32 = tvm::cast(i32v, re_uint32);
  auto u8v = tensorize_targets_meta_.at(tensorize_target_).u8v;
  auto vec_uint8 = tvm_codegen::Reinterpret(u8v, vec_uint32);
  return vec_uint8;
}

tvm::Buffer CreateTensorBuffer(tvm::Tensor tensor, const std::string& name) {
  tvm::Buffer buffer = tvm::BufferNode::make(
      tvm::Var(name, tvm::Handle()),
      tensor->dtype,
      tensor->shape,
      /*strides*/ {tvm::Var("stride_" + name), 1},
      tvm::Var("offset_" + name),
      "buffer_" + name,
      "",
      0,
      /*offset_factor*/ 1);

  return buffer;
}

void TensorizeIntGemm8bit::TensorizeReduceKernel(std::vector<tvm::Stmt>& inits, std::vector<tvm::Stmt>& bodys, std::vector<tvm::Stmt>& updates,
                                                 tvm::Buffer& a_buf, tvm::Buffer& b_buf, tvm::Buffer& c_buf,
                                                 int inner_m, int inner_n) {
  // tensorize dim meta
  auto tensorize_dim_m = tensorize_dims_.at("m");
  auto tensorize_dim_n = tensorize_dims_.at("n");
  auto tensorize_dim_k = tensorize_dims_.at("k");

  // vector types
  auto u8x4 = tensorize_targets_meta_.at(tensorize_target_).u8x4;
  auto u1v = tensorize_targets_meta_.at(tensorize_target_).u1v;
  auto i8v = tensorize_targets_meta_.at(tensorize_target_).i8v;
  auto i32v = tensorize_targets_meta_.at(tensorize_target_).i32v;
  auto i16v = tensorize_targets_meta_.at(tensorize_target_).i16v;

  // vector constant
  auto _0_i32v = tvm::make_const(i32v, 0);
  auto _1_i16v = tvm::make_const(i16v, 1);

  tvm::Expr symbo_cond = (tensorize_dim_m.dim_iter * tensorize_dim_m.tile_size + inner_m < tensorize_dim_m.dim_size);
  tvm::Expr pred_none = (tensorize_dim_n.dim_iter * tensorize_dim_n.tile_size + inner_n > tensorize_dim_n.dim_size);
  tvm::Expr pred_full = (tensorize_dim_n.dim_iter * tensorize_dim_n.tile_size + inner_n + tensorize_dim_n.layout_size <= tensorize_dim_n.dim_size);

  tvm::Expr c_i32v = _0_i32v;
  tvm::Expr c_i32v_pred = _0_i32v;

  for (int inner_k = 0; inner_k < tensorize_dim_k.tile_size / tensorize_dim_k.layout_size; inner_k++) {
    // buffer a regular load
    auto a_u8x4 = a_buf.vload({inner_m, inner_k * tensorize_dim_k.layout_size}, u8x4);
    auto a_u8v = ExpandScalarUInt8(a_u8x4);
    // buffer b regular load
    auto b_i8v = b_buf.vload({inner_n + inner_k % tensorize_dim_n.layout_size,
                              (inner_k / tensorize_dim_n.layout_size) * (tensorize_dim_k.layout_size * tensorize_dim_n.layout_size) + tensorize_dim_k.load_offset},
                             i8v);
    // buffer b load with predicate condition
    auto b_i8v_pred = tvm::ir::Select::make(pred_none, tvm::make_const(i8v, 0), b_i8v);

    // maddubs 8bit to 16bit
    std::string vpmaddubsw = tensorize_targets_meta_.at(tensorize_target_).vpmaddubsw;
    auto c_i16v = tvm_codegen::LLVMIntrinsic(i16v, vpmaddubsw, {a_u8v, b_i8v});
    auto c_i16v_pred = tvm_codegen::LLVMIntrinsic(i16v, vpmaddubsw, {a_u8v, b_i8v_pred});

    // madd 16bit to 32bit
    std::string vpmaddwd = tensorize_targets_meta_.at(tensorize_target_).vpmaddwd;
    c_i32v += tvm_codegen::LLVMIntrinsic(i32v, vpmaddwd, {c_i16v, _1_i16v});
    c_i32v_pred += tvm_codegen::LLVMIntrinsic(i32v, vpmaddwd, {c_i16v_pred, _1_i16v});
  }

  // conditions for predicate store
  auto _0_u1v = tvm::make_const(u1v, 0);
  auto _1_u1v = tvm::make_const(u1v, 1);
  auto mask = CreatePredicateMask(tensorize_dim_n.tail_size);
  tvm::Expr pred_mask = tvm::ir::Select::make(pred_full, _1_u1v, tvm::ir::Select::make(pred_none, _0_u1v, mask));

  // statements hold generated code
  tvm::Stmt init, body, update;

  auto init_regular = c_buf.vstore({inner_m, inner_n}, _0_i32v);
  auto body_regular = c_buf.vstore({inner_m, inner_n}, c_i32v);
  auto update_regular = c_buf.vstore({inner_m, inner_n}, c_i32v + c_buf.vload({inner_m, inner_n}, i32v));

  auto init_pred = c_buf.pvstore({inner_m, inner_n}, _0_i32v, pred_mask);
  auto body_pred = c_buf.pvstore({inner_m, inner_n}, c_i32v_pred, pred_mask);
  auto update_pred = c_buf.pvstore({inner_m, inner_n}, c_i32v_pred + c_buf.vload({inner_m, inner_n}, i32v), pred_mask);

  auto init_symbo = tvm::ir::IfThenElse::make(symbo_cond, init_regular);
  auto body_symbo = tvm::ir::IfThenElse::make(symbo_cond, body_regular);
  auto update_symbo = tvm::ir::IfThenElse::make(symbo_cond, update_regular);

  auto init_pred_symbo = tvm::ir::IfThenElse::make(symbo_cond, init_pred);
  auto body_pred_symbo = tvm::ir::IfThenElse::make(symbo_cond, body_pred);
  auto update_pred_symbo = tvm::ir::IfThenElse::make(symbo_cond, update_pred);

  if (codegen::CodeGenSettings::Instance().HasOption(kNupharTensorize_IGEMM_Split_Last_Tile)) {
    if (tensorize_dim_n.has_tail) {
      if (tensorize_dim_m.has_tail) {
        init_pred = tvm::ir::IfThenElse::make(tensorize_dim_m.tail_cond, init_pred_symbo, init_pred);
        body_pred = tvm::ir::IfThenElse::make(tensorize_dim_m.tail_cond, body_pred_symbo, body_pred);
        update_pred = tvm::ir::IfThenElse::make(tensorize_dim_m.tail_cond, update_pred_symbo, update_pred);

        init_regular = tvm::ir::IfThenElse::make(tensorize_dim_m.tail_cond, init_symbo, init_regular);
        body_regular = tvm::ir::IfThenElse::make(tensorize_dim_m.tail_cond, body_symbo, body_regular);
        update_regular = tvm::ir::IfThenElse::make(tensorize_dim_m.tail_cond, update_symbo, update_regular);
      }
      init = tvm::ir::IfThenElse::make(tensorize_dim_n.tail_cond, init_pred, init_regular);
      body = tvm::ir::IfThenElse::make(tensorize_dim_n.tail_cond, body_pred, body_regular);
      update = tvm::ir::IfThenElse::make(tensorize_dim_n.tail_cond, update_pred, update_regular);
    } else {
      if (tensorize_dim_m.has_tail) {
        init_regular = tvm::ir::IfThenElse::make(tensorize_dim_m.tail_cond, init_symbo, init_regular);
        body_regular = tvm::ir::IfThenElse::make(tensorize_dim_m.tail_cond, body_symbo, body_regular);
        update_regular = tvm::ir::IfThenElse::make(tensorize_dim_m.tail_cond, update_symbo, update_regular);
      }
      init = init_regular;
      body = body_regular;
      update = update_regular;
    }
  } else {
    if (tensorize_dim_n.has_tail) {
      init = tvm::ir::IfThenElse::make(tensorize_dim_n.tail_cond, init_pred_symbo, init_symbo);
      body = tvm::ir::IfThenElse::make(tensorize_dim_n.tail_cond, body_pred_symbo, body_symbo);
      update = tvm::ir::IfThenElse::make(tensorize_dim_n.tail_cond, update_pred_symbo, update_symbo);
    } else {
      init = tvm::ir::IfThenElse::make(symbo_cond, init_regular);
      body = tvm::ir::IfThenElse::make(symbo_cond, body_regular);
      update = tvm::ir::IfThenElse::make(symbo_cond, update_regular);
    }
  }

  inits.push_back(init);
  bodys.push_back(body);
  updates.push_back(update);
}

tvm::TensorIntrin TensorizeIntGemm8bit::CreateTensorIntrin() {
  auto tensorize_dim_m = tensorize_dims_.at("m");
  auto tensorize_dim_n = tensorize_dims_.at("n");
  auto tensorize_dim_k = tensorize_dims_.at("k");

  tvm::Expr m(tensorize_dim_m.tile_size);
  tvm::Expr n(tensorize_dim_n.tile_size);
  tvm::Expr l(tensorize_dim_k.tile_size);

  auto a = tvm::placeholder({m, l}, HalideIR::UInt(8), "placeholder_a");
  auto b = tvm::placeholder({n, l}, HalideIR::Int(8), "placeholder_b");
  auto k = tvm::reduce_axis({0, l}, "k");

  auto c = tvm::compute(
      {m, n},
      [&](tvm::Var i, tvm::Var j) {
        return tvm::sum(tvm::cast(HalideIR::Int(32), a(i, k)) * tvm::cast(HalideIR::Int(32), b(j, k)), {k});
      },
      "tensor_c");

  auto a_buf = CreateTensorBuffer(a, "a");
  auto b_buf = CreateTensorBuffer(b, "b");
  auto c_buf = CreateTensorBuffer(c, "c");

  // prepare tensorize targets meta
  if (tensorize_target_ == "avx512-skylake") {
    TensorizeTargetInfo avx512_info(HalideIR::UInt(1, 16), HalideIR::UInt(8, 64),
                                    HalideIR::Int(8, 64), HalideIR::Int(16, 32), HalideIR::Int(32, 16),
                                    "llvm.x86.avx512.pmaddubs.w.512", "llvm.x86.avx512.pmaddw.d.512");
    tensorize_targets_meta_.emplace(tensorize_target_, avx512_info);
  } else if (tensorize_target_ == "avx2") {
    TensorizeTargetInfo avx2_info(HalideIR::UInt(1, 8), HalideIR::UInt(8, 32),
                                  HalideIR::Int(8, 32), HalideIR::Int(16, 16), HalideIR::Int(32, 8),
                                  "llvm.x86.avx2.pmadd.ub.sw", "llvm.x86.avx2.pmadd.wd");
    tensorize_targets_meta_.emplace(tensorize_target_, avx2_info);
  } else if (tensorize_target_ == "avx") {
    TensorizeTargetInfo avx_info(HalideIR::UInt(1, 4), HalideIR::UInt(8, 16),
                                 HalideIR::Int(8, 16), HalideIR::Int(16, 8), HalideIR::Int(32, 4),
                                 "llvm.x86.ssse3.pmadd.ub.sw.128", "llvm.x86.sse2.pmadd.wd");

    tensorize_targets_meta_.emplace(tensorize_target_, avx_info);
  } else {
    ORT_NOT_IMPLEMENTED("Tensorization only support avx2/avx512-skylake currently!");
  }

  // generated tvm statments
  std::vector<tvm::Stmt> inits, bodys, updates;

  codegen::CodeGenSettings& settings = codegen::CodeGenSettings::Instance();
  if (settings.HasOption(kNupharTensorize_IGEMM_Permute) &&
      (settings.OptionMatches(kNupharTensorize_IGEMM_Permute, kNupharTensorize_IGEMM_Permute_All) ||
       settings.OptionMatches(kNupharTensorize_IGEMM_Permute, kNupharTensorize_IGEMM_Permute_Inner))) {
    for (int inner_m = 0; inner_m < tensorize_dim_m.tile_size; inner_m++) {
      for (int inner_n = 0; inner_n < tensorize_dim_n.tile_size; inner_n += tensorize_dim_n.layout_size) {
        TensorizeReduceKernel(inits, bodys, updates, a_buf, b_buf, c_buf, inner_m, inner_n);
      }
    }
  } else {
    for (int inner_n = 0; inner_n < tensorize_dim_n.tile_size; inner_n += tensorize_dim_n.layout_size) {
      for (int inner_m = 0; inner_m < tensorize_dim_m.tile_size; inner_m++) {
        TensorizeReduceKernel(inits, bodys, updates, a_buf, b_buf, c_buf, inner_m, inner_n);
      }
    }
  }
  tvm::Stmt init = tvm_codegen::MergeStmts(inits);
  tvm::Stmt body = tvm_codegen::MergeStmts(bodys);
  tvm::Stmt update = tvm_codegen::MergeStmts(updates);

  return tvm::TensorIntrinNode::make(
      "intrin_gemm_8bit",
      c->op,
      {a, b},
      {a_buf, b_buf, c_buf},
      body, init, update);
}
}  // namespace nuphar
}  // namespace onnxruntime
