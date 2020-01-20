// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/nuphar/compiler/x86/scheduler/tensorize/tensorize_base.h"

namespace onnxruntime {
namespace nuphar {

struct TensorizeDimMeta {
  tvm::Expr dim_iter;
  tvm::Expr dim_size;

  int tile_size;
  int layout_size;

  bool has_tail;
  int tail_size;
  tvm::Expr tail_cond;
  tvm::Expr load_offset;

  TensorizeDimMeta(tvm::Expr dim_iter, tvm::Expr dim_size, int tile_size, int layout_size, bool has_tail, int tail_size, tvm::Expr tail_cond)
      : dim_iter(dim_iter), dim_size(dim_size), tile_size(tile_size), layout_size(layout_size), has_tail(has_tail), tail_size(tail_size), tail_cond(tail_cond) {}

  TensorizeDimMeta(tvm::Expr dim_iter, tvm::Expr dim_size, int tile_size, bool has_tail, tvm::Expr tail_cond)
      : dim_iter(dim_iter), dim_size(dim_size), tile_size(tile_size), has_tail(has_tail), tail_cond(tail_cond) {}

  TensorizeDimMeta(tvm::Expr dim_iter, tvm::Expr dim_size, int tile_size, int layout_size, tvm::Expr load_offset)
      : dim_iter(dim_iter), dim_size(dim_size), tile_size(tile_size), layout_size(layout_size), load_offset(load_offset) {}

  int64_t DimSizeValue() {
    const int64_t* p_dim_size = tvm::as_const_int(dim_size);
    if (p_dim_size != nullptr) {
      return *p_dim_size;
    }
    return 0;
  }
};

struct TensorizeTargetInfo {
  // type of one uint32
  tvm::Type u8x4 = HalideIR::UInt(8, 4);

  // various vector type
  tvm::Type u1v;
  tvm::Type u8v;
  tvm::Type i8v;
  tvm::Type i16v;
  tvm::Type i32v;

  // llvm intrisics
  std::string vpmaddubsw;
  std::string vpmaddwd;

  TensorizeTargetInfo(tvm::Type u1, tvm::Type u8,
                      tvm::Type i8, tvm::Type i16, tvm::Type i32,
                      std::string maddubsw, std::string maddwd)
      : u1v(u1), u8v(u8), i8v(i8), i16v(i16), i32v(i32), vpmaddubsw(maddubsw), vpmaddwd(maddwd) {}
};

class TensorizeIntGemm8bit : public tvm_codegen::TensorizeBase {
 public:
  TensorizeIntGemm8bit(const std::string& name, const std::vector<int32_t>& vshape, const std::string& target);
  virtual ~TensorizeIntGemm8bit() = default;

  tvm::TensorIntrin CreateTensorIntrin() override;

  tvm::Expr ExpandScalarUInt8(HalideIR::Expr& v);
  tvm::Expr CreatePredicateMask(int tail_size);
  void InsertTensorizeDimInfo(std::string name, TensorizeDimMeta dim_meta);

  void TensorizeReduceKernel(std::vector<tvm::Stmt>& inits,
                             std::vector<tvm::Stmt>& bodys,
                             std::vector<tvm::Stmt>& updates,
                             tvm::Buffer& a_buf,
                             tvm::Buffer& b_buf,
                             tvm::Buffer& c_buf,
                             int inner_m,
                             int inner_n);

 protected:
  // tensorization target string: avx2/avx512-skylake/avx512-vnni
  std::string tensorize_target_;
  // tensorization target meta: <target_str, target_info>
  std::map<std::string, TensorizeTargetInfo> tensorize_targets_meta_;
  // tensorization dimension meta: <dim_str, dim_meta>
  std::map<std::string, TensorizeDimMeta> tensorize_dims_;
};

}  // namespace nuphar
}  // namespace onnxruntime
