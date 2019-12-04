// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/scheduler/tensorize/intrin_gemv_8bit.h"
#include "core/providers/nuphar/compiler/x86/scheduler/tensorize/tensorize_utilities.h"
#include <tvm/buffer.h>
#include <tvm/codegen.h>
#include <tvm/ir.h>

namespace onnxruntime {
namespace nuphar {

TensorizeIntGemv8bit::TensorizeIntGemv8bit(const std::string& name, const std::vector<int32_t>& vshape)
    : TensorizeBase(name, "TensorizeIntGemv8bit_Parameter", {vshape[0], vshape[1]}) {
}

tvm::TensorIntrin TensorizeIntGemv8bit::CreateTensorIntrin() {
  tvm::Expr m(shape_[0]);
  tvm::Expr l(shape_[1]);

  auto a = tvm::placeholder({l}, HalideIR::UInt(8));
  auto b = tvm::placeholder({m, l}, HalideIR::Int(8));
  auto k = tvm::reduce_axis({0, l});

  auto c = tvm::compute({m}, [&](tvm::Var i) {
    return tvm::sum(tvm::cast(HalideIR::Int(32), a(k)) * tvm::cast(HalideIR::Int(32), b(i, k)), {k});
  });

  auto a_buf = tvm::BufferNode::make(
      tvm::Var("a", tvm::Handle()),
      a->dtype,
      a->shape,
      /*strides*/ {1},
      tvm::Var("a_offset"),
      "a",
      "",
      0,
      /*offset_factor*/ 1);

  auto b_buf = tvm::BufferNode::make(
      tvm::Var("b", tvm::Handle()),
      b->dtype,
      b->shape,
      /*strides*/ {tvm::Var("s1"), 1},
      tvm::Var("b_offset"),
      "b",
      "",
      0,
      /*offset_factor*/ 1);

  auto c_buf = tvm::BufferNode::make(
      tvm::Var("c", tvm::Handle()),
      c->dtype,
      c->shape,
      /*strides*/ {1},
      tvm::Var("c_offset"),
      "c",
      "",
      0,
      /*offset_factor*/ 1);

  int h_unroll = shape_[1] / 32;
  auto sum_int32x8 = tvm::make_const(HalideIR::Int(32, 8), 0);
  auto one = tvm::make_const(HalideIR::Int(16, 16), 1);

  for (int i = 0; i < h_unroll; ++i) {
    auto a_uint8x32 = a_buf.vload({i * 32}, HalideIR::UInt(8, 32));
    auto b_int8x32 = b_buf.vload({0, i * 32}, HalideIR::Int(8, 32));

    auto axb_int16x16 = tvm_codegen::LLVMIntrinsic(HalideIR::Int(16, 16),
                                                   "llvm.x86.avx2.pmadd.ub.sw",
                                                   {a_uint8x32, b_int8x32});
    auto axb_int32x8 = tvm_codegen::LLVMIntrinsic(HalideIR::Int(32, 8),
                                                  "llvm.x86.avx2.pmadd.wd",
                                                  {axb_int16x16, one});
    sum_int32x8 += axb_int32x8;
  }

  sum_int32x8 = tvm_codegen::LLVMIntrinsic(HalideIR::Int(32, 8),
                                           "llvm.x86.avx2.phadd.d",
                                           {sum_int32x8, sum_int32x8});
  sum_int32x8 = tvm_codegen::LLVMIntrinsic(HalideIR::Int(32, 8),
                                           "llvm.x86.avx2.phadd.d",
                                           {sum_int32x8, sum_int32x8});

  auto sum_int32x4_l = tvm_codegen::VectorLow(sum_int32x8);
  auto sum_int32x4_h = tvm_codegen::VectorHigh(sum_int32x8);
  auto sum_int32x4 = sum_int32x4_l + sum_int32x4_h;
  auto sum_int32x1 = tvm_codegen::ExtractElement(sum_int32x4, 0);

  auto reset = c_buf.vstore({0}, tvm::make_const(HalideIR::Int(32, 1), 0));
  auto body = c_buf.vstore({0}, sum_int32x1);
  auto update = c_buf.vstore({0}, sum_int32x1 + c_buf.vload({0}, HalideIR::Int(32, 1)));

  return tvm::TensorIntrinNode::make(
      "intrin_gemv_8bit",
      c->op,
      {a, b},
      {a_buf, b_buf, c_buf},
      body,
      reset,
      update);
}
}  // namespace nuphar
}  // namespace onnxruntime
