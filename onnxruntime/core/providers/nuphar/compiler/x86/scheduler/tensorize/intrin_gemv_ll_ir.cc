// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "intrin_gemv_ll_ir.h"

#include "core/providers/nuphar/compiler/x86/scheduler/tensorize/tensorize_utilities.h"
#include <tvm/buffer.h>
#include <tvm/codegen.h>
#include <tvm/ir.h>

namespace onnxruntime {
namespace nuphar {

const int32_t dim0 = 1;
const int32_t dim1 = 8;

NaiveLLVMIRGemvTensorization::NaiveLLVMIRGemvTensorization(const std::string& name)
    : TensorizeBase(name, "NaiveLLVMIRGemvTensorization_Parameter", {dim0, dim1}) {}

tvm::TensorIntrin NaiveLLVMIRGemvTensorization::CreateTensorIntrin() {
  tvm::Expr m(dim0);
  tvm::Expr l(dim1);

  auto a = tvm::placeholder({l});
  auto b = tvm::placeholder({m, l});
  auto k = tvm::reduce_axis({0, l});

  auto c = tvm::compute({m}, [&](tvm::Var i) {
    return tvm::sum(a(k) * b(i, k), {k});
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

  auto a_float32x8 = a_buf.vload({0}, HalideIR::Float(32, 8));
  auto b_float32x8 = b_buf.vload({0, 0}, HalideIR::Float(32, 8));
  auto z_float32x8 = tvm::make_const(HalideIR::Float(32, 8), 0);

  auto axb = tvm_codegen::LLVMIntrinsic(HalideIR::Float(32, 8),
                                        "llvm.x86.fma.vfmadd.ps.256",
                                        {a_float32x8,
                                         b_float32x8,
                                         z_float32x8});

  auto sum = tvm_codegen::ExtractElement(axb, 0);

  for (int i = 1; i < 8; ++i) {
    auto z0 = tvm_codegen::ExtractElement(axb, i);
    sum += z0;
  }

  auto body = c_buf.vstore({0}, sum);
  auto reset = c_buf.vstore({0}, tvm::make_const(HalideIR::Float(32, 1), 0));
  auto update = c_buf.vstore({0}, sum + c_buf.vload({0}, HalideIR::Float(32, 1)));

  return tvm::TensorIntrinNode::make(
      "intrin_gemv_ll_ir",
      c->op,
      {a, b},
      {a_buf, b_buf, c_buf},
      body,
      reset,
      update);
}
}  // namespace nuphar
}  // namespace onnxruntime
