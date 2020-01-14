// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/scheduler/tensorize/intrin_gemv_ll_extern.h"
#include "core/providers/nuphar/compiler/x86/scheduler/tensorize/ll/gemv_impl.h"
#include <tvm/buffer.h>
#include <tvm/ir.h>

namespace onnxruntime {
namespace nuphar {

const char* gemv_update_func_name = "gemv_update";
const char* gemv_reset_func_name = "gemv_reset";

NaiveLLVMExternGemvTensorization::NaiveLLVMExternGemvTensorization(const std::string& name,
                                                                   const std::vector<int32_t>& shape)
    : TensorizeWithLLVMImport(name, "NaiveLLVMExternGemvTensorization_Parameter", shape) {}

tvm::TensorIntrin NaiveLLVMExternGemvTensorization::CreateTensorIntrin() {
  tvm::Expr m(shape_[0]);
  tvm::Expr l(shape_[1]);

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

  auto body = tvm::ir::Call::make(
      HalideIR::Type(HalideIR::Type::Int, 32, 1),
      gemv_update_func_name,
      {
          c_buf.access_ptr(static_cast<int>(tvm::AccessMask::kWrite)),
          a_buf.access_ptr(static_cast<int>(tvm::AccessMask::kRead)),
          b_buf.access_ptr(static_cast<int>(tvm::AccessMask::kRead)),
          m,
          l,
          /*stride*/ b_buf->strides[0],
      },
      tvm::ir::Call::CallType::Extern);

  auto reduce_init = tvm::ir::Call::make(
      HalideIR::Type(HalideIR::Type::Int, 32, 1),
      gemv_reset_func_name,
      {
          c_buf.access_ptr(static_cast<int>(tvm::AccessMask::kWrite)),
          m,
      },
      tvm::ir::Call::CallType::Extern);

  auto reduce_update = body;

  return tvm::TensorIntrinNode::make(
      "intrin_gemv_ll_extern",
      c->op,
      {a, b},
      {a_buf, b_buf, c_buf},
      tvm::ir::Evaluate::make(body),
      tvm::ir::Evaluate::make(reduce_init),
      tvm::ir::Evaluate::make(reduce_update));
}

const std::string NaiveLLVMExternGemvTensorization::LLVMImportDef() {
  return std::string(gemv_stubs_ir);
}

}  // namespace nuphar
}  // namespace onnxruntime
