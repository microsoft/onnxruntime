// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tensorize_utilities.h"
#include "core/codegen/common/common.h"

#include <tvm/codegen.h>
#include <tvm/ir_pass.h>
#include <tvm/ir.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Expr LLVMIntrinsic(tvm::Type type,
                        const std::string& name,
                        const tvm::Array<tvm::Expr>& args) {
  tvm::Array<tvm::Expr> llvm_intrinsic_args;
  auto llvm_intrinsic_id = tvm::codegen::LookupLLVMIntrinsic(name);
  ORT_ENFORCE(llvm_intrinsic_id != 0);

  llvm_intrinsic_args.push_back(tvm::make_const(HalideIR::UInt(32), llvm_intrinsic_id));
  llvm_intrinsic_args.push_back(tvm::make_const(HalideIR::UInt(32), 0));

  for (auto& arg : args) {
    llvm_intrinsic_args.push_back(arg);
  }

  // llvm intrinsic is always a pure intrinsic for now
  return PureIntrinsic(type, "llvm_intrin", llvm_intrinsic_args);
}

tvm::Expr PureIntrinsic(tvm::Type type,
                        const std::string& name,
                        const tvm::Array<tvm::Expr>& args) {
  return tvm::ir::Call::make(type,
                             name,
                             args,
                             tvm::ir::Call::CallType::PureIntrinsic);
}

tvm::Expr ExtractElement(tvm::Expr expr,
                         int32_t id) {
  // element type
  tvm::Type type = HalideIR::Type(expr.type().code(), expr.type().bits(), 1);
  return PureIntrinsic(type,
                       "extract_element",
                       {expr,
                        tvm::make_const(HalideIR::UInt(32), id)});
}

tvm::Expr InsertElement(tvm::Expr expr,
                        tvm::Expr elem,
                        int32_t id) {
  tvm::Type type = HalideIR::Type(expr.type().code(), expr.type().bits(), expr.type().lanes());
  return PureIntrinsic(type,
                       "insert_element",
                       {expr,
                        elem,
                        tvm::make_const(HalideIR::UInt(32), id)});
}

tvm::Expr VectorLow(tvm::Expr expr) {
  // Generated vector will have half of the lanes
  tvm::Type type = HalideIR::Type(expr.type().code(), expr.type().bits(), expr.type().lanes() / 2);
  return PureIntrinsic(type,
                       "vectorlow",
                       {expr});
}

tvm::Expr VectorHigh(tvm::Expr expr) {
  // Generated vector will have half of the lanes
  tvm::Type type = HalideIR::Type(expr.type().code(), expr.type().bits(), expr.type().lanes() / 2);
  return PureIntrinsic(type,
                       "vectorhigh",
                       {expr});
}

tvm::Expr VectorConcat(tvm::Array<tvm::Expr> exprs) {
  int lanes = 0;
  for (auto expr : exprs) {
    lanes += expr.type().lanes();
  }
  tvm::Type type = HalideIR::Type(exprs[0].type().code(),
                                  exprs[0].type().bits(),
                                  lanes);
  return PureIntrinsic(type,
                       "vectorconcat",
                       exprs);
}

tvm::Expr VectorCombine(tvm::Expr expr1, tvm::Expr expr2) {
  tvm::Type type = HalideIR::Type(expr1.type().code(), expr1.type().bits(),
                                  expr1.type().lanes() + expr2.type().lanes());
  return PureIntrinsic(type,
                       "vectorcombine",
                       {expr1, expr2});
}

tvm::Expr Reinterpret(tvm::Type type,
                      tvm::Expr expr) {
  return PureIntrinsic(type,
                       "reinterpret",
                       {expr});
}

tvm::Stmt MergeStmts(std::vector<tvm::Stmt>& stmts) {
  if (stmts.size() == 0)
    return tvm::ir::Evaluate::make(0);

  tvm::Stmt res = stmts.back();
  res = tvm::ir::Simplify(res);
  for (size_t i = stmts.size() - 1; i > 0; i--) {
    stmts[i - 1] = tvm::ir::Simplify(stmts[i - 1]);
    res = tvm::ir::Block::make(stmts[i - 1], res);
  }

  res = tvm::ir::Simplify(res);
  return res;
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
