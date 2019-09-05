// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tensorize_utilities.h"
#include "core/codegen/common/common.h"

#include <tvm/codegen.h>
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

tvm::Expr VectorLow(tvm::Expr expr) {
  tvm::Type type = HalideIR::Type(expr.type().code(), expr.type().bits(), expr.type().lanes());
  return PureIntrinsic(type,
                       "vectorlow",
                       {expr});
}

tvm::Expr VectorHigh(tvm::Expr expr) {
  tvm::Type type = HalideIR::Type(expr.type().code(), expr.type().bits(), expr.type().lanes());
  return PureIntrinsic(type,
                       "vectorhigh",
                       {expr});
}

tvm::Expr Reinterpret(tvm::Type type,
                      tvm::Expr expr) {
  return PureIntrinsic(type,
                       "reinterpret",
                       {expr});
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
