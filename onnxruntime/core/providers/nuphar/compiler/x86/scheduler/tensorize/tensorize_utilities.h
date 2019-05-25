// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Expr LLVMIntrinsic(tvm::Type type,
                        const std::string& name,
                        const tvm::Array<tvm::Expr>& args);

tvm::Expr PureIntrinsic(tvm::Type type,
                        const std::string& name,
                        const tvm::Array<tvm::Expr>& args);

tvm::Expr ExtractElement(tvm::Expr expr,
                         int32_t id);

tvm::Expr VectorLow(tvm::Expr expr);
tvm::Expr VectorHigh(tvm::Expr expr);

tvm::Expr Reinterpret(tvm::Type type,
                      tvm::Expr expr);

}  // namespace tvm_codegen
}  // namespace onnxruntime
