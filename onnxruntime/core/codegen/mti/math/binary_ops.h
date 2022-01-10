// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/te/operation.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor Add(const tvm::te::Tensor& lhs, const tvm::te::Tensor& rhs, const std::string& name = "add");
tvm::te::Tensor Add(const tvm::te::Tensor& lhs, const  tvm::PrimExpr& rhs, const std::string& name = "add");
tvm::te::Tensor Add(const  tvm::PrimExpr& lhs, const tvm::te::Tensor& rhs, const std::string& name = "add");
tvm::te::Tensor Div(const tvm::te::Tensor& lhs, const tvm::te::Tensor& rhs, const std::string& name = "div");
tvm::te::Tensor Div(const tvm::te::Tensor& lhs, const  tvm::PrimExpr& rhs, const std::string& name = "div");
tvm::te::Tensor Div(const  tvm::PrimExpr& lhs, const tvm::te::Tensor& rhs, const std::string& name = "div");
tvm::te::Tensor Equal(const tvm::te::Tensor& lhs, const tvm::te::Tensor& rhs, const std::string& name = "equal");
tvm::te::Tensor Equal(const tvm::te::Tensor& lhs, const  tvm::PrimExpr& rhs, const std::string& name = "equal");
tvm::te::Tensor Equal(const  tvm::PrimExpr& lhs, const tvm::te::Tensor& rhs, const std::string& name = "equal");
tvm::te::Tensor Greater(const tvm::te::Tensor& lhs, const tvm::te::Tensor& rhs, const std::string& name = "greater");
tvm::te::Tensor Greater(const tvm::te::Tensor& lhs, const  tvm::PrimExpr& rhs, const std::string& name = "greater");
tvm::te::Tensor Greater(const  tvm::PrimExpr& lhs, const tvm::te::Tensor& rhs, const std::string& name = "greater");
tvm::te::Tensor Less(const tvm::te::Tensor& lhs, const tvm::te::Tensor& rhs, const std::string& name = "less");
tvm::te::Tensor Less(const tvm::te::Tensor& lhs, const  tvm::PrimExpr& rhs, const std::string& name = "less");
tvm::te::Tensor Less(const  tvm::PrimExpr& lhs, const tvm::te::Tensor& rhs, const std::string& name = "less");
tvm::te::Tensor Max(const tvm::te::Tensor& lhs, const tvm::te::Tensor& rhs, const std::string& name = "max");
tvm::te::Tensor Max(const tvm::te::Tensor& lhs, const  tvm::PrimExpr& rhs, const std::string& name = "max");
tvm::te::Tensor Max(const  tvm::PrimExpr& lhs, const tvm::te::Tensor& rhs, const std::string& name = "max");
tvm::te::Tensor Min(const tvm::te::Tensor& lhs, const tvm::te::Tensor& rhs, const std::string& name = "min");
tvm::te::Tensor Min(const tvm::te::Tensor& lhs, const  tvm::PrimExpr& rhs, const std::string& name = "min");
tvm::te::Tensor Min(const  tvm::PrimExpr& lhs, const tvm::te::Tensor& rhs, const std::string& name = "min");
tvm::te::Tensor Mul(const tvm::te::Tensor& lhs, const tvm::te::Tensor& rhs, const std::string& name = "mul");
tvm::te::Tensor Mul(const tvm::te::Tensor& lhs, const  tvm::PrimExpr& rhs, const std::string& name = "mul");
tvm::te::Tensor Mul(const  tvm::PrimExpr& lhs, const tvm::te::Tensor& rhs, const std::string& name = "mul");
tvm::te::Tensor PRelu(const tvm::te::Tensor& lhs, const tvm::te::Tensor& rhs, const std::string& name = "prelu");
tvm::te::Tensor PRelu(const tvm::te::Tensor& lhs, const  tvm::PrimExpr& rhs, const std::string& name = "prelu");
tvm::te::Tensor Sub(const tvm::te::Tensor& lhs, const tvm::te::Tensor& rhs, const std::string& name = "sub");
tvm::te::Tensor Sub(const tvm::te::Tensor& lhs, const  tvm::PrimExpr& rhs, const std::string& name = "sub");
tvm::te::Tensor Sub(const  tvm::PrimExpr& lhs, const tvm::te::Tensor& rhs, const std::string& name = "sub");

}  // namespace tvm_codegen
}  // namespace onnxruntime
