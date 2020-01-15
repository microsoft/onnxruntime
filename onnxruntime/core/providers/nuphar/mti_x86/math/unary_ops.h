// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace nuphar {

tvm::Expr exp(const tvm::Expr& x_full);
tvm::Expr log(const tvm::Expr& x_full);

tvm::Tensor Erf(const tvm::Tensor& X, const std::string& name = "erf");
tvm::Tensor Exp(const tvm::Tensor& X, const std::string& name = "exp");
tvm::Tensor Log(const tvm::Tensor& X, const std::string& name = "log");
tvm::Tensor ParametricSoftplus(const tvm::Tensor& X, float alpha, float beta, const std::string& name = "parametric_softplus");
tvm::Tensor ScaledTanh(const tvm::Tensor& X, float alpha, float beta, const std::string& name = "scaled_tanh");
tvm::Tensor Selu(const tvm::Tensor& X, float alpha, float gamma, const std::string& name = "selu");
tvm::Tensor Sigmoid(const tvm::Tensor& X, const std::string& name = "sigmoid");
tvm::Tensor Softplus(const tvm::Tensor& X, const std::string& name = "softplus");
tvm::Tensor Tanh(const tvm::Tensor& X, const std::string& name = "tanh");

}  // namespace nuphar
}  // namespace onnxruntime
