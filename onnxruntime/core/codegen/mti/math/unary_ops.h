// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Abs(const tvm::Tensor& X, const std::string& name = "abs");
tvm::Tensor Affine(const tvm::Tensor& X, float alpha, float beta, const std::string& name = "affine");
tvm::Tensor Ceil(const tvm::Tensor& X, const std::string& name = "ceil");
tvm::Tensor Clip(const tvm::Tensor& X, tvm::Expr min_value, tvm::Expr max_value, const std::string& name = "clip");
tvm::Tensor Elu(const tvm::Tensor& X, float alpha, const std::string& name = "elu");
tvm::Tensor Exp(const tvm::Tensor& X, const std::string& name = "exp");
tvm::Tensor Floor(const tvm::Tensor& X, const std::string& name = "floor");
tvm::Tensor HardSigmoid(const tvm::Tensor& X, float alpha, float beta, const std::string& name = "hard_sigmoid");
tvm::Tensor LeakyRelu(const tvm::Tensor& X, float alpha, const std::string& name = "leaky_relu");
tvm::Tensor Log(const tvm::Tensor& X, const std::string& name = "log");
tvm::Tensor Neg(const tvm::Tensor& X, const std::string& name = "neg");
tvm::Tensor ParametricSoftplus(const tvm::Tensor& X, float alpha, float beta, const std::string& name = "parametric_softplus");
tvm::Tensor Reciprocal(const tvm::Tensor& X, const std::string& name = "reciprocal");
tvm::Tensor Relu(const tvm::Tensor& X, const std::string& name = "relu");
tvm::Tensor ScaledTanh(const tvm::Tensor& X, float alpha, float beta, const std::string& name = "scaled_tanh");
tvm::Tensor Selu(const tvm::Tensor& X, float alpha, float gamma, const std::string& name = "selu");
tvm::Tensor Sigmoid(const tvm::Tensor& X, const std::string& name = "sigmoid");
tvm::Tensor SignNoZero(const tvm::Tensor& X, const std::string& name = "sign_no_zero");
tvm::Tensor Softplus(const tvm::Tensor& X, const std::string& name = "softplus");
tvm::Tensor Softsign(const tvm::Tensor& X, const std::string& name = "softsign");
tvm::Tensor Sqrt(const tvm::Tensor& X, const std::string& name = "sqrt");
tvm::Tensor Tanh(const tvm::Tensor& X, const std::string& name = "tanh");
tvm::Tensor ThresholdedRelu(const tvm::Tensor& X, float alpha, const std::string& name = "thresholded_relu");

}  // namespace tvm_codegen
}  // namespace onnxruntime
