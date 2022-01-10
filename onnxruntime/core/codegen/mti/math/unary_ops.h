// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <tvm/te/operation.h>

namespace onnxruntime {
namespace tvm_codegen {

tvm::te::Tensor Abs(const tvm::te::Tensor& X, const std::string& name = "abs");
tvm::te::Tensor Affine(const tvm::te::Tensor& X, float alpha, float beta, const std::string& name = "affine");
tvm::te::Tensor Ceil(const tvm::te::Tensor& X, const std::string& name = "ceil");
tvm::te::Tensor Clip(const tvm::te::Tensor& X,  tvm::PrimExpr min_value,  tvm::PrimExpr max_value, const std::string& name = "clip");
tvm::te::Tensor Elu(const tvm::te::Tensor& X, float alpha, const std::string& name = "elu");
tvm::te::Tensor Exp(const tvm::te::Tensor& X, const std::string& name = "exp");
tvm::te::Tensor Floor(const tvm::te::Tensor& X, const std::string& name = "floor");
tvm::te::Tensor HardSigmoid(const tvm::te::Tensor& X, float alpha, float beta, const std::string& name = "hard_sigmoid");
tvm::te::Tensor LeakyRelu(const tvm::te::Tensor& X, float alpha, const std::string& name = "leaky_relu");
tvm::te::Tensor Log(const tvm::te::Tensor& X, const std::string& name = "log");
tvm::te::Tensor Neg(const tvm::te::Tensor& X, const std::string& name = "neg");
tvm::te::Tensor ParametricSoftplus(const tvm::te::Tensor& X, float alpha, float beta, const std::string& name = "parametric_softplus");
tvm::te::Tensor Reciprocal(const tvm::te::Tensor& X, const std::string& name = "reciprocal");
tvm::te::Tensor Relu(const tvm::te::Tensor& X, const std::string& name = "relu");
tvm::te::Tensor ScaledTanh(const tvm::te::Tensor& X, float alpha, float beta, const std::string& name = "scaled_tanh");
tvm::te::Tensor Selu(const tvm::te::Tensor& X, float alpha, float gamma, const std::string& name = "selu");
tvm::te::Tensor Sigmoid(const tvm::te::Tensor& X, const std::string& name = "sigmoid");
tvm::te::Tensor SignNoZero(const tvm::te::Tensor& X, const std::string& name = "sign_no_zero");
tvm::te::Tensor Softplus(const tvm::te::Tensor& X, const std::string& name = "softplus");
tvm::te::Tensor Softsign(const tvm::te::Tensor& X, const std::string& name = "softsign");
tvm::te::Tensor Sqrt(const tvm::te::Tensor& X, const std::string& name = "sqrt");
tvm::te::Tensor Tanh(const tvm::te::Tensor& X, const std::string& name = "tanh");
tvm::te::Tensor ThresholdedRelu(const tvm::te::Tensor& X, float alpha, const std::string& name = "thresholded_relu");

}  // namespace tvm_codegen
}  // namespace onnxruntime
