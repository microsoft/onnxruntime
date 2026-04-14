// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <torch/torch.h>

namespace onnxruntime {
namespace lazytensor {
// This function contains function for comparing values
// and printing values. They are mainly used for debugging.

bool CompareTensor(const at::Tensor& left, const at::Tensor& right);
bool CompareScalar(const at::Scalar& left, const at::Scalar& right);
bool Compare(const c10::IValue& left, const c10::IValue& right);
bool CompareStack(const torch::jit::Stack& left, const torch::jit::Stack& right);
std::string ToString(const c10::IValue& value);
std::string ToString(const at::ArrayRef<c10::IValue>& values);
// "torch::jit::Value" is abstract symbol in torch::jit::Graph.
// It represents inputs and outputs for graph, block, and node.
// Note that the actual computation reuslt's type is c10::IValue.
std::string ToString(const torch::jit::Value& value);
std::string ToString(const torch::jit::Node& node);
}  // namespace lazytensor
}  // namespace onnxruntime
