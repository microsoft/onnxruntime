#include "torch/script.h"

// See tutorial: https://github.com/onnx/tutorials/blob/master/PyTorchCustomOperator/README.md

// A dummy PyTorch custom operator for the LongformerAttention (https://github.com/microsoft/onnxruntime/blob/31a6be3d675d09fbe053e03d668695be1bd5fd89/onnxruntime/core/graph/contrib_ops/contrib_defs.cc#L445)
torch::Tensor longformer_attention(
  torch::Tensor input,
  torch::Tensor weight,
  torch::Tensor bias,
  torch::Tensor mask,
  torch::Tensor global_weight,
  torch::Tensor global_bias,
  torch::Tensor global, int64_t num_heads, int64_t window) {
  // Make sure the output shape is correct (same as input). The tensor value does not matter.
  torch::Tensor output = torch::ones(input.sizes(), torch::dtype(torch::kFloat32));
  return output;
}

// namespace is onnxruntime
static auto registry = torch::RegisterOperators("onnxruntime::LongformerAttention", &longformer_attention);
