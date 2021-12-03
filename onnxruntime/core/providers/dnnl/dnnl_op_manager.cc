// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#include "dnnl_op_manager.h"
#include <iostream>

namespace onnxruntime {
DnnlOpManager::DnnlOpManager() {
  dnnl_ops_map_.emplace(std::make_pair("Abs", std::unique_ptr<DnnlNodeCapability>(new DnnlElementwiseCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Add", std::unique_ptr<DnnlNodeCapability>(new DnnlBinaryNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("AveragePool", std::unique_ptr<DnnlNodeCapability>(new DnnlPoolNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("BatchNormalization", std::unique_ptr<DnnlNodeCapability>(new DnnlBatchNormalizationNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Conv", std::unique_ptr<DnnlNodeCapability>(new DnnlDefaultNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Div", std::unique_ptr<DnnlNodeCapability>(new DnnlBinaryNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("DynamicQuantizeLinear", std::unique_ptr<DnnlNodeCapability>(new DnnlDynamicQuantizeLinearNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Elu", std::unique_ptr<DnnlNodeCapability>(new DnnlElementwiseCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Exp", std::unique_ptr<DnnlNodeCapability>(new DnnlElementwiseCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Gemm", std::unique_ptr<DnnlNodeCapability>(new DnnlGemmNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("GlobalAveragePool", std::unique_ptr<DnnlNodeCapability>(new DnnlPoolNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("GlobalMaxPool", std::unique_ptr<DnnlNodeCapability>(new DnnlPoolNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("LeakyRelu", std::unique_ptr<DnnlNodeCapability>(new DnnlElementwiseCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Log", std::unique_ptr<DnnlNodeCapability>(new DnnlElementwiseCapability())));
  dnnl_ops_map_.emplace(std::make_pair("LRN", std::unique_ptr<DnnlNodeCapability>(new DnnlDefaultNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("MatMul", std::unique_ptr<DnnlNodeCapability>(new DnnlMatMulNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("MatMulInteger", std::unique_ptr<DnnlNodeCapability>(new DnnlMatMulIntegerNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("MaxPool", std::unique_ptr<DnnlNodeCapability>(new DnnlPoolNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Mul", std::unique_ptr<DnnlNodeCapability>(new DnnlBinaryNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Pow", std::unique_ptr<DnnlNodeCapability>(new DnnlPowNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("ReduceMean", std::unique_ptr<DnnlNodeCapability>(new DnnlReduceMeanNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Relu", std::unique_ptr<DnnlNodeCapability>(new DnnlElementwiseCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Reshape", std::unique_ptr<DnnlNodeCapability>(new DnnlReshapeNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Round", std::unique_ptr<DnnlNodeCapability>(new DnnlElementwiseCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Sigmoid", std::unique_ptr<DnnlNodeCapability>(new DnnlElementwiseCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Softmax", std::unique_ptr<DnnlNodeCapability>(new DnnlSoftmaxNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Softplus", std::unique_ptr<DnnlNodeCapability>(new DnnlElementwiseCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Squeeze", std::unique_ptr<DnnlNodeCapability>(new DnnlSqueezeNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Sqrt", std::unique_ptr<DnnlNodeCapability>(new DnnlElementwiseCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Sub", std::unique_ptr<DnnlNodeCapability>(new DnnlBinaryNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Sum", std::unique_ptr<DnnlNodeCapability>(new DnnlSumNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Tanh", std::unique_ptr<DnnlNodeCapability>(new DnnlElementwiseCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Transpose", std::unique_ptr<DnnlNodeCapability>(new DnnlDefaultNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("Unsqueeze", std::unique_ptr<DnnlNodeCapability>(new DnnlSqueezeNodeCapability())));
#if defined(ENABLE_TRAINING)
  dnnl_ops_map_.emplace(std::make_pair("AveragePoolGrad", std::unique_ptr<DnnlNodeCapability>(new DnnlPoolNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("ConvGrad", std::unique_ptr<DnnlNodeCapability>(new DnnlDefaultNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("MaxPoolGrad", std::unique_ptr<DnnlNodeCapability>(new DnnlPoolNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("ReluGrad", std::unique_ptr<DnnlNodeCapability>(new DnnlDefaultNodeCapability())));
  dnnl_ops_map_.emplace(std::make_pair("SoftmaxGrad", std::unique_ptr<DnnlNodeCapability>(new DnnlSoftmaxNodeCapability())));
#endif  // ENABLE_TRAINING
}

bool DnnlOpManager::IsNodeSupported(const Node* node, const GraphViewer& graph_viewer) const {
  auto it = dnnl_ops_map_.find(node->OpType());
  if (it == dnnl_ops_map_.end()) {
    return false;
  }
  return it->second->Supported(node, graph_viewer);
}

bool DnnlOpManager::IsOpTypeAvalible(const std::string& opType) const {
  auto op_it = dnnl_ops_map_.find(opType);
  return (op_it != dnnl_ops_map_.end());
}
}  // namespace onnxruntime
