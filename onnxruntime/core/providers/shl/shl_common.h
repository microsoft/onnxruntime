// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/framework/tensor.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/shared/utils/utils.h"

extern "C" {
#include "csi_nn.h"
}

namespace onnxruntime {
namespace shl_ep {
template <typename T>
T* GetShlParams(csinn_session* sess) {
  return (T*)csinn_alloc_params(sizeof(T), sess);
}

template <typename T, typename... Args>
bool is_one_of(const T& value, const Args&... args) {
  return ((value == args) || ...);
}

void UpdateShlTensorDim(csinn_tensor* tensor, std::vector<int64_t> shape);
csinn_dtype_enum GetShlDtypeEnum(const ONNX_NAMESPACE::TypeProto_Tensor type);
csinn_quant_enum GetShlQDtypeEnum(std::string type);
csinn_debug_enum GetShlDebugLevelEnum(std::string type);
csinn_profiler_enum GetShlProfilerLevelEnum(std::string type);
std::vector<std::vector<int>> GetSupportedNodes(const onnxruntime::GraphViewer& graph_viewer,
                                                std::unordered_map<const Node*, std::string> all_fusible_nodes);
csinn_tensor* CreateShlTensor(const NodeArg* onnx_tensor, csinn_session* sess);
csinn_layout_enum GetShlWeightLayoutEnum(int dim_count);
std::unordered_map<std::string, std::vector<std::unordered_map<std::string, const Node*>>>
MarkfusibleNodes(const onnxruntime::GraphViewer& graph_viewer);
std::unordered_map<const Node*, std::string>
GetAllFusionNode(std::unordered_map<std::string, std::vector<std::unordered_map<std::string, const Node*>>> marked_fusible_map);

}  // namespace shl_ep
}  // namespace onnxruntime
