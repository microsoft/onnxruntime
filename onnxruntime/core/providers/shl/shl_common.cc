// Copyright (C) 2016-2023 T-Head Semiconductor Co., Ltd. All rights reserved.
// Licensed under the MIT License.

#include "shl_common.h"

namespace onnxruntime {
namespace shl_ep {
using FuseMarkerFn = std::function<std::vector<std::unordered_map<std::string, const Node*>>(const onnxruntime::GraphViewer& graph_viewer)>;

csinn_dtype_enum GetShlDtypeEnum(const ONNX_NAMESPACE::TypeProto_Tensor type) {
  if (type.has_elem_type()) {
    auto type_ = type.elem_type();
    switch (type_) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        return CSINN_DTYPE_FLOAT32;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
        return CSINN_DTYPE_FLOAT16;
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
        return CSINN_DTYPE_UINT8;
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:
        return CSINN_DTYPE_INT8;
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
        return CSINN_DTYPE_INT32;
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:
        return CSINN_DTYPE_INT64;
      default:
        // TODO: support other type
        throw std::invalid_argument("The input of graph doesn't have valid type");
    }
  }
  throw std::invalid_argument("The input of graph doesn't have valid type");
  return CSINN_DTYPE_FLOAT32;
}

csinn_layout_enum GetShlActLayoutEnum(int dim_count) {
  switch (dim_count) {
    case 1:
      return CSINN_LAYOUT_N;
    case 2:
      return CSINN_LAYOUT_NC;
    case 3:
      return CSINN_LAYOUT_NCW;
    case 4:
      return CSINN_LAYOUT_NCHW;
    case 5:
      return CSINN_LAYOUT_NCDHW;
    default:
      return CSINN_LAYOUT_NCHW;
  }
  throw std::invalid_argument("The input of graph doesn't have valid type");
  return CSINN_LAYOUT_NCHW;
}

csinn_layout_enum GetShlWeightLayoutEnum(int dim_count) {
  switch (dim_count) {
    case 1:
      return CSINN_LAYOUT_O;
    case 2:
      return CSINN_LAYOUT_OI;
    case 3:
      return CSINN_LAYOUT_OIW;
    case 4:
      return CSINN_LAYOUT_OIHW;
    case 5:
      return CSINN_LAYOUT_OIDHW;
    default:
      return CSINN_LAYOUT_OIHW;
  }
  throw std::invalid_argument("The input of graph doesn't have valid type");
  return CSINN_LAYOUT_OIHW;
}

csinn_quant_enum GetShlQDtypeEnum(std::string type) {
  if (type == "CSINN_QUANT_FLOAT32")
    return CSINN_QUANT_FLOAT32;
  else if (type == "CSINN_QUANT_FLOAT16")
    return CSINN_QUANT_FLOAT16;

  throw std::invalid_argument("The input of graph doesn't have valid type");
  return CSINN_QUANT_FLOAT32;
}

csinn_debug_enum GetShlDebugLevelEnum(std::string type) {
  if (type == "CSINN_DEBUG_LEVEL_DEBUG")
    return CSINN_DEBUG_LEVEL_DEBUG;
  else if (type == "CSINN_DEBUG_LEVEL_INFO")
    return CSINN_DEBUG_LEVEL_INFO;
  else if (type == "CSINN_DEBUG_LEVEL_WARNING")
    return CSINN_DEBUG_LEVEL_WARNING;
  else if (type == "CSINN_DEBUG_LEVEL_ERROR")
    return CSINN_DEBUG_LEVEL_ERROR;
  else if (type == "CSINN_DEBUG_LEVEL_FATAL")
    return CSINN_DEBUG_LEVEL_FATAL;

  throw std::invalid_argument("Shl debug level get error.");
  return CSINN_DEBUG_LEVEL_INFO;
}

csinn_profiler_enum GetShlProfilerLevelEnum(std::string type) {
  if (type == "CSINN_PROFILER_LEVEL_UNSET")
    return CSINN_PROFILER_LEVEL_UNSET;
  else if (type == "CSINN_PROFILER_LEVEL_TIMER")
    return CSINN_PROFILER_LEVEL_TIMER;
  else if (type == "CSINN_PROFILER_LEVEL_DUMP")
    return CSINN_PROFILER_LEVEL_DUMP;
  else if (type == "CSINN_PROFILER_LEVEL_ALL")
    return CSINN_PROFILER_LEVEL_ALL;

  throw std::invalid_argument("Shl prifiler level get error.");
  return CSINN_PROFILER_LEVEL_UNSET;
}

std::pair<bool, std::string> IsNodeSupported(const GraphViewer& graph_viewer, const Node* node) {
  const auto& op = node->OpType();

  const std::vector<std::string> supported_types{
      "Conv", "Gemm"};
  if (std::find(supported_types.begin(), supported_types.end(), op) ==
      supported_types.end()) {
    return {false, "Unsupported operator"};
  }

  NodeAttrHelper helper(*node);

  if (op == "Conv") {
    const NodeArg* input = node->InputDefs()[0];
    auto in_shape = input->Shape();
    if (!is_one_of<int64_t>(in_shape->dim_size(), 4)) {
      return {false, "conv only supporte conv2d now"};
    }
  } else if (op == "Gemm") {
    const auto transA = helper.Get("transA", 0);
    const auto transB = helper.Get("transB", 0);
    const auto alpha = helper.Get("alpha", 1.0f);
    const auto beta = helper.Get("beta", 1.0f);
    if (!(transA == 0 && transB == 1 && alpha == 1.f && beta == 1.f)) {
      return {false,
              "Only transA == 0, transB == 1, alpha == 1.0 and beta == "
              "1.0 is supported."};
    }
  }

  return {true, ""};
}


std::pair<bool, std::string> IsfusibleNode(std::unordered_map<const Node*, std::string> all_fusible_nodes, const Node* node) {
  if (all_fusible_nodes.count(node)) {
    return {true, "node is fusible."};
  }
  return {false, "node is nonfusible."};
}

std::vector<std::vector<int>> GetSupportedNodes(const onnxruntime::GraphViewer& graph_viewer,
                                                std::unordered_map<const Node*, std::string> all_fusible_nodes) {
  std::vector<std::vector<int>> supported_node_vecs;
  std::vector<int> supported_node_vec;
  const std::vector<NodeIndex>& node_index = graph_viewer.GetNodesInTopologicalOrder();
  for (auto i : node_index) {
    bool supported;
    std::string error_msg;
    std::tie(supported, error_msg) = IsNodeSupported(graph_viewer, graph_viewer.GetNode(i));
    if (!supported)
      std::tie(supported, error_msg) = IsfusibleNode(all_fusible_nodes, graph_viewer.GetNode(i));
    if (supported) {
      supported_node_vec.push_back(i);
    } else {
      const auto& op = graph_viewer.GetNode(i)->OpType();
      LOGS_DEFAULT(INFO) << op << ": " << error_msg;
      if (!supported_node_vec.empty()) {
        supported_node_vecs.push_back(supported_node_vec);
        supported_node_vec.clear();
      }
    }
  }
  if (!supported_node_vec.empty()) {
    supported_node_vecs.push_back(supported_node_vec);
  }

  return supported_node_vecs;
}

void UpdateShlTensorDim(csinn_tensor* tensor, std::vector<int64_t> shape) {
  tensor->dim_count = shape.size();
  std::transform(shape.begin(), shape.end(), tensor->dim, [](int64_t val) -> int32_t {
    return static_cast<int32_t>(val);
  });
}

csinn_tensor* CreateShlTensor(const NodeArg* onnx_tensor, csinn_session* sess) {
  csinn_tensor* shl_tensor = csinn_alloc_tensor(sess);

  shl_tensor->name = const_cast<char*>(onnx_tensor->Name().c_str());
  shl_tensor->dtype = shl_ep::GetShlDtypeEnum(onnx_tensor->TypeAsProto()->tensor_type());

  auto onnx_shape = onnx_tensor->Shape();
  if (onnx_shape != NULL) {
    auto tensor_shape = utils::GetTensorShapeFromTensorShapeProto(*onnx_shape).AsShapeVector();
    if (tensor_shape.size() == 0) {
      shl_tensor->dim_count = 1;
      shl_tensor->dim[0] = 1;
    } else {
      shl_tensor->dim_count = tensor_shape.size();
      for (int32_t i = 0; i < shl_tensor->dim_count; i++) {
        shl_tensor->dim[i] = tensor_shape[i];
      }
    }
  } else {
    shl_tensor->dim_count = 0;
    shl_tensor->dim[0] = 0;
  }

  shl_tensor->layout = GetShlActLayoutEnum(shl_tensor->dim_count);

  return shl_tensor;
};

std::unordered_map<std::string, std::vector<std::unordered_map<std::string, const Node*>>>
MarkfusibleNodes(const onnxruntime::GraphViewer& graph_viewer) {
  // TODO: add fusion marker here
  static std::unordered_map<std::string, FuseMarkerFn> fuse_markers;

  std::unordered_map<std::string, std::vector<std::unordered_map<std::string, const Node*>>> marked_fusible_map;
  for (const auto& iter : fuse_markers) {
    std::vector<std::unordered_map<std::string, const Node*>> fuse_nodes = iter.second(graph_viewer);
    if (fuse_nodes.size())
      marked_fusible_map.emplace(iter.first, fuse_nodes);
  }
  return marked_fusible_map;
}

std::unordered_map<const Node*, std::string>
GetAllFusionNode(std::unordered_map<std::string, std::vector<std::unordered_map<std::string, const Node*>>> marked_fusible_map) {
  std::unordered_map<const Node*, std::string> new_all_fusible_nodes;

  for (auto& iter : marked_fusible_map) {
    for (auto& node_map : iter.second) {
      for (auto& iter2 : node_map) {
        if (iter2.first == "key_node") {
          if (new_all_fusible_nodes.count(iter2.second)) {
            new_all_fusible_nodes[iter2.second] = iter.first;
          } else {
            new_all_fusible_nodes.emplace(iter2.second, iter.first);
          }
        } else {
          if (!new_all_fusible_nodes.count(iter2.second))
            new_all_fusible_nodes.emplace(iter2.second, "");
        }
      }
    }
  }
  return new_all_fusible_nodes;
}

}  // namespace shl_ep
}  // namespace onnxruntime
