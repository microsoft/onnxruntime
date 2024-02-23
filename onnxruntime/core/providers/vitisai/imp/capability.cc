// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "vaip/capability.h"
#include "./vai_assert.h"

namespace vaip {
using namespace ::onnxruntime;

static std::vector<NodeIndex> node_names_to_nodes(const GraphViewer& graph, const std::vector<std::string>& node_names) {
  auto ret = std::vector<NodeIndex>();
  ret.reserve(node_names.size());
  for (auto& onnx_node_name : node_names) {
    // onnnx_node_name is actually node arg name.
    auto node = graph.GetProducerNode(onnx_node_name);
    vai_assert(node != nullptr, std::string("cannot find producer. onnx_node_arg_name=" + onnx_node_name));
    ret.push_back(node->Index());
  }
  return ret;
}

std::unique_ptr<ComputeCapability> XirSubgraphToComputeCapability1(const onnxruntime::GraphViewer& graph, vaip_core::ExecutionProvider* ep, size_t index) {
  auto meta_def = IndexedSubGraph_MetaDef::Create();
  meta_def->constant_initializers() = *ep->get_meta_def_constant_initializer();
  meta_def->inputs() = *ep->get_meta_def_inputs();
  meta_def->outputs() = *ep->get_meta_def_outputs();
  auto indexed_subgraph = IndexedSubGraph::Create();
  indexed_subgraph->Nodes() = node_names_to_nodes(graph, *ep->get_meta_def_nodes());
  static auto g_counter = 1;
  meta_def->name() = std::string("vitis_ai_ep_") + std::to_string(g_counter++);
  meta_def->domain() = "com.xilinx";
  meta_def->since_version() = 1;
  meta_def->status() = ONNX_NAMESPACE::EXPERIMENTAL;
  auto index_proto = ONNX_NAMESPACE::AttributeProto::Create();
  index_proto->set_name("index");
  index_proto->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  index_proto->set_i(index);
  meta_def->attributes()["index"] = *index_proto;
  indexed_subgraph->SetMetaDef(std::move(meta_def));
  return ComputeCapability::Create(std::move(indexed_subgraph));
}

std::vector<std::unique_ptr<ComputeCapability>>
GetComputeCapabilityOps(const onnxruntime::GraphViewer& graph,
                        vaip_core::DllSafe<std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>>* eps,
                        const std::set<std::string>& all_support_optypes_by_eps) {
  std::set<NodeIndex> all_nodes_included_eps;
  for (auto& ep : **eps) {
    auto nodes = node_names_to_nodes(graph, *ep->get_meta_def_nodes());
    all_nodes_included_eps.insert(nodes.begin(), nodes.end());
  }

  std::vector<NodeIndex> node_indexs = graph.GetNodesInTopologicalOrder();
  node_indexs.erase(std::remove_if(node_indexs.begin(), node_indexs.end(), [&](NodeIndex index) { return all_nodes_included_eps.count(index) > 0; }), node_indexs.end());
  node_indexs.erase(std::remove_if(node_indexs.begin(), node_indexs.end(), [&](NodeIndex index) { return all_support_optypes_by_eps.count(graph.GetNode(index)->OpType()) == 0; }), node_indexs.end());

  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (auto& n : node_indexs) {
    auto indexed_subgraph = IndexedSubGraph::Create();
    indexed_subgraph->Nodes() = {n};
    result.emplace_back(ComputeCapability::Create(std::move(indexed_subgraph)));
  }
  return result;
}
}  // namespace vaip
