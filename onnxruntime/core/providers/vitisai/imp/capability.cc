// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "vaip/capability.h"
#include "./vai_assert.h"

#include "core/graph/basic_types.h"

#include "./attr_proto.h"

namespace vaip {
using namespace ::onnxruntime;
std::vector<const Node*> node_arg_names_to_nodes(const GraphViewer& graph,
                                                 const std::vector<std::string>& node_arg_names) {
  auto ret = std::vector<const Node*>();
  ret.reserve(node_arg_names.size());
  for (auto& onnx_node_arg_name : node_arg_names) {
    auto deq = graph.GetProducerNode(onnx_node_arg_name);
    vai_assert(deq != nullptr, std::string("cannot find producer. onnx_node_arg_name=" + onnx_node_arg_name));
    // to support multiple output, some nodes mighe already be inserted.
    auto found = std::find(ret.begin(), ret.end(), deq) != ret.end();
    if (!found) {
      ret.push_back(deq);
    }
  }
  return ret;
}

static std::vector<NodeIndex> node_names_to_nodes(const GraphViewer& graph,
                                                  const std::vector<std::string>& node_names) {
  auto find_node = [&](const std::string& node_name) -> NodeIndex {
    for (auto& n : graph.Nodes()) {
      if (n.Name() == node_name) {
        return n.Index();
      }
    }
    vai_assert(false, std::string("cannot find node: " + node_name));
    return onnxruntime::NodeIndex(-1);
  };
  auto ret = std::vector<NodeIndex>();
  ret.reserve(node_names.size());
  for (auto& onnx_node_name : node_names) {
    ret.push_back(find_node(onnx_node_name));
  }
  return ret;
}

std::unique_ptr<ComputeCapability> XirSubgraphToComputeCapability1(const onnxruntime::GraphViewer& graph, vaip_core::ExecutionProvider* ep, size_t index) {
  auto meta_def = std::make_unique<IndexedSubGraph::MetaDef>();
  auto input_nodes = node_arg_names_to_nodes(graph, *ep->get_meta_def_inputs());
  auto output_nodes = node_arg_names_to_nodes(graph, *ep->get_meta_def_outputs());
  meta_def->constant_initializers = *ep->get_meta_def_constant_initializer();
  meta_def->inputs = *ep->get_meta_def_inputs();
  meta_def->outputs = *ep->get_meta_def_outputs();
  auto indexed_subgraph = std::make_unique<IndexedSubGraph>();
  auto indexed_subgraph_ptr = indexed_subgraph.get();
  indexed_subgraph_ptr->nodes = node_names_to_nodes(graph, *ep->get_meta_def_nodes());
  static auto g_counter = 1;
  meta_def->name = std::string("vitis_ai_ep_") + std::to_string(g_counter++);
  meta_def->domain = "com.xilinx";
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
  auto index_proto = std::unique_ptr<ONNX_NAMESPACE::AttributeProto>(vaip::attr_proto_new_int("index", (int64_t)index));
  meta_def->attributes["index"] = *index_proto;
  indexed_subgraph->SetMetaDef(std::move(meta_def));
  return std::make_unique<ComputeCapability>(std::move(indexed_subgraph));
}

std::vector<std::unique_ptr<ComputeCapability>>
GetComputeCapabilityOps(const onnxruntime::GraphViewer& graph,
                        vaip_core::DllSafe<std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>>* eps,
                        const std::set<std::string>& all_not_support_optypes) {
  std::set<std::string> all_compute_capability_nodes;
  for (auto& ep : **eps) {
    auto nodes = *ep->get_meta_def_nodes();
    for (auto n : nodes)
      all_compute_capability_nodes.insert(n);
  }
  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (auto& n : graph.Nodes()) {
    if ((!all_compute_capability_nodes.count(n.Name())) && all_not_support_optypes.count(n.OpType())) {
      auto meta_def = std::make_unique<IndexedSubGraph::MetaDef>();
      meta_def->name = n.OpType();
      meta_def->domain = n.Domain();
      meta_def->since_version = 1;
      meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
      auto indexed_subgraph = std::make_unique<IndexedSubGraph>();
      indexed_subgraph->nodes.push_back(n.Index());
      for (auto i : n.InputDefs()) {
        meta_def->inputs.push_back(i->Name());
      }
      for (auto i : n.OutputDefs()) {
        meta_def->outputs.push_back(i->Name());
      }
      indexed_subgraph->SetMetaDef(std::move(meta_def));
      result.emplace_back(std::make_unique<ComputeCapability>(std::move(indexed_subgraph)));
    }
  }
  return result;
}
}  // namespace vaip
