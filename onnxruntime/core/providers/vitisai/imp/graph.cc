// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "vaip/graph.h"

#include <codecvt>
#include <fstream>
#include <filesystem>
#include <limits>
#include <locale>
#include <string>

#include "core/providers/shared_library/provider_api.h"
#include "./vai_assert.h"

#include "vaip/node.h"
#include "vaip/node_arg.h"

namespace vaip {

struct NodeEdgeT {
  const onnxruntime::NodeIndex src_node_index;
  const onnxruntime::NodeIndex dst_node_index;
  const int src_arg_index;
  const int dst_arg_index;
};

static void graph_remove_node(Graph& graph, const Node& node) {
  auto remove_edges = std::vector<NodeEdgeT>();
  for (auto it = node.InputEdgesBegin(); it != node.InputEdgesEnd(); ++it) {
    remove_edges.push_back(NodeEdgeT{it->GetNode().Index(), node.Index(), it->GetSrcArgIndex(), it->GetDstArgIndex()});
  }
  for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
    remove_edges.push_back(NodeEdgeT{node.Index(), it->GetNode().Index(), it->GetSrcArgIndex(), it->GetDstArgIndex()});
  }
  for (auto it : remove_edges) {
    graph.RemoveEdge(it.src_node_index, it.dst_node_index, it.src_arg_index, it.dst_arg_index);
  }
  graph.RemoveNode(node.Index());
}

static std::vector<const NodeArg*> node_get_implicit_input_node_args(const Node& node) {
  auto ret = std::vector<const NodeArg*>();
  auto implicit_input_defs = node.ImplicitInputDefs();
  ret.reserve(implicit_input_defs.size());
  for (auto i : implicit_input_defs) {
    ret.push_back(i);
  }
  return ret;
}
Node& graph_add_node(Graph& graph, const std::string& name, const std::string& op_type, const std::string& description,
                     const std::vector<const NodeArg*>& input_args, const std::vector<const NodeArg*>& output_args,
                     const NodeAttributes& attributes, const std::string& domain) {
  std::vector<NodeArg*> inputs;
  inputs.reserve(input_args.size());
  for (auto i : input_args) {
    inputs.push_back(const_cast<NodeArg*>(i));
  }
  std::vector<NodeArg*> outputs;
  outputs.reserve(output_args.size());
  for (auto i : output_args) {
    outputs.push_back(const_cast<NodeArg*>(i));
  }
  auto& ret = graph.AddNode(name, op_type, description, inputs, outputs, &attributes, domain);
  auto src_arg_index = 0;
  for (auto& o : outputs) {
    auto consumers = graph.GetConsumerNodes(o->Name());
    for (auto& consumer : consumers) {
      auto dst_arg_index = 0u;
      auto tmp_inputs = node_get_inputs(*consumer);
      for (auto ni : *tmp_inputs) {
        auto name1 = ni.node_arg->Name();
        if (name1 == o->Name()) {
          graph.AddEdge(ret.Index(), consumer->Index(), src_arg_index, dst_arg_index);
        }
        dst_arg_index = dst_arg_index + 1;
      }
      // dst_arg_index should not init again.
      for (auto implicit_node_arg : node_get_implicit_input_node_args(*consumer)) {
        auto name1 = implicit_node_arg->Name();
        if (name1 == o->Name()) {
          graph.AddEdge(ret.Index(), consumer->Index(), src_arg_index, dst_arg_index);
        }
        dst_arg_index = dst_arg_index + 1;
      }
    }
    src_arg_index = src_arg_index + 1;
  }
  return ret;
}

void graph_remove_node(Graph& graph, const NodeInput& node_input) {
  if (node_input.node == nullptr && node_input.node_arg != nullptr) {
    assert(node_input.node_arg->Exists());
    assert(node_arg_is_constant(graph, *node_input.node_arg));
    graph.RemoveInitializedTensor(node_input.node_arg->Name());
  } else if (node_input.node != nullptr && node_input.node_arg != nullptr) {
    graph_remove_node(graph, *node_input.node);
  } else if (node_input.node != nullptr && node_input.node_arg == nullptr) {
    graph_remove_node(graph, *node_input.node);
  } else if (node_input.node == nullptr && node_input.node_arg == nullptr) {
    vai_assert(false, "both node and node_arg are nullptr. not allowed");
  }
}

void graph_save(const Graph& graph, const std::string& filename, const std::string& filename_dat, size_t initializer_size_threshold) {
  auto& model = const_cast<Model&>(graph.GetModel());
  std::unique_ptr<ONNX_NAMESPACE::ModelProto> model_proto;

  if (initializer_size_threshold == std::numeric_limits<size_t>::max()) {
    model_proto = model.ToProto();
  } else {
    model_proto = model.ToGraphProtoWithExternalInitializers(filename_dat, graph.ModelPath().ToPathString(), initializer_size_threshold);
  }
  auto& metadata = model.MetaData();
  if (!metadata.empty()) {
    auto metadata_props = model_proto->mutable_metadata_props();
    metadata_props->Clear();
    for (auto& m : metadata) {
      auto prop = metadata_props->Add();
      *prop->mutable_key() = m.first;
      *prop->mutable_value() = m.second;
    }
  }
  // use relative path as data storage.
  auto graph_proto = model_proto->mutable_graph();
  *graph_proto = *graph.ToGraphProto();
  for (int i = 0; i < graph_proto->mutable_initializer()->size(); i++) {
    auto mutable_external_data = graph_proto->mutable_initializer()->at(i).mutable_external_data();
    for (int j = 0; j < mutable_external_data->size(); j++) {
      auto& external_data = mutable_external_data->at(j);
      if (*external_data.mutable_key() == "location")
        *external_data.mutable_value() = std::filesystem::path(*external_data.mutable_value()).filename().u8string();
    }
  }

  std::fstream output(filename, std::ios::out | std::ios::trunc | std::ios::binary);
  bool result = model_proto->SerializeToOstream(output);
  output << std::flush;
  vai_assert(result, "model serialize to ostream error");
}

Node& graph_fuse(Graph& graph, const std::string& name,
                 const std::string& op_type,
                 const std::vector<size_t>& nodes,
                 const std::vector<std::string>& inputs,
                 const std::vector<std::string>& outputs,
                 const std::vector<std::string>& constant_initializers) {
  auto meta_def = IndexedSubGraph_MetaDef::Create();
  meta_def->inputs() = inputs;
  meta_def->outputs() = outputs;
  meta_def->constant_initializers() = constant_initializers;
  meta_def->name() = "super_layer";
  meta_def->domain() = "com.xilinx";
  meta_def->since_version() = 1;
  meta_def->status() = ONNX_NAMESPACE::EXPERIMENTAL;

  auto indexed_subgraph = IndexedSubGraph::Create();
  indexed_subgraph->Nodes() = nodes;
  indexed_subgraph->SetMetaDef(std::move(meta_def));

  auto& fused_node = graph.FuseSubGraph(*indexed_subgraph, name);
  auto function_body = fused_node.GetFunctionBody();
  if (function_body) {
    auto proto = function_body->Body().ToGraphProto();
    *proto->mutable_name() = name;
    fused_node.AddAttribute("body", *proto);
  }
  for (auto&& o : fused_node.OutputDefs()) {
    graph.UpdateProducerNode(o->Name(), fused_node.Index());
  }
  return fused_node;
}
}  // namespace vaip
