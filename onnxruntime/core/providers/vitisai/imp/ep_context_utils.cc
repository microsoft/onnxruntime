// Standard headers/libs.
#include <fstream>
#include <filesystem>
#include <sstream>

#include "ep_context_utils.h"

namespace fs = std::filesystem;

namespace onnxruntime {

std::unique_ptr<ONNX_NAMESPACE::FunctionProto> ConvertIndexedSubGraphToFunctionProto(
    const IndexedSubGraph& sub_graph, const Graph& parent_graph) {
  auto p_func_proto = ONNX_NAMESPACE::FunctionProto::Create();
  auto* p_meta_def = const_cast<IndexedSubGraph_MetaDef*>(sub_graph.GetMetaDef());
  if (p_meta_def) {
    p_func_proto->set_name(p_meta_def->name());
    p_func_proto->set_domain(p_meta_def->domain());
    for (const auto& input : p_meta_def->inputs()) {
      p_func_proto->add_input(input);
    }
    auto* p_metadata_props_0 = p_func_proto->add_metadata_props();
    *(p_metadata_props_0->mutable_key()) = "meta_def_inputs_size";
    *(p_metadata_props_0->mutable_value()) = std::to_string(p_meta_def->inputs().size());
    for (const auto& output : p_meta_def->outputs()) {
      p_func_proto->add_output(output);
    }
    // XXX: SerDes with different fields.
    for (const auto& initializer : p_meta_def->constant_initializers()) {
      p_func_proto->add_input(initializer);
    }
    // XXX: SerDes with different numbers of fields.
    for (const auto& attr_pair : p_meta_def->attributes()) {
      p_func_proto->add_attribute(attr_pair.first);
      auto* p_attr_proto = p_func_proto->add_attribute_proto();
      *p_attr_proto = attr_pair.second;
    }
    p_func_proto->set_doc_string(p_meta_def->doc_string());
    // "since_version"
    auto* p_metadata_props_1 = p_func_proto->add_metadata_props();
    *(p_metadata_props_1->mutable_key()) = "meta_def_since_version";
    *(p_metadata_props_1->mutable_value()) = std::to_string(p_meta_def->since_version());
    // "status"
    auto* p_metadata_props_2 = p_func_proto->add_metadata_props();
    *(p_metadata_props_2->mutable_key()) = "meta_def_status";
    *(p_metadata_props_2->mutable_value()) = std::to_string(static_cast<int>(p_meta_def->status()));
    // TODO: `MetaDef::type_and_shape_inference_function`.
  }
  auto p_parent_graph_proto = parent_graph.ToGraphProto();
  for (auto node_index : const_cast<IndexedSubGraph&>(sub_graph).Nodes()) {
    auto* p_node_proto = p_parent_graph_proto->mutable_node(node_index);
    auto* p_attr_proto = p_node_proto->add_attribute();
    p_attr_proto->set_name("parent_graph_node_index");
    p_attr_proto->set_type(ONNX_NAMESPACE::AttributeProto::INT);
    p_attr_proto->set_i(node_index);
    *(p_func_proto->add_node()) = *p_node_proto;
  }
#if 0
  // Alternative.
  for (const auto node_index : sub_graph.Nodes()) {
    const auto* p_node = parent_graph.GetNode(node_index);
    auto p_node_proto = ONNX_NAMESPACE::NodeProto::Create();
    // XXX
    p_node->ToProto(*p_node_proto, true);
    p_attr_proto->set_name("parent_graph_node_index");
    p_attr_proto->set_type(ONNX_NAMESPACE::AttributeProto::INT);
    p_attr_proto->set_i(node_index);
    *(p_func_proto.add_node()) = *p_node_proto;
  }
#endif
  auto* p_metadata_props_3 = p_func_proto->add_metadata_props();
  *p_metadata_props_3->mutable_key() = "schema_source";
  *p_metadata_props_3->mutable_value() = std::to_string(static_cast<uint8_t>(sub_graph.GetSchemaSource()));
  return p_func_proto;
}

std::unique_ptr<IndexedSubGraph> ConvertFunctionProtoToIndexedSubGraph(
    const std::unique_ptr<ONNX_NAMESPACE::FunctionProto>& p_func_proto) {
  auto p_isg = IndexedSubGraph::Create();
  // "meta_def_inputs_size" (optional) and "schema_source".
  int func_metadata_props_size = p_func_proto->metadata_props_size();
  // Precisely, func_metadata_props_size == 4, which implies
  // `IndexedSubGraph::meta_def_` is not null and `IndexedSubGraph::nodes` > 1.
  if (func_metadata_props_size > 1) {
    auto& prop0 = const_cast<ONNX_NAMESPACE::StringStringEntryProto&>(p_func_proto->metadata_props(0));
    int isg_meta_def_inputs_size = std::stoi(*(prop0.mutable_value()));
    auto p_meta_def = IndexedSubGraph_MetaDef::Create();
    p_meta_def->name() = p_func_proto->name();
    p_meta_def->domain() = p_func_proto->domain();
    auto& prop1 = const_cast<ONNX_NAMESPACE::StringStringEntryProto&>(p_func_proto->metadata_props(1));
    p_meta_def->since_version() = std::stoi(*(prop1.mutable_value()));
    auto& prop2 = const_cast<ONNX_NAMESPACE::StringStringEntryProto&>(p_func_proto->metadata_props(2));
    p_meta_def->status() = static_cast<ONNX_NAMESPACE::OperatorStatus>(std::stoi(*(prop2.mutable_value())));
    auto& meta_def_inputs = p_meta_def->inputs();
    for (int i = 0; i < isg_meta_def_inputs_size; i++) {
      meta_def_inputs.push_back(p_func_proto->input(i));
    }
    auto& meta_def_outputs = p_meta_def->outputs();
    for (int i = 0, l = p_func_proto->output_size(); i < l; i++) {
      meta_def_outputs.push_back(p_func_proto->output(i));
    }
    auto& meta_def_initializers = p_meta_def->constant_initializers();
    for (int i = isg_meta_def_inputs_size, l = p_func_proto->input_size(); i < l; i++) {
      meta_def_initializers.push_back(p_func_proto->input(i));
    }
    auto& meta_def_attrs = p_meta_def->attributes();
    for (int i = 0, l = p_func_proto->attribute_size(); i < l; i++) {
      meta_def_attrs.emplace(p_func_proto->attribute(i), p_func_proto->attribute_proto(i));
    }
    p_meta_def->doc_string() = p_func_proto->doc_string();
    // TODO: `IndexedSubGraph::type_and_shape_inference_function`.
    p_isg->SetMetaDef(std::move(p_meta_def));
  }
  auto& isg_nodes = p_isg->Nodes();
  for (int i = 0, l = p_func_proto->node_size(); i < l; i++) {
    const auto& node_proto = p_func_proto->node(i);
    isg_nodes.push_back(node_proto.attribute(const_cast<ONNX_NAMESPACE::NodeProto&>(node_proto).attribute_size() - 1).i());
  }
  auto schema_source = static_cast<IndexedSubGraph_SourceOfSchema>(
      std::stoi(*(const_cast<ONNX_NAMESPACE::StringStringEntryProto&>(p_func_proto->metadata_props(func_metadata_props_size - 1)).mutable_value())));
  p_isg->SetSchemaSource(schema_source);
  return p_isg;
}

std::string SerializeCapabilities(
    const std::vector<std::unique_ptr<ComputeCapability>>& capability_ptrs,
    const Graph& graph) {
  std::stringstream ss;
  for (const auto& p : capability_ptrs) {
    auto& p_subgraph = p->SubGraph();
    auto p_func_proto = ConvertIndexedSubGraphToFunctionProto(*p_subgraph, graph);
    std::string func_proto_buf;
    p_func_proto->SerializeToString(func_proto_buf);
    size_t buf_len = func_proto_buf.length();
    ss.write(reinterpret_cast<const char*>(&buf_len), sizeof(buf_len));
    ss.write(func_proto_buf.data(), buf_len);
  }
  if (!ss.good()) {
    ORT_THROW("Serialization stream bad");
  }
  return ss.str();
}

void DeserializeCapabilities(const std::string& ser_capabilities,
                             std::vector<std::unique_ptr<ComputeCapability>>& capability_ptrs) {
  std::istringstream ss(ser_capabilities);
  while (!ss.eof()) {
    size_t buf_len;
    ss.read(reinterpret_cast<char*>(&buf_len), sizeof(buf_len));
    std::string buf(buf_len, '\0');
    ss.read(&buf[0], buf_len);
    auto p_func_proto = ONNX_NAMESPACE::FunctionProto::Create();
    p_func_proto->ParseFromString(buf);
    auto p_subgraph = ConvertFunctionProtoToIndexedSubGraph(p_func_proto);
    capability_ptrs.push_back(ComputeCapability::Create(std::move(p_subgraph)));
  }
}

std::unique_ptr<Model> CreateEPContexModel(
    const GraphViewer& graph_viewer,
    const std::string& serialized_ctx_cache,
    const std::string& ctx_cache_file_loc,
    const int64_t embed_mode,
    const logging::Logger* p_logger) {
  // Create a new graph/model, reusing the graph name,
  // the op-domain-to-opset-version map,
  // and the op schema registry of the current graph.
  auto& ep_ctx_graph = graph_viewer.CreateModel(*p_logger)->MainGraph();

  std::vector<NodeArg*> input_node_arg_ptrs;
  // XXX: vs `GraphViewer::GetInputsIncludingInitializers()`.
  for (const auto* p_node_arg : graph_viewer.GetInputs()) {
    auto& temp_node_arg = ep_ctx_graph.GetOrCreateNodeArg(
        p_node_arg->Name(), p_node_arg->TypeAsProto());
    input_node_arg_ptrs.push_back(&temp_node_arg);
  }
  std::vector<NodeArg*> output_node_arg_ptrs;
  for (const auto* p_node_arg : graph_viewer.GetOutputs()) {
    auto& temp_node_arg = ep_ctx_graph.GetOrCreateNodeArg(p_node_arg->Name(), p_node_arg->TypeAsProto());
    output_node_arg_ptrs.push_back(&temp_node_arg);
  }

  // Attr "embed_mode".
  auto p_attr_0 = ONNX_NAMESPACE::AttributeProto::Create();
  p_attr_0->set_name(kEmbedModeAttr);
  // p_attr_0->set_type(onnx::AttributeProto_AttributeType_INT);
  p_attr_0->set_type(ONNX_NAMESPACE::AttributeProto::INT);
  p_attr_0->set_i(embed_mode);
  // Attr "ep_cache_context".
  auto p_attr_1 = ONNX_NAMESPACE::AttributeProto::Create();
  p_attr_1->set_name(kEPCacheContextAttr);
  // p_attr_1->set_type(onnx::AttributeProto_AttributeType_STRING);
  p_attr_1->set_type(ONNX_NAMESPACE::AttributeProto::STRING);
  p_attr_1->set_s(embed_mode == 0 ? ctx_cache_file_loc : serialized_ctx_cache);
  // Attr "source".
  auto p_attr_2 = ONNX_NAMESPACE::AttributeProto::Create();
  p_attr_2->set_name(kSourceAttr);
  // p_attr_2->set_type(onnx::AttributeProto_AttributeType_STRING);
  p_attr_2->set_type(ONNX_NAMESPACE::AttributeProto::STRING);
  p_attr_2->set_s(kVitisAIExecutionProvider);

  auto p_node_attrs = NodeAttributes::Create();
  constexpr int num_attrs = 3;
  p_node_attrs->reserve(num_attrs);
  p_node_attrs->emplace(kEmbedModeAttr, *p_attr_0);
  p_node_attrs->emplace(kEPCacheContextAttr, *p_attr_1);
  p_node_attrs->emplace(kSourceAttr, *p_attr_2);

  ep_ctx_graph.AddNode(kEPContextOp, kEPContextOp, "", input_node_arg_ptrs, output_node_arg_ptrs, p_node_attrs.get(), kEPContextOpDomain);
  ORT_ENFORCE(ep_ctx_graph.Resolve().IsOK());
  auto p_ep_ctx_graph_viewer = ep_ctx_graph.CreateGraphViewer();
  auto p_ep_ctx_model = p_ep_ctx_graph_viewer->CreateModel(*p_logger);
  auto p_ep_ctx_model_proto = p_ep_ctx_model->ToProto();
  p_ep_ctx_model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  p_ep_ctx_graph_viewer->ToProto(*(p_ep_ctx_model_proto->mutable_graph()), true, true);

  return p_ep_ctx_model;
}

void DumpEPContextModel(
    const std::unique_ptr<Model>& p_model, const std::string& ep_ctx_model_file_loc) {
  std::fstream dump_stream(ep_ctx_model_file_loc, std::ios::out | std::ios::trunc | std::ios::binary);
  p_model->ToProto()->SerializeToOstream(dump_stream);
  LOGS_DEFAULT(VERBOSE) << "[VitisAI EP] Dumped " << ep_ctx_model_file_loc;
}

bool ValidateEPContextNode(const Graph& graph) {
  // TODO: Support for multi-node EP context model.
  assert(graph.Nodes().size() == 1);
  auto* p_node = graph.GetNode(0);
  assert(p_node->OpType() == kEPContextOp);
  auto& attrs = p_node->GetAttributes();
  assert(attrs.count(kEmbedModeAttr) > 0);
  assert(attrs.count(kEPCacheContextAttr) > 0);
  assert(attrs.count(kSourceAttr) > 0);
  (void)attrs;
  return true;
}

std::string RetrieveEPContextCache(const Graph& graph) {
  if (!ValidateEPContextNode(graph)) {
    ORT_THROW("Invalid EP context model for Vitis AI");
  }
  // TODO: Support for multi-node EP context model.
  auto* p_node = graph.GetNode(0);
  const auto& attrs = p_node->GetAttributes();
  int64_t embed_mode = attrs.at(kEmbedModeAttr).i();
  const std::string& ep_ctx_cache = attrs.at(kEPCacheContextAttr).s();
  if (embed_mode) {
    return ep_ctx_cache;
  }
  fs::path ep_ctx_file_loc(ep_ctx_cache);
  // TODO: Validaion of the file location to make sure security is met.
  if (!fs::exists(ep_ctx_file_loc) || !fs::is_regular_file(ep_ctx_file_loc)) {
    ORT_THROW("File for EP context cache is missing");
  }
  std::ifstream ifs(ep_ctx_cache, std::ios::binary | std::ios::in);
  if (!ifs.is_open()) {
    ORT_THROW("Exception opening EP context cache file");
  }
  ifs.seekg(0, ifs.end);
  int cache_len = ifs.tellg();
  ifs.seekg(0, ifs.beg);
  char* buf = new char[cache_len];
  ifs.read(buf, cache_len);
  if (!ifs.good()) {
    ifs.close();
    ORT_THROW("Exception reading EP context cache file");
  }
  ifs.close();
  std::string cache_payload(buf);
  delete[] buf;
  return cache_payload;
}

bool GraphHasEPContextNode(const GraphViewer& graph_viewer) {
  for (size_t i = 0, l = static_cast<size_t>(graph_viewer.MaxNodeIndex()); i < l; i++) {
    auto* p_node = graph_viewer.GetNode(i);
    if (p_node != nullptr && p_node->OpType() == kEPContextOp) {
      const auto& attrs = p_node->GetAttributes();
      if (attrs.count(kSourceAttr) > 0 && attrs.at(kSourceAttr).s() == kVitisAIExecutionProvider) {
        return true;
      }
    }
  }
  return false;
}

bool FusedGraphHasEPContextNode(
    const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs) {
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    bool has_node = GraphHasEPContextNode(fused_node_graph.filtered_graph);
    if (has_node) {
      return true;
    }
  }
  return false;
}

const Path& GetTopLevelModelPath(const GraphViewer& graph_viewer) {
  const auto& graph = graph_viewer.GetGraph();
  const Graph* p_graph = &graph;
  while (p_graph->IsSubgraph()) {
    p_graph = p_graph->ParentGraph();
  }
  return p_graph->ModelPath();
}

bool GetEPContextModelFileLocation(
    const std::string& ep_ctx_model_path_cfg,
    const PathString& model_path_str,
    bool is_ep_ctx_model,
    PathString& ep_ctx_model_file_loc) {
  // if (!ep_ctx_model_file_loc.empty()) {
  //   return true;
  // }
  if (!ep_ctx_model_path_cfg.empty()) {
    ep_ctx_model_file_loc = ToPathString(ep_ctx_model_path_cfg);
  } else if (!model_path_str.empty()) {
    if (is_ep_ctx_model) {
      ep_ctx_model_file_loc = model_path_str;
    } else {
      ep_ctx_model_file_loc =
          ToPathString(fs::path(model_path_str).stem().string() + "_ctx.onnx");
    }
  }
  return !ep_ctx_model_file_loc.empty() && fs::exists(ep_ctx_model_file_loc) && fs::is_regular_file(ep_ctx_model_file_loc);
}

// The file for EP context binary is in the same folder as the EP context model file.
PathString GetEPContextCacheFileLocation(
    const PathString& ep_ctx_model_file_loc, const PathString& model_path_str) {
  if (!ep_ctx_model_file_loc.empty()) {
    fs::path ep_ctx_model_fs_path(ep_ctx_model_file_loc);
    auto ep_ctx_cache_fs_path =
        ep_ctx_model_fs_path.replace_extension(fs::path("__ep_ctx_cache.bin"));
    return ToPathString(ep_ctx_cache_fs_path.string());
  }
  fs::path model_fs_path(model_path_str);
  auto ep_ctx_cache_fs_path =
      model_fs_path.replace_extension(fs::path("__ep_ctx_cache.bin"));
  return ToPathString(ep_ctx_cache_fs_path.string());
}

}  // namespace onnxruntime
