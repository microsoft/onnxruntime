// Standard headers/libs.
#include <fstream>
#include <filesystem>
#include <sstream>

// 3rd-party headers/libs.
#include <nlohmann/json.hpp>
#include "./md5.h"

#include "ep_context_utils.h"

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
    *(p_metadata_props_2->mutable_value()) =
        std::to_string(static_cast<int>(p_meta_def->status()));
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
  *(p_metadata_props_3->mutable_key()) = "schema_source";
  *(p_metadata_props_3->mutable_value()) =
      std::to_string(static_cast<uint8_t>(sub_graph.GetSchemaSource()));
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
    isg_nodes.push_back(
        node_proto.attribute(const_cast<ONNX_NAMESPACE::NodeProto&>(node_proto).attribute_size() - 1).i());
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

std::string SerializeOrigialGraph(const GraphViewer& graph_viewer) {
  // XXX: Will Steps 1/2/3 suffice for restoring a model/graph later?
  // Any information loss or mismatch?
  // Step 1
  const Graph& orig_graph = graph_viewer.GetGraph();
  // Step 2
  const Model& orig_model = orig_graph.GetModel();
  // Step 3
  auto p_orig_model_proto = const_cast<Model&>(orig_model).ToProto();
  std::string ser_buf;
  p_orig_model_proto->SerializeToString(ser_buf);

  nlohmann::json j_obj;
  j_obj["orig_graph_name"] = graph_viewer.Name();
  j_obj["orig_model_path"] = PathToUTF8String(graph_viewer.ModelPath().ToPathString());
  j_obj["orig_model_proto_ser_str"] = ser_buf;
  return j_obj.dump();
}

std::unique_ptr<Model> CreateEPContexModel(
    const GraphViewer& graph_viewer,
    const std::string& serialized_ctx_cache,
    const std::string& ctx_cache_file_loc,
    const int64_t embed_mode,
    bool saving_orig_graph,
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
  // Relative to the ONNX model file.
  p_attr_1->set_s(
      embed_mode == 0 ? fs::path(ctx_cache_file_loc).filename().string() : serialized_ctx_cache);
  // Attr "source".
  auto p_attr_2 = ONNX_NAMESPACE::AttributeProto::Create();
  p_attr_2->set_name(kSourceAttr);
  // p_attr_2->set_type(onnx::AttributeProto_AttributeType_STRING);
  p_attr_2->set_type(ONNX_NAMESPACE::AttributeProto::STRING);
  p_attr_2->set_s(kVitisAIExecutionProvider);
  // Attr "onnx_model_filename".
  auto p_attr_3 = ONNX_NAMESPACE::AttributeProto::Create();
  p_attr_3->set_name(kONNXModelFileNameAttr);
  // p_attr_3->set_type(onnx::AttributeProto_AttributeType_STRING);
  p_attr_3->set_type(ONNX_NAMESPACE::AttributeProto::STRING);
  p_attr_3->set_s(fs::path(graph_viewer.ModelPath().ToPathString()).filename().string());
  // Attr "notes".
  auto p_attr_4 = ONNX_NAMESPACE::AttributeProto::Create();
  p_attr_4->set_name(kNotesAttr);
  // p_attr_4->set_type(onnx::AttributeProto_AttributeType_STRING);
  p_attr_4->set_type(ONNX_NAMESPACE::AttributeProto::STRING);
  // FIXME: 2G-limit of ProtoBuf.
  p_attr_4->set_s(saving_orig_graph ? SerializeOrigialGraph(graph_viewer) : "N/A");

  auto p_node_attrs = NodeAttributes::Create();
  constexpr int num_attrs = 5;
  p_node_attrs->reserve(num_attrs);
  p_node_attrs->emplace(kEmbedModeAttr, *p_attr_0);
  p_node_attrs->emplace(kEPCacheContextAttr, *p_attr_1);
  p_node_attrs->emplace(kSourceAttr, *p_attr_2);
  p_node_attrs->emplace(kONNXModelFileNameAttr, *p_attr_3);
  p_node_attrs->emplace(kNotesAttr, *p_attr_4);

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

std::string RetrieveEPContextCache(
    const Graph& graph, const PathString& ep_ctx_model_loc, bool binary_mode) {
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
  fs::path ep_ctx_fs_path(ep_ctx_model_loc);
  // Attr "ep_cache_context" stores a relative path.
  ep_ctx_fs_path.replace_filename(fs::path(ep_ctx_cache));
  // TODO: Validaion of the file location to make sure security is met.
  if (!fs::exists(ep_ctx_fs_path) || !fs::is_regular_file(ep_ctx_fs_path)) {
    ORT_THROW("File for EP context cache is missing");
  }
  auto open_mode = binary_mode ? (std::ios::in | std::ios::binary) : std::ios::in;
  std::ifstream ifs(ep_ctx_cache, open_mode);
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

std::unique_ptr<GraphViewer> RetrieveOriginalGraph(const Graph& ep_ctx_graph) {
  if (!ValidateEPContextNode(ep_ctx_graph)) {
    ORT_THROW("Invalid EP context model for Vitis AI");
  }
  // TODO: Support for multi-node EP context model.
  auto* p_node = ep_ctx_graph.GetNode(0);
  const auto& attrs = p_node->GetAttributes();
  const auto& notes_str = attrs.at(kNotesAttr).s();
  nlohmann::json j_obj = nlohmann::json::parse(notes_str);

  auto& logger = logging::LoggingManager::DefaultLogger();

  auto p_model_proto = ONNX_NAMESPACE::ModelProto::Create();
  p_model_proto->ParseFromString(j_obj["orig_model_proto_ser_str"]);
  auto p_model = Model::Create(std::move(*p_model_proto), j_obj["orig_model_path"], nullptr, logger);
  auto& graph = p_model->MainGraph();
  // XXX: maybe ineffective.
  graph.ToGraphProto()->set_name(j_obj["orig_graph_name"]);

  return p_model->MainGraph().CreateGraphViewer();
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
      fs::path model_fs_path(model_path_str);
      fs::path ep_ctx_model_fs_path(model_fs_path.parent_path() / model_fs_path.stem());
      ep_ctx_model_fs_path += fs::path("_ctx.onnx");
      ep_ctx_model_file_loc = ToPathString(ep_ctx_model_fs_path.string());
    }
  }
  // return !ep_ctx_model_file_loc.empty() && fs::exists(ep_ctx_model_file_loc) && fs::is_regular_file(ep_ctx_model_file_loc);
  return !ep_ctx_model_file_loc.empty();
}

// The file for EP context cache is in the same folder as the EP context model file.
PathString GetEPContextCacheFileLocation(
    const PathString& ep_ctx_model_file_loc, const PathString& model_path_str) {
  if (!ep_ctx_model_file_loc.empty()) {
    fs::path ep_ctx_model_fs_path(ep_ctx_model_file_loc);
    fs::path ep_ctx_cache_fs_path(ep_ctx_model_fs_path.parent_path() / ep_ctx_model_fs_path.stem());
    ep_ctx_cache_fs_path += fs::path("__ep_ctx_cache.bin");
    return ToPathString(ep_ctx_cache_fs_path.string());
  }
  fs::path model_fs_path(model_path_str);
  fs::path ep_ctx_cache_fs_path(model_fs_path.parent_path() / model_fs_path.stem());
  ep_ctx_cache_fs_path += fs::path("__ep_ctx_cache.bin");
  return ToPathString(ep_ctx_cache_fs_path.string());
}

std::string Slurp(const fs::path& file_location) {
  // const char* location_str = file_location.u8string().c_str();
  const char* location_str = file_location.c_str();
  std::ifstream ifs;
  ifs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  std::stringstream ss;
  try {
    ifs.open(location_str, std::ifstream::in);
    ss << ifs.rdbuf();
    if (!ss.good()) {
      LOGS_DEFAULT(WARNING) << "Failed to write to stream";
    }
    ifs.close();
  } catch (std::system_error& se) {
    LOGS_DEFAULT(WARNING) << "Failed to read " << location_str << ". " << se.code().message();
  }
  return ss.str();
}

std::string GetBackendCompileCache(const fs::path& backend_cache_file_location) {
  if (!fs::exists(backend_cache_file_location) || !fs::is_regular_file(backend_cache_file_location)) {
    LOGS_DEFAULT(WARNING) << "[VitisAI EP]Bad file for compilation cache: "
                          << backend_cache_file_location;
    return "";
  }
  LOGS_DEFAULT(VERBOSE) << "Reading backend compilation cache";
  return Slurp(backend_cache_file_location);
}

void RestoreBackendCompileCache(
    const fs::path& backend_cache_file_location, const std::string& compile_cache) {
  std::ofstream ofs(backend_cache_file_location, std::ios::out | std::ios::trunc);
  if (!ofs.is_open()) {
    LOGS_DEFAULT(WARNING) << "[VitisAI EP]Failed to open a file for restoring backend compilation cache: "
                          << backend_cache_file_location;
    return;
  }
  ofs.write(compile_cache.data(), compile_cache.length());
  if (!ofs.good()) {
    ofs.close();
    LOGS_DEFAULT(WARNING) << "[VitisAI EP]Failed to restore backend compilation cache: "
                          << backend_cache_file_location;
    return;
  }
  LOGS_DEFAULT(VERBOSE) << "[VitisAI EP]Succeeded to restore backend compilation cache: "
                        << backend_cache_file_location;
}

#if 0
// TODO: To be removed.
// `onnxruntime::GraphViewer::GetNodesInTopologicalOrder()`
static std::vector<NodeIndex> GetNodeIndicesInTopologicalOrder(const GraphViewer& graph_viewer) {
  const auto& node_arg_ptrs = graph_viewer.GetOutputs();
  std::vector<const Node*> leaf_node_ptrs;
  leaf_node_ptrs.reserve(node_arg_ptrs.size());
  for (const auto* p : node_arg_ptrs) {
    if (p != nullptr) {
      auto* p_node = graph_viewer.GetProducerNode(p->Name());
      if (p_node != nullptr) {
        leaf_node_ptrs.push_back(p_node);
      }
    }
  }
  std::vector<NodeIndex> node_indices;
  const auto& graph = graph_viewer.GetGraph();
  graph.ReverseDFSFrom(
      leaf_node_ptrs,  // from
      nullptr,         // enter
      [&node_indices](const Node* p_node) mutable {
        node_indices.push_back(p_node->Index());
      },        // leave
      nullptr,  // comp
      nullptr   // stop
  );
  return node_indices;
}
#endif

std::vector<const NodeArg*> FilterOutputNodeArgs(const Node& node) {
  auto node_arg_ptrs = node.OutputDefs();
  auto num_ptrs = node_arg_ptrs.size();
  std::vector<const NodeArg*> res(num_ptrs);
  for (auto i = 0u; i < num_ptrs; i++) {
    // Some operators have outputs that are optional.
    // When an actual output parameter of an operator is not specified,
    // the operator implementation MAY forgo computing values for such outputs.
    // There are two ways to leave an optional input or output unspecified:
    // the first, available only for trailing inputs and outputs, is to simply not provide that input;
    // the second is to use an empty string in place of an input or output name.
    // So optional output maybe output != null && false output->Exists().
    // Our processing: nullptr means an optional output, and client code needs to handle nullptr.
    assert(node_arg_ptrs[i] != nullptr);
    if (node_arg_ptrs[i]->Exists()) {
      res[i] = node_arg_ptrs[i];
    } else {
      res[i] = nullptr;
    }
  }
  return res;
}

std::vector<int64_t> GetNodeArgShape_I64(const NodeArg& node_arg) {
  const auto* p_shape_proto = node_arg.Shape();
  if (p_shape_proto == nullptr) {
    return std::vector<int64_t>();
  }
  int num_dims = p_shape_proto->dim_size();
  std::vector<int64_t> shape_vec;
  shape_vec.reserve(num_dims);
  for (auto i = 0; i < num_dims; i++) {
    auto& curr_dim = p_shape_proto->dim(i);
    shape_vec.push_back(curr_dim.has_dim_value() ? curr_dim.dim_value() : (int64_t)-1);
  }
  return shape_vec;
}

std::string GetModelSignature(const GraphViewer& graph_viewer) {
  MD5 md5_obj;
  for (auto ni : graph_viewer.GetNodesInTopologicalOrder()) {
    auto* p_node = graph_viewer.GetNode(ni);
    for (const auto* p_node_arg : FilterOutputNodeArgs(*p_node)) {
      auto node_arg_name = p_node_arg->Name();
      md5_obj.add(node_arg_name.data(), node_arg_name.length());
      auto node_arg_shape = GetNodeArgShape_I64(*p_node_arg);
      if (!node_arg_shape.empty()) {
        md5_obj.add(node_arg_shape.data(), node_arg_shape.size() * sizeof(node_arg_shape.at(0)));
      }
    }
  }
  return md5_obj.getHash();
}

std::string HashFileContentWithMD5(const std::string& file_location) {
  std::ifstream ifs(file_location, std::ios::in | std::ios::binary);
  if (!ifs.is_open()) {
    LOGS_DEFAULT(ERROR) << "Failed to open file for checksum: " << file_location;
    ORT_THROW("Failed to open file for checksum");
  }
  constexpr const std::uint32_t kBufferSize = 1024;
  char* buffer = new char[kBufferSize];
  MD5 md5_obj;
  while (ifs.read(buffer, kBufferSize)) {
    md5_obj.add(buffer, kBufferSize);
  }
  auto num_last_read = ifs.gcount();
  md5_obj.add(buffer, num_last_read);
  delete[] buffer;
  ifs.close();
  return md5_obj.getHash();
}

}  // namespace onnxruntime
