// Standard headers/libs.
#include <fstream>
#include <sstream>
#include <cctype>
#include <cstring>

// 3rd-party headers/libs.
#include <nlohmann/json.hpp>

#include "ep_context_utils.h"

namespace onnxruntime {

constexpr const char* kVitisAI = "vitisai";

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
    auto* p_node_proto = p_parent_graph_proto->mutable_node(static_cast<int>(node_index));
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
    auto* p_attr_proto = p_node_proto->add_attribute();
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
  if (p_orig_model_proto->opset_import_size() == 0) {
    for (const auto& it : graph_viewer.DomainToVersionMap()) {
      auto* p_opset_import = p_orig_model_proto->add_opset_import();
      *(p_opset_import->mutable_domain()) = it.first;
      p_opset_import->set_version(it.second);
    }
  }

  nlohmann::json j_obj;
  if (p_orig_model_proto->opset_import_size() > 0) {
    for (int i = 0, n = p_orig_model_proto->opset_import_size(); i < n; ++i) {
      auto& op_set_id_proto = const_cast<ONNX_NAMESPACE::OperatorSetIdProto&>(p_orig_model_proto->opset_import(i));
      j_obj[*op_set_id_proto.mutable_domain()] = std::to_string(op_set_id_proto.version());
    }
  }
  j_obj["orig_graph_name"] = graph_viewer.Name();
  // TODO: platform dependency (Linux vs Windows).
  j_obj["orig_model_path"] = graph_viewer.ModelPath().string();

  // XXX: `ModelProto::SerializeToString` will lose some info,
  // e.g., ModelProto.opset_import.
  std::string ser_buf;
  p_orig_model_proto->SerializeToString(ser_buf);
  j_obj["orig_model_proto_ser_str"] = ser_buf;

  return j_obj.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

// Ref.: `CreateEpContextModel()` in the file "graph_partitioner.cc".
ONNX_NAMESPACE::ModelProto* CreateEPContexModel(
    const GraphViewer& graph_viewer,
    const std::string& serialized_ctx_cache,
    const std::string& ctx_cache_file_loc,
    const int64_t embed_mode,
    const std::string& backend_cache_dir,
    const std::string& backend_cache_key,
    bool saving_orig_graph,
    const logging::Logger* p_logger) {
  LOGS_DEFAULT(VERBOSE) << "[VitisAI EP]Creating EP context node";
  // Create a new graph/model, reusing the graph name,
  // the op-domain-to-opset-version map,
  // and the op schema registry of the current graph.
  // XXX: This approach (immediately below) has a memory fault issue (std::bad_alloc).
  // auto& ep_ctx_graph = graph_viewer.CreateModel(*p_logger)->MainGraph();
  // This apporach (immediately below) has no memory falut issue.
  auto p_temp_model = graph_viewer.CreateModel(*p_logger);
  auto& ep_ctx_graph = p_temp_model->MainGraph();

  const auto& graph_inputs = graph_viewer.GetInputs();
  std::vector<NodeArg*> input_node_arg_ptrs;
  input_node_arg_ptrs.reserve(graph_inputs.size());
  // XXX: vs `GraphViewer::GetInputsIncludingInitializers()`.
  for (const auto* p_node_arg : graph_inputs) {
    auto& temp_node_arg = ep_ctx_graph.GetOrCreateNodeArg(
        p_node_arg->Name(), p_node_arg->TypeAsProto());
    input_node_arg_ptrs.push_back(&temp_node_arg);
  }
  const auto& graph_outputs = graph_viewer.GetOutputs();
  std::vector<NodeArg*> output_node_arg_ptrs;
  output_node_arg_ptrs.reserve(graph_outputs.size());
  for (const auto* p_node_arg : graph_outputs) {
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
  p_attr_3->set_s(graph_viewer.ModelPath().filename().string());
  // Attr "notes".
  auto p_attr_4 = ONNX_NAMESPACE::AttributeProto::Create();
  p_attr_4->set_name(kNotesAttr);
  // p_attr_4->set_type(onnx::AttributeProto_AttributeType_STRING);
  p_attr_4->set_type(ONNX_NAMESPACE::AttributeProto::STRING);
  // FIXME: 2G-limit of ProtoBuf.
  if (saving_orig_graph) {
    p_attr_4->set_s(SerializeOrigialGraph(graph_viewer));
  } else {
    nlohmann::json j_obj;
    j_obj["backend_cache_dir"] = backend_cache_dir;
    j_obj["backend_cache_key"] = backend_cache_key;
    p_attr_4->set_s(j_obj.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace));
  }

  auto p_node_attrs = NodeAttributes::Create();
  constexpr int num_attrs = 5;
  p_node_attrs->reserve(num_attrs);
  p_node_attrs->emplace(kEmbedModeAttr, *p_attr_0);
  p_node_attrs->emplace(kEPCacheContextAttr, *p_attr_1);
  p_node_attrs->emplace(kSourceAttr, *p_attr_2);
  p_node_attrs->emplace(kONNXModelFileNameAttr, *p_attr_3);
  p_node_attrs->emplace(kNotesAttr, *p_attr_4);

  // Since we don't implement `IExecutionProvider::GetEpContextNodes()` and
  // thus don't leverage `CreateEpContextModel()` in the file "graph_partitioner.cc",
  // we specify a brand-new node name here.
  ep_ctx_graph.AddNode(kEPContextOpName, kEPContextOp, "", input_node_arg_ptrs, output_node_arg_ptrs, p_node_attrs.get(), kEPContextOpDomain);

  auto res_status = ep_ctx_graph.Resolve();
  ORT_ENFORCE(res_status.IsOK(), res_status.ErrorMessage());
  LOGS_DEFAULT(VERBOSE) << "Created EP context model graph resolved";

  auto p_ep_ctx_graph_viewer = ep_ctx_graph.CreateGraphViewer();
  auto p_temp_model_2 = p_ep_ctx_graph_viewer->CreateModel(*p_logger);
  auto p_ep_ctx_model_proto = p_temp_model_2->ToProto();
  p_ep_ctx_graph_viewer->ToProto(*p_ep_ctx_model_proto->mutable_graph(), true, true);
  p_ep_ctx_model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  return p_ep_ctx_model_proto.release();
}

// Ref.: `static common::Status Save(Model& model, int fd)` in the file "model.h".
void DumpEPContextModel(
    const std::unique_ptr<ONNX_NAMESPACE::ModelProto>& p_model_proto, const std::string& ep_ctx_model_file_loc) {
  std::fstream dump_stream(ep_ctx_model_file_loc, std::ios::out | std::ios::trunc | std::ios::binary);
  p_model_proto->SerializeToOstream(dump_stream);
  LOGS_DEFAULT(VERBOSE) << "[VitisAI EP] Dumped " << ep_ctx_model_file_loc;
}

const Node* GetEPContextNodePtr(const Graph& graph) {
  // TODO: Support for multi-node EP context model.
  for (const auto* p_node : graph.Nodes()) {
    if (p_node->OpType() == kEPContextOp) {
      return p_node;
    }
  }
  return nullptr;
}

bool ValidateEPContextNode(const Graph& graph) {
  // TODO: Support for multi-node EP context model.
  const auto* p_node = GetEPContextNodePtr(graph);
  assert(p_node != nullptr);
  auto& attrs = p_node->GetAttributes();
  assert(attrs.count(kEmbedModeAttr) > 0);
  assert(attrs.count(kEPCacheContextAttr) > 0);
  assert(attrs.count(kSourceAttr) > 0);
  const auto& source_val = attrs.at(kSourceAttr).s();
  if (source_val == kVitisAIExecutionProvider) {
    return true;
  }
  size_t vitisai_len = std::strlen(kVitisAI);
  assert(source_val.length() == vitisai_len);
  for (size_t i = 0; i < vitisai_len; ++i) {
    assert(static_cast<unsigned char>(std::tolower(source_val[i])) == kVitisAI[i]);
  }
  return true;
}

// Ref.: `CreateEpContextModel()` in the file "graph_partitioner.cc".
void CreateEPContexNodes(
    Graph* p_ep_ctx_graph,
    const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs,
    const std::string& serialized_ctx_cache,
    const std::string& ctx_cache_file_loc,
    const int64_t embed_mode,
    const std::string& backend_cache_dir,
    const std::string& backend_cache_key,
    bool saving_orig_graph,
    const logging::Logger* p_logger) {
  LOGS_DEFAULT(VERBOSE) << "[VitisAI EP]Creating EP context nodes";
  int fused_index = 0;
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    Node& fused_node = fused_node_graph.fused_node;
    const auto& fused_name = fused_node.Name();
    const GraphViewer& graph_viewer = fused_node_graph.filtered_graph;
    // FIXME
    const auto& graph_inputs = graph_viewer.GetInputs();
    std::vector<NodeArg*> input_node_arg_ptrs;
    input_node_arg_ptrs.reserve(graph_inputs.size());
    // XXX: vs `GraphViewer::GetInputsIncludingInitializers()`.
    for (const auto* p_node_arg : graph_inputs) {
      auto& temp_node_arg = p_ep_ctx_graph->GetOrCreateNodeArg(
          p_node_arg->Name(), p_node_arg->TypeAsProto());
      input_node_arg_ptrs.push_back(&temp_node_arg);
    }
    const auto& graph_outputs = graph_viewer.GetOutputs();
    std::vector<NodeArg*> output_node_arg_ptrs;
    output_node_arg_ptrs.reserve(graph_outputs.size());
    for (const auto* p_node_arg : graph_outputs) {
      auto& temp_node_arg = p_ep_ctx_graph->GetOrCreateNodeArg(p_node_arg->Name(), p_node_arg->TypeAsProto());
      output_node_arg_ptrs.push_back(&temp_node_arg);
    }

    auto p_node_attrs = NodeAttributes::Create();
    if (fused_index == 0) {
      p_node_attrs->reserve(7);
      // Attr "ep_cache_context".
      auto p_attr_1 = ONNX_NAMESPACE::AttributeProto::Create();
      p_attr_1->set_name(kEPCacheContextAttr);
      p_attr_1->set_type(ONNX_NAMESPACE::AttributeProto::STRING);
      // Relative to the ONNX model file.
      p_attr_1->set_s(
          embed_mode == 0 ? fs::path(ctx_cache_file_loc).filename().string() : serialized_ctx_cache);
      p_node_attrs->emplace(kEPCacheContextAttr, *p_attr_1);
      // Attr "notes".
      auto p_attr_4 = ONNX_NAMESPACE::AttributeProto::Create();
      p_attr_4->set_name(kNotesAttr);
      p_attr_4->set_type(ONNX_NAMESPACE::AttributeProto::STRING);
      // FIXME: 2G-limit of ProtoBuf.
      if (saving_orig_graph) {
        p_attr_4->set_s(SerializeOrigialGraph(graph_viewer));
      } else {
        nlohmann::json j_obj;
        j_obj["backend_cache_dir"] = backend_cache_dir;
        j_obj["backend_cache_key"] = backend_cache_key;
        p_attr_4->set_s(j_obj.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace));
      }
      p_node_attrs->emplace(kNotesAttr, *p_attr_4);
      // Attr "main_context".
      auto p_attr_5 = ONNX_NAMESPACE::AttributeProto::Create();
      p_attr_5->set_name(kMainContextAttr);
      p_attr_5->set_type(ONNX_NAMESPACE::AttributeProto::INT);
      p_attr_5->set_i(1);
      p_node_attrs->emplace(kMainContextAttr, *p_attr_5);
    } else {
      p_node_attrs->reserve(5);
      // Attr "main_context".
      auto p_attr_5 = ONNX_NAMESPACE::AttributeProto::Create();
      p_attr_5->set_name(kMainContextAttr);
      p_attr_5->set_type(ONNX_NAMESPACE::AttributeProto::INT);
      p_attr_5->set_i(0);
      p_node_attrs->emplace(kMainContextAttr, *p_attr_5);
    }
    // Attr "embed_mode".
    auto p_attr_0 = ONNX_NAMESPACE::AttributeProto::Create();
    p_attr_0->set_name(kEmbedModeAttr);
    p_attr_0->set_type(ONNX_NAMESPACE::AttributeProto::INT);
    p_attr_0->set_i(embed_mode);
    p_node_attrs->emplace(kEmbedModeAttr, *p_attr_0);
    // Attr "source".
    auto p_attr_2 = ONNX_NAMESPACE::AttributeProto::Create();
    p_attr_2->set_name(kSourceAttr);
    p_attr_2->set_type(ONNX_NAMESPACE::AttributeProto::STRING);
    p_attr_2->set_s(kVitisAIExecutionProvider);
    p_node_attrs->emplace(kSourceAttr, *p_attr_2);
    // Attr "onnx_model_filename".
    auto p_attr_3 = ONNX_NAMESPACE::AttributeProto::Create();
    p_attr_3->set_name(kONNXModelFileNameAttr);
    p_attr_3->set_type(ONNX_NAMESPACE::AttributeProto::STRING);
    p_attr_3->set_s(graph_viewer.ModelPath().filename().string());
    p_node_attrs->emplace(kONNXModelFileNameAttr, *p_attr_3);
    // Attr "partition_name".
    auto p_attr_6 = ONNX_NAMESPACE::AttributeProto::Create();
    p_attr_6->set_name(kPartitionNameAttr);
    p_attr_6->set_type(ONNX_NAMESPACE::AttributeProto::STRING);
    p_attr_6->set_s(fused_name);
    p_node_attrs->emplace(kPartitionNameAttr, *p_attr_6);

    p_ep_ctx_graph->AddNode(fused_name, kEPContextOp, "", input_node_arg_ptrs, output_node_arg_ptrs, p_node_attrs.get(), kEPContextOpDomain);

    ++fused_index;
  }
  auto res_status = p_ep_ctx_graph->Resolve();
  ORT_ENFORCE(res_status.IsOK(), res_status.ErrorMessage());
  LOGS_DEFAULT(VERBOSE) << "Created EP context model graph resolved";
}

std::string RetrieveEPContextCache(
    const Graph& graph, const PathString& ep_ctx_model_loc, bool binary_mode) {
  // TODO: Support for multi-node EP context model.
  const auto* p_node = GetEPContextNodePtr(graph);
  const auto& attrs = p_node->GetAttributes();
  int64_t embed_mode = attrs.at(kEmbedModeAttr).i();
  const std::string& ep_ctx_cache = attrs.at(kEPCacheContextAttr).s();
  if (embed_mode) {
    return ep_ctx_cache;
  }
  fs::path ep_ctx_fs_path(ep_ctx_model_loc);
  // Attr "ep_cache_context" stores a relative path.
  ep_ctx_fs_path.replace_filename(fs::path(ep_ctx_cache));
  // TODO: Validation of the file location to make sure security is met.
  if (!fs::exists(ep_ctx_fs_path) || !fs::is_regular_file(ep_ctx_fs_path)) {
    ORT_THROW("File for EP context cache is missing");
  }
  auto open_mode = binary_mode ? (std::ios::in | std::ios::binary) : std::ios::in;
  std::ifstream ifs(ep_ctx_fs_path.string().c_str(), open_mode);
  if (!ifs.is_open()) {
    ORT_THROW("Exception opening EP context cache file");
  }
  ifs.seekg(0, ifs.end);
  std::streampos cache_len = ifs.tellg();
  if (cache_len == -1) {
    ifs.close();
    ORT_THROW("Error when operating EP context cache file");
  } else if (cache_len == 0) {
    ifs.close();
    LOGS_DEFAULT(WARNING) << "Empty EP context cache file: " << ep_ctx_fs_path.string();
    return "";
  }
  ifs.seekg(0, ifs.beg);
  char* buf = new char[static_cast<size_t>(cache_len)];
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

void RetrieveBackendCacheInfo(const Graph& graph, std::string& cache_dir, std::string& cache_key) {
  // TODO: Support for multi-node EP context model.
  const auto* p_node = GetEPContextNodePtr(graph);
  if (p_node == nullptr) {
    LOGS_DEFAULT(WARNING) << "Failed to retrieve cache info due to no EP context nodes";
    return;
  }
  const auto& attrs = p_node->GetAttributes();
  const auto& notes_str = attrs.at(kNotesAttr).s();
  nlohmann::json j_obj = nlohmann::json::parse(notes_str);
  cache_dir = j_obj["backend_cache_dir"].get<std::string>();
  cache_key = j_obj["backend_cache_key"].get<std::string>();
  if (cache_dir.empty()) {
    LOGS_DEFAULT(WARNING) << "Retrieved backend cache dir empty";
  }
  if (cache_key.empty()) {
    LOGS_DEFAULT(WARNING) << "Retrieved backend cache key empty";
  }
}

std::unique_ptr<GraphViewer> RetrieveOriginalGraph(const Graph& ep_ctx_graph) {
  // TODO: Support for multi-node EP context model.
  const auto* p_node = GetEPContextNodePtr(ep_ctx_graph);
  const auto& attrs = p_node->GetAttributes();
  const auto& notes_str = attrs.at(kNotesAttr).s();
  nlohmann::json j_obj = nlohmann::json::parse(notes_str);

  const auto& orig_model_path = j_obj["orig_model_path"].get<std::string>();
  bool model_loaded = false;
  auto p_model_proto = ONNX_NAMESPACE::ModelProto::Create();
  if (!orig_model_path.empty() && fs::exists(orig_model_path) && fs::is_regular_file(orig_model_path)) {
    auto load_status = Model::Load(ToPathString(orig_model_path), *p_model_proto);
    model_loaded = load_status.IsOK();
  }
  if (!model_loaded) {
    p_model_proto->ParseFromString(j_obj["orig_model_proto_ser_str"].get<std::string>());
    if (p_model_proto->opset_import_size() == 0) {
      for (auto& elem : j_obj.items()) {
        if (elem.key() == "orig_model_path" || elem.key() == "orig_graph_name" || elem.key() == "orig_model_proto_ser_str") {
          continue;
        }
        auto* p_op_set_id_proto = p_model_proto->add_opset_import();
        *(p_op_set_id_proto->mutable_domain()) = elem.key();
        p_op_set_id_proto->set_version(std::stoll(elem.value().get<std::string>()));
      }
    }
  }
  auto& logger = logging::LoggingManager::DefaultLogger();
  auto p_model = Model::Create(std::move(*p_model_proto), ToPathString(orig_model_path), nullptr, logger);
  auto& graph = p_model->MainGraph();
  graph.ToGraphProto()->set_name(j_obj["orig_graph_name"].get<std::string>());

  return graph.CreateGraphViewer();
}

bool GraphHasEPContextNode(const Graph& graph) {
  size_t vitisai_len = std::strlen(kVitisAI);
  for (const auto* p_node : graph.Nodes()) {
    if (p_node->OpType() != kEPContextOp) {
      continue;
    }
    const auto& attrs = p_node->GetAttributes();
    if (attrs.count(kSourceAttr) == 0) {
      continue;
    }
    const auto& source_val = attrs.at(kSourceAttr).s();
    if (source_val == kVitisAIExecutionProvider) {
      return true;
    }
    if (source_val.length() != vitisai_len) {
      continue;
    }
    size_t j = 0;
    do {
      if (static_cast<unsigned char>(std::tolower(source_val[j])) != kVitisAI[j]) {
        break;
      }
      ++j;
    } while (j < vitisai_len);
    if (j == vitisai_len) {
      return true;
    }
  }
  return false;
}

bool FusedGraphHasEPContextNode(
    const std::vector<IExecutionProvider::FusedNodeAndGraph>& fused_nodes_and_graphs) {
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    bool has_node = GraphHasEPContextNode(fused_node_graph.filtered_graph.get().GetGraph());
    if (has_node) {
      return true;
    }
  }
  return false;
}

const fs::path& GetTopLevelModelPath(const GraphViewer& graph_viewer) {
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
  if (!ep_ctx_model_file_loc.empty()) {
    return true;
  }
  if (!ep_ctx_model_path_cfg.empty()) {
    ep_ctx_model_file_loc = ToPathString(ep_ctx_model_path_cfg);
  } else if (!model_path_str.empty()) {
    if (is_ep_ctx_model) {
      ep_ctx_model_file_loc = model_path_str;
    } else {
      // Two alternatives for this case.
      // Alternative 1:
      // 1) Implement/override the method `IExecutionProvider::GetEpContextNodes()`.
      // 2) And follow how the default path is implemented in `CreateEpContextModel()`
      // in the file "graph_partitioner.cc".
      // 3) Model dump is not required.
      // Alternative 2:
      // 1) Do NOT implement/override `IExecutionProvider::GetEpContextNodes()`.
      // 2) No need to follow `CreateEpContextModel()` in the file "graph_partitioner.cc",
      // freely implement what the default path is like.
      // 3) Model dump is required.
#if 0
      ep_ctx_model_file_loc = model_path_str + ToPathString("_ctx.onnx");
#endif
#if 1
      fs::path model_fs_path(model_path_str);
      fs::path ep_ctx_model_fs_path(model_fs_path.parent_path() / model_fs_path.stem());
      ep_ctx_model_fs_path += fs::path("_ctx.onnx");
      ep_ctx_model_file_loc = ToPathString(ep_ctx_model_fs_path.string());
#endif
    }
  }
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

std::string Slurp(const fs::path& file_location, bool binary_mode) {
  // std::filesystem::value_type == onnxruntime::PathChar == ORTCHAR_T
  // std::filesystem::string_type == onnxruntime::PathString
  // const char* location_str = PathToUTF8String(file_location.native()).c_str();
  std::ifstream ifs;
  ifs.exceptions(std::ifstream::failbit | std::ifstream::badbit);
  std::stringstream ss;
  try {
    auto open_mode = binary_mode ? (std::ios::in | std::ios::binary) : std::ios::in;
    ifs.open(file_location.string().c_str(), open_mode);
    ss << ifs.rdbuf();
    if (!ss.good()) {
      LOGS_DEFAULT(WARNING) << "Failed to write to stream";
    }
    ifs.close();
  } catch (std::system_error& se) {
    LOGS_DEFAULT(WARNING) << "Failed to read " << file_location << ": " << se.code().message();
  }
  return ss.str();
}

}  // namespace onnxruntime
