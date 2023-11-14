
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include <atomic>
#include <filesystem>
#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "core/common/exceptions.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/graph/model.h"
#include "core/providers/shared/common.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_env.h"

#include "vaip/custom_op.h"
#include "vaip/global_api.h"
#include "vaip/dll_safe.h"
#include "vaip/vaip_ort_api.h"
#include "vaip/vai_assert.h"
#include "vaip/graph.h"
#include "vaip/node.h"
#include "vaip/node_arg.h"
#include "./tensor_proto.h"
#include "./attr_proto.h"
#include "onnxruntime_config.h"
#include "version_info.h"  // version_info.hpp.in

using namespace onnxruntime;
using namespace onnx;

// The filename extension for a shared library is different per platform
#ifdef _WIN32
#define LIBRARY_PREFIX
#define LIBRARY_EXTENSION ORT_TSTR(".dll")
#elif defined(__APPLE__)
#define LIBRARY_PREFIX "lib"
#define LIBRARY_EXTENSION ".dylib"
#else
#define LIBRARY_PREFIX "lib"
#define LIBRARY_EXTENSION ".so"
#endif

vaip_core::OrtApiForVaip* create_org_api_hook();
struct OrtVitisAIEpAPI {
  void (*initialize_onnxruntime_vitisai_ep)(vaip_core::OrtApiForVaip* api, std::vector<OrtCustomOpDomain*>& ret_domain);
  std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>* (*compile_onnx_model_with_options)(const std::string& model_path, const onnxruntime::Graph& graph, const onnxruntime::ProviderOptions& options);
  void Ensure() {
    if (handle_)
      return;
    auto full_path = Env::Default().GetRuntimePath() + PathString(LIBRARY_PREFIX ORT_TSTR("onnxruntime_vitisai_ep") LIBRARY_EXTENSION);
    ORT_THROW_IF_ERROR(Env::Default().LoadDynamicLibrary(full_path, true, &handle_));
    ORT_THROW_IF_ERROR(Env::Default().GetSymbolFromLibrary(handle_, "initialize_onnxruntime_vitisai_ep", (void**)&initialize_onnxruntime_vitisai_ep));
    ORT_THROW_IF_ERROR(Env::Default().GetSymbolFromLibrary(handle_, "compile_onnx_model_vitisai_ep_with_options", (void**)&compile_onnx_model_with_options));
  }

 private:
  void* handle_{};
};

static OrtVitisAIEpAPI s_library_vitisaiep;

vaip_core::DllSafe<std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>> compile_onnx_model_with_options(const std::string& model_path, const onnxruntime::Graph& graph, const onnxruntime::ProviderOptions& options) {
  return vaip_core::DllSafe(s_library_vitisaiep.compile_onnx_model_with_options(model_path, graph, options));
}

std::vector<OrtCustomOpDomain*> initialize_vitisai_ep() {
  s_library_vitisaiep.Ensure();
  Status status = Status::OK();
  try {
    OrtEnv::LoggingManagerConstructionInfo lm_info{nullptr, nullptr, ORT_LOGGING_LEVEL_WARNING, "onnxruntime-vitisai-ep"};
    std::ignore = OrtEnv::GetInstance(lm_info, status);
  } catch (onnxruntime::OnnxRuntimeException& /*e*/) {
  }
  auto domains = std::vector<OrtCustomOpDomain*>();
  domains.reserve(100);
  s_library_vitisaiep.initialize_onnxruntime_vitisai_ep(create_org_api_hook(), domains);

  return domains;
}

static vaip_core::OrtApiForVaip the_global_api;
vaip_core::OrtApiForVaip* create_org_api_hook() {
  the_global_api.host_ = Provider_GetHost();
  the_global_api.model_load = [](const std::string& filename) -> Model* {
    ONNX_NAMESPACE::ModelProto model_proto;
    auto& logger = logging::LoggingManager::DefaultLogger();
    auto file_path = ToPathString(filename);
    auto status = Model::Load(file_path, model_proto);
    vai_assert(status.IsOK(), "load model proto error");
    auto model = std::make_unique<Model>(std::move(model_proto), file_path, nullptr, logger);
    return model.release();
  };
  the_global_api.model_delete = [](Model* model) { delete model; };
  the_global_api.model_clone = [](const Model& model) -> Model* {
    auto& logger = logging::LoggingManager::DefaultLogger();
    auto model_proto =
        const_cast<onnxruntime::Model&>(model).ToProto();
    auto file_path = model.ModelPath().ToPathString();
    auto ret = std::make_unique<Model>(std::move(model_proto), file_path, nullptr, logger);
    auto status = ret->MainGraph().Resolve();
    vai_assert(status.IsOK(), status.ErrorMessage());
    return ret.release();
  };
  the_global_api.model_set_meta_data = [](Model& model, const std::string& key,
                                          const std::string& value)
      -> void {
    const_cast<ModelMetaData&>(model.MetaData())[key] = value;
  };
  the_global_api.model_get_meta_data = [](const Model& model,
                                          const std::string& key) -> vaip_core::DllSafe<std::string> {
    auto& m = model.MetaData();
    auto it = m.find(key);
    auto ret = std::string();
    if (it != m.end()) {
      ret = it->second;
    }
    return vaip_core::DllSafe(ret);
  };

  the_global_api.model_has_meta_data = [](const Model& model, const std::string& key) -> int {
    auto& m = model.MetaData();
    return m.find(key) != m.end() ? 1 : 0;
  };

  the_global_api.model_main_graph = [](Model& model) -> Graph& {
    return model.MainGraph();
  };
  the_global_api.graph_get_model = [](const Graph& graph) -> const Model& {
    return graph.GetModel();
  };
  the_global_api.graph_get_inputs_unsafe =
      [](const Graph& graph) -> vaip_core::DllSafe<std::vector<const NodeArg*>> {
    auto ret = std::vector<const NodeArg*>();
    auto inputs = graph.GetInputs();
    for (auto input : inputs) {
      vai_assert(input->Exists(), input->Name());
      ret.push_back(input);
    }
    return vaip_core::DllSafe(std::move(ret));
  };
  the_global_api.graph_get_outputs_unsafe =
      [](const Graph& graph) -> vaip_core::DllSafe<std::vector<const NodeArg*>> {
    return vaip_core::DllSafe(graph.GetOutputs());
  };

  the_global_api.graph_set_outputs =
      [](Graph& graph, gsl::span<const NodeArg* const> outputs) -> void {
    return graph.SetOutputs(outputs);
  };

  the_global_api.graph_get_node_arg =
      [](const Graph& graph, const std::string& name) -> const NodeArg* {
    return graph.GetNodeArg(name);
  };
  the_global_api.graph_producer_node = [](const Graph& graph, const std::string& name) -> const Node* {
    return graph.GetProducerNode(name);
  };

  the_global_api.graph_get_node = [](const Graph& graph,
                                     size_t index) -> const Node* {
    return graph.GetNode(index);
  };

  the_global_api.graph_save = vaip::graph_save;
  the_global_api.graph_fuse = vaip::graph_fuse;
  the_global_api.graph_remove_node = vaip::graph_remove_node;
  the_global_api.graph_add_node =
      [](Graph& graph, const std::string& name, const std::string& op_type,
         const std::string& description,
         const std::vector<const NodeArg*>& input_args,
         const std::vector<const NodeArg*>& output_args,
         vaip_core::NodeAttributes& attributes,
         const std::string& domain) -> Node& {
    return vaip::graph_add_node(
        graph, name, op_type, description, input_args, output_args,
        std::move(reinterpret_cast<onnxruntime::NodeAttributes&>(attributes)),
        domain);
  };

  the_global_api.graph_get_all_initialized_tensors =
      [](const Graph& graph) -> const InitializedTensorSet& {
    return graph.GetAllInitializedTensors();
  };

  the_global_api.graph_resolve = [](Graph& graph, bool force) {
    if (force) {
      graph.SetGraphResolveNeeded();
    }
    auto status = graph.Resolve();
    return status.Code();
  };

  the_global_api.graph_get_consumer_nodes_unsafe =
      [](const Graph& graph,
         const std::string& node_arg_name) -> vaip_core::DllSafe<std::vector<const Node*>> {
    return vaip_core::DllSafe(graph.GetConsumerNodes(node_arg_name));
  };
  the_global_api.graph_nodes_unsafe =
      [](const Graph& graph) -> vaip_core::DllSafe<std::vector<const Node*>> {
    auto& node_refererence = graph.Nodes();
    std::vector<const Node*> nodes((size_t)graph.NumberOfNodes(), nullptr);
    std::transform(node_refererence.begin(), node_refererence.end(),
                   nodes.begin(), [](const Node& n) { return &n; });
    return vaip_core::DllSafe(std::move(nodes));
  };
  the_global_api.graph_get_name = [](const Graph& graph) -> const std::string& {
    return graph.Name();
  };
  the_global_api.graph_reverse_dfs_from =
      [](const Graph& graph, gsl::span<const Node* const> from,
         const std::function<void(const Node*)>& enter,
         const std::function<void(const Node*)>& leave,
         const std::function<bool(const Node* from, const Node* to)>& stop) {
        graph.ReverseDFSFrom(from, enter, leave, nullptr, stop);
      };
  // node
  the_global_api.node_get_inputs_unsafe = vaip::node_get_inputs;
  the_global_api.node_get_output_node_args_unsafe = vaip::node_get_output_node_args;

  the_global_api.node_op_type = [](const Node& node) -> const std::string& {
    return node.OpType();
  };
  the_global_api.node_op_domain = [](const Node& node) -> const std::string& {
    return node.Domain();
  };
  the_global_api.node_get_index = [](const Node& node) -> size_t {
    return (size_t)node.Index();
  };
  the_global_api.node_get_name = [](const Node& node) -> const std::string& {
    return node.Name();
  };
  the_global_api.node_description = [](const Node& node) -> const std::string& {
    return node.Description();
  };

  the_global_api.node_get_attributes =
      [](Node& node) -> vaip_core::NodeAttributes& {
    return reinterpret_cast<vaip_core::NodeAttributes&>(
        node.GetMutableAttributes());
  };

  the_global_api.node_type_is_fused = [](const Node& node) {
    return node.NodeType() == onnxruntime::Node::Type::Fused;
  };
  the_global_api.node_get_function_body =
      [](const Node& node) -> const onnxruntime::Graph& {
    assert(node.GetFunctionBody() != nullptr);
    return node.GetFunctionBody()->Body();
  };

  // node_arg
  the_global_api.node_arg_get_name_unsafe =
      [](const NodeArg& node_arg) -> const std::string& {
    return node_arg.Name();
  };
  the_global_api.node_arg_clone = vaip::node_arg_clone;
  the_global_api.node_arg_new = vaip::node_arg_new;
  the_global_api.node_arg_is_exists = vaip::node_arg_is_exists;
  the_global_api.node_arg_is_constant = vaip::node_arg_is_constant;
  the_global_api.node_arg_get_shape_i64_unsafe = vaip::node_arg_get_shape_i64;
  the_global_api.node_arg_set_shape_i64 = vaip::node_arg_set_shape_i64;
  the_global_api.node_arg_get_denotation_unsafe = vaip::node_arg_get_denotation;
  the_global_api.node_arg_set_denotation = vaip::node_arg_set_denotation;
  the_global_api.node_arg_get_const_data_as_tensor =
      vaip::node_arg_get_const_data_as_tensor;

  the_global_api.node_arg_get_element_type = vaip::node_arg_get_element_type;
  the_global_api.node_arg_set_element_type = [](NodeArg& node_arg, int type) {
    auto data_type = ONNX_NAMESPACE::TensorProto::UNDEFINED;
    switch (type) {
      case 1:
        data_type = ONNX_NAMESPACE::TensorProto::FLOAT;
        break;
      case 2:
        data_type = ONNX_NAMESPACE::TensorProto::UINT8;
        break;
      case 3:
        data_type = ONNX_NAMESPACE::TensorProto::INT8;
        break;

      case 4:
        data_type = ONNX_NAMESPACE::TensorProto::UINT16;
        break;
      case 5:
        data_type = ONNX_NAMESPACE::TensorProto::INT16;
        break;
      case 6:
        data_type = ONNX_NAMESPACE::TensorProto::INT32;
        break;
      case 7:
        data_type = ONNX_NAMESPACE::TensorProto::INT64;
        break;
      case 8:
        data_type = ONNX_NAMESPACE::TensorProto::STRING;
        break;
      case 9:
        data_type = ONNX_NAMESPACE::TensorProto::BOOL;
        break;
      case 10:
        data_type = ONNX_NAMESPACE::TensorProto::FLOAT16;
        break;
      case 11:
        data_type = ONNX_NAMESPACE::TensorProto::DOUBLE;
        break;
      case 12:
        data_type = ONNX_NAMESPACE::TensorProto::UINT32;
        break;
      case 13:
        data_type = ONNX_NAMESPACE::TensorProto::UINT64;
        break;
      case 14:
        data_type = ONNX_NAMESPACE::TensorProto::COMPLEX64;
        break;
      case 15:
        data_type = ONNX_NAMESPACE::TensorProto::COMPLEX128;
        break;
      case 16:
        data_type = ONNX_NAMESPACE::TensorProto::BFLOAT16;
        break;
      default:
        vai_assert(false, "TensorProto::DataType not supoort");
    }
    return vaip::node_arg_set_element_type(node_arg, data_type);
  };
  /// attr proto
  the_global_api.attr_proto_delete = [](onnx::AttributeProto* v) { delete v; };
  the_global_api.attr_proto_clone =
      [](const onnx::AttributeProto& v) -> onnx::AttributeProto* {
    return new onnx::AttributeProto(v);
  };
  the_global_api.attr_proto_get_name =
      [](const onnx::AttributeProto& attr_proto) -> const std::string& {
    return attr_proto.name();
  };
  the_global_api.attr_proto_set_name = [](onnx::AttributeProto* attr_proto,
                                          const std::string& name) {
    attr_proto->set_name(name);
  };
  the_global_api.attr_proto_new_int = vaip::attr_proto_new_int;
  the_global_api.attr_proto_new_float = vaip::attr_proto_new_float;
  the_global_api.attr_proto_new_string = vaip::attr_proto_new_string;
  the_global_api.attr_proto_new_tensor = vaip::attr_proto_new_tensor;
  the_global_api.attr_proto_new_ints = vaip::attr_proto_new_ints;
  the_global_api.attr_proto_new_floats = vaip::attr_proto_new_floats;
  the_global_api.attr_proto_new_strings = vaip::attr_proto_new_strings;
  the_global_api.attr_proto_get_int = vaip::attr_proto_get_int;
  the_global_api.attr_proto_get_float = vaip::attr_proto_get_float;
  the_global_api.attr_proto_get_string = vaip::attr_proto_get_string;
  the_global_api.attr_proto_get_tensor = vaip::attr_proto_get_tensor;
  the_global_api.attr_proto_get_ints = vaip::attr_proto_get_ints;
  the_global_api.attr_proto_get_floats = vaip::attr_proto_get_floats;
  the_global_api.attr_proto_get_strings = vaip::attr_proto_get_strings;
  the_global_api.attr_proto_get_type =
      [](const onnx::AttributeProto& attr) -> int { return attr.type(); };

  /// node attributes
  the_global_api.node_attributes_new = []() {
    return reinterpret_cast<vaip_core::NodeAttributes*>(new NodeAttributes());
  };
  the_global_api.node_attributes_add = [](vaip_core::NodeAttributes& p,
                                          onnx::AttributeProto&& attr) {
    reinterpret_cast<NodeAttributes&>(p).insert_or_assign(attr.name(),
                                                          std::move(attr));
  };
  the_global_api.node_attributes_delete = [](vaip_core::NodeAttributes* p) {
    delete reinterpret_cast<NodeAttributes*>(p);
  };
  the_global_api.node_attributes_get = [](vaip_core::NodeAttributes& p,
                                          const std::string& name) -> ONNX_NAMESPACE::AttributeProto* {
    auto& attr = reinterpret_cast<NodeAttributes&>(p);
    auto it = attr.find(name);
    if (it == attr.end()) {
      return nullptr;
    }
    return &it->second;
  };
  the_global_api.node_attributes_get_keys = [](vaip_core::NodeAttributes& p) -> vaip_core::DllSafe<std::vector<std::string>> {
    auto ret = std::vector<std::string>();
    auto& attr = reinterpret_cast<NodeAttributes&>(p);
    ret.reserve(attr.size());
    for (auto& it : attr) {
      ret.push_back(it.first);
    }
    return vaip_core::DllSafe(std::move(ret));
  };
  /// tensor proto
  the_global_api.tensor_proto_get_shape_unsafe = [](const onnx::TensorProto& t) -> vaip_core::DllSafe<std::vector<int64_t>> {
    return vaip_core::DllSafe<std::vector<int64_t>>(vaip::tensor_proto_get_shape(t));
  };

  the_global_api.tensor_proto_data_type =
      [](const onnx::TensorProto& t) -> int { return t.data_type(); };

  the_global_api.tensor_proto_delete = [](onnx::TensorProto* tp) { delete tp; };

  the_global_api.tensor_proto_new_floats =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<float>& data) -> onnx::TensorProto* {
    return new onnx::TensorProto{
        vaip::tensor_proto_new_floats(name, shape, data)};
  };
  the_global_api.tensor_proto_new_i32 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int32_t>& data) -> onnx::TensorProto* {
    return new onnx::TensorProto{vaip::tensor_proto_new_i32(name, shape, data)};
  };
  the_global_api.tensor_proto_new_i64 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int64_t>& data) -> onnx::TensorProto* {
    return new onnx::TensorProto{vaip::tensor_proto_new_i64(name, shape, data)};
  };
  the_global_api.tensor_proto_new_i8 =
      [](const std::string& name, const std::vector<int64_t>& shape,
         const std::vector<int8_t>& data) -> onnx::TensorProto* {
    return new onnx::TensorProto{vaip::tensor_proto_new_i8(name, shape, data)};
  };
  the_global_api.tensor_proto_raw_data_size = vaip::tensor_proto_raw_data_size;

  the_global_api.tensor_proto_as_raw = vaip::tensor_proto_as_raw;
  the_global_api.tensor_proto_get_name = vaip::tensor_proto_get_name;

  the_global_api.get_lib_name = []() -> vaip_core::DllSafe<std::string> {
    return vaip_core::DllSafe(std::string("onnxruntime.") + std::string(ORT_VERSION));
  };

  the_global_api.get_lib_id = []() -> vaip_core::DllSafe<std::string> {
    return vaip_core::DllSafe(std::string(GIT_COMMIT_ID));
  };
  return &the_global_api;
}
