// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "vaip/global_api.h"

#include <atomic>
#include <iostream>
#include <codecvt>
#include <fstream>

#include "./vai_assert.h"

#include "core/common/exceptions.h"
#include "core/framework/error_code_helper.h"
#include "core/providers/shared/common.h"

#include <nlohmann/json.hpp>

#include "vaip/dll_safe.h"
#include "vaip/vaip_ort_api.h"
#include "vaip/graph.h"
#include "vaip/node.h"
#include "vaip/node_arg.h"

#include "./tensor_proto.h"
#include "./attr_proto.h"
#include "./register_xir_ops.h"

#include "onnxruntime_config.h"
#include "version_info.h"  // version_info.hpp.in

using namespace onnxruntime;
using json = nlohmann::json;

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
  std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>* (*compile_onnx_model_3)(const std::string& model_path,
                                                                                      const onnxruntime::Graph& graph,
                                                                                      const char* json_config);
  std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>* (*compile_onnx_model_with_options)(
      const std::string& model_path, const onnxruntime::Graph& graph, const onnxruntime::ProviderOptions& options);
  void Ensure() {
    if (handle_)
      return;
    auto& env = Provider_GetHost()->Env__Default();
    auto full_path = env.GetRuntimePath() + PathString(LIBRARY_PREFIX ORT_TSTR("onnxruntime_vitisai_ep") LIBRARY_EXTENSION);
    ORT_THROW_IF_ERROR(env.LoadDynamicLibrary(full_path, true, &handle_));
    ORT_THROW_IF_ERROR(env.GetSymbolFromLibrary(handle_, "initialize_onnxruntime_vitisai_ep", (void**)&initialize_onnxruntime_vitisai_ep));
    auto status1 = env.GetSymbolFromLibrary(handle_, "compile_onnx_model_vitisai_ep_with_options", (void**)&compile_onnx_model_with_options);
    auto status2 = env.GetSymbolFromLibrary(handle_, "compile_onnx_model_vitisai_ep", (void**)&compile_onnx_model_3);
    if (!status1.IsOK() && !status2.IsOK()) {
      ::onnxruntime::LogRuntimeError(0, status1, __FILE__, static_cast<const char*>(__FUNCTION__), __LINE__);
      ORT_THROW(status1);
    }
  }

 private:
  void* handle_{};
};

static OrtVitisAIEpAPI s_library_vitisaiep;
static std::shared_ptr<KernelRegistry> s_kernel_registry_vitisaiep;
static std::vector<OrtCustomOpDomain*> s_domains_vitisaiep;
static vaip_core::OrtApiForVaip the_global_api;
std::shared_ptr<KernelRegistry> get_kernel_registry_vitisaiep() { return s_kernel_registry_vitisaiep; }
const std::vector<OrtCustomOpDomain*>& get_domains_vitisaiep() { return s_domains_vitisaiep; }

static std::string config_to_json_str(const onnxruntime::ProviderOptions& config) {
  auto iter = config.find("config_file");
  if (iter == config.end()) {
    std::cerr << "Error: Key 'config_file' not found in config" << std::endl;
    return "";
  }
  const auto& filename = config.at("config_file");
  std::ifstream f(filename);
  if (!f.is_open()) {
    std::cerr << "Error: Failed to open file: " << filename << std::endl;
    return "";
  }
  nlohmann::json data;
  try {
    data = nlohmann::json::parse(f);
  } catch (const std::exception& e) {
    std::cerr << "Error: Failed to parse JSON from file: " << filename << ", Reason: " << e.what() << std::endl;
    return "";
  }
  for (const auto& entry : config) {
    data[entry.first] = entry.second;
  }
  try {
    return data.dump();
  } catch (const std::exception& e) {
    std::cerr << "Error: Failed to convert JSON data to string, Reason: " << e.what() << std::endl;
    return "";
  }
}

vaip_core::DllSafe<std::vector<std::unique_ptr<vaip_core::ExecutionProvider>>> compile_onnx_model(
    const onnxruntime::GraphViewer& graph_viewer, const logging::Logger& logger, const ProviderOptions& options) {
#ifndef _WIN32
  auto model_path = graph_viewer.ModelPath().ToPathString();
#else
  using convert_t = std::codecvt_utf8<wchar_t>;
  std::wstring_convert<convert_t, wchar_t> strconverter;
  auto model_path = strconverter.to_bytes(graph_viewer.ModelPath().ToPathString());
#endif
  if (s_library_vitisaiep.compile_onnx_model_with_options) {
    return vaip_core::DllSafe(s_library_vitisaiep.compile_onnx_model_with_options(model_path, graph_viewer.GetGraph(), options));
  } else {
    auto json_str = config_to_json_str(options);
    return vaip_core::DllSafe(s_library_vitisaiep.compile_onnx_model_3(model_path, graph_viewer.GetGraph(), json_str.c_str()));
  }
}

struct MyCustomOpKernel : OpKernel {
  MyCustomOpKernel(const OpKernelInfo& info, const OrtCustomOp& op) : OpKernel(info), op_(op) {
    op_kernel_ =
        op_.CreateKernel(&op_, Ort::Global<void>::api_, reinterpret_cast<const OrtKernelInfo*>(&info));
  }

  ~MyCustomOpKernel() override { op_.KernelDestroy(op_kernel_); }

  Status Compute(OpKernelContext* ctx) const override {
    op_.KernelCompute(op_kernel_, reinterpret_cast<OrtKernelContext*>(ctx));
    return Status::OK();
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MyCustomOpKernel);

  const OrtCustomOp& op_;
  void* op_kernel_;
};

void create_kernel_registry(std::vector<OrtCustomOpDomain*> domains) {
  s_kernel_registry_vitisaiep = KernelRegistry::Create();
  for (const auto& domain : domains) {
    for (const auto* op : domain->custom_ops_) {
      auto def_builder = KernelDefBuilder::Create();
      def_builder->SetName(op->GetName(op));
      def_builder->SetDomain(domain->domain_.c_str());
      def_builder->SinceVersion(1);
      if (op->version > 12) {
        auto input_count = op->GetInputTypeCount(op);
        for (auto i = 0u; i < input_count; i++) {
          def_builder->InputMemoryType(op->GetInputMemoryType(op, i), i);
        }
      }
      def_builder->Provider(onnxruntime::kVitisAIExecutionProvider);
      KernelCreateFn kernel_create_fn =
          [op](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
        // out = std::make_unique<MyCustomOpKernel>(info, *op);
        return Status::OK();
      };
      std::ignore = s_kernel_registry_vitisaiep->Register(KernelCreateInfo(def_builder->Build(), kernel_create_fn));
    }
  }
}
void initialize_vitisai_ep() {
  s_library_vitisaiep.Ensure();
  s_domains_vitisaiep.reserve(100);
  s_library_vitisaiep.initialize_onnxruntime_vitisai_ep(create_org_api_hook(), s_domains_vitisaiep);
  vaip::register_xir_ops(s_domains_vitisaiep);
  create_kernel_registry(s_domains_vitisaiep);
}

vaip_core::OrtApiForVaip* create_org_api_hook() {
  InitProviderOrtApi();
  the_global_api.host_ = Provider_GetHost();
  assert(Ort::Global<void>::api_ != nullptr);
  the_global_api.ort_api_ = Ort::Global<void>::api_;
  the_global_api.model_load = [](const std::string& filename) -> Model* {
    auto model_proto = ONNX_NAMESPACE::ModelProto::Create();
    auto& logger = logging::LoggingManager::DefaultLogger();
    auto file_path = ToPathString(filename);
    auto status = Model::Load(file_path, *model_proto);
    vai_assert(status.IsOK(), "load model proto error");
    auto model = Model::Create(std::move(*model_proto), file_path, logger);
    return model.release();
  };
  the_global_api.model_delete = [](Model* model) { delete model; };

  the_global_api.model_clone = [](const Model& const_model) -> Model* {
    auto& logger = logging::LoggingManager::DefaultLogger();
    auto& model = const_cast<onnxruntime::Model&>(const_model);
    auto model_proto = model.ToProto();
    auto file_path = model.MainGraph().ModelPath().ToPathString();
    auto ret = Model::Create(std::move(*model_proto), file_path, logger);
    auto status = ret->MainGraph().Resolve();
    vai_assert(status.IsOK(), status.ErrorMessage());
    return ret.release();
  };
  the_global_api.model_set_meta_data = [](Model& model, const std::string& key, const std::string& value) {
    const_cast<ModelMetaData&>(model.MetaData())[key] = value;
  };
  the_global_api.model_get_meta_data =
      [](const Model& model, const std::string& key) -> vaip_core::DllSafe<std::string> {
    if (model.MetaData().count(key)) {
      return vaip_core::DllSafe(model.MetaData().at(key));
    }
    return vaip_core::DllSafe(std::string());
  };
  the_global_api.model_has_meta_data = [](const Model& model, const std::string& key) -> int {
    return int(model.MetaData().count(key));
  };
  the_global_api.model_main_graph = [](Model& model) -> Graph& { return model.MainGraph(); };
  the_global_api.graph_get_model = [](const Graph& graph) -> const Model& { return graph.GetModel(); };
  the_global_api.graph_get_inputs_unsafe = [](const Graph& graph) -> auto {
    return vaip_core::DllSafe(graph.GetInputs());
  };
  the_global_api.graph_get_outputs_unsafe = [](const Graph& graph) -> auto {
    return vaip_core::DllSafe(graph.GetOutputs());
  };
  the_global_api.graph_set_outputs = [](Graph& graph, gsl::span<const NodeArg* const> outputs) {
    graph.SetOutputs(outputs);
  };
  the_global_api.graph_get_node_arg = [](const Graph& graph, const std::string& name) -> const NodeArg* {
    return graph.GetNodeArg(name);
  };
  the_global_api.graph_producer_node = [](const Graph& graph, const std::string& name) -> const Node* {
    return graph.GetProducerNode(name);
  };
  the_global_api.graph_get_node = [](const Graph& graph, size_t index) -> const Node* {
    return graph.GetNode(index);
  };
  the_global_api.graph_save = vaip::graph_save;
  the_global_api.graph_fuse = vaip::graph_fuse;
  the_global_api.graph_remove_node = vaip::graph_remove_node;
  the_global_api.graph_add_node = vaip::graph_add_node;
  the_global_api.graph_get_all_initialized_tensors = [](const Graph& graph) -> const InitializedTensorSet& {
    return graph.GetAllInitializedTensors();
  };
  the_global_api.graph_resolve = [](Graph& graph, bool force) {
    if (force) {
      graph.SetGraphResolveNeeded();
    }
    auto status = graph.Resolve();
    return status.Code();
  };
  the_global_api.graph_get_consumer_nodes_unsafe = [](const Graph& graph, const std::string& node_arg_name) -> auto {
    return vaip_core::DllSafe(graph.GetConsumerNodes(node_arg_name));
  };
  the_global_api.graph_nodes_unsafe = [](const Graph& graph) -> auto { return vaip_core::DllSafe(graph.Nodes()); };
  the_global_api.graph_get_name = [](const Graph& graph) -> const std::string& { return graph.Name(); };
  the_global_api.graph_reverse_dfs_from = [](const Graph& graph, gsl::span<const Node* const> from,
                                             const auto& enter, const auto& leave, const auto& stop) {
    graph.ReverseDFSFrom(from, enter, leave, nullptr, stop);
  };
  // node
  the_global_api.node_get_inputs_unsafe = vaip::node_get_inputs;
  the_global_api.node_get_output_node_args_unsafe = vaip::node_get_output_node_args;
  the_global_api.node_op_type = [](const Node& node) -> const std::string& { return node.OpType(); };
  the_global_api.node_op_domain = [](const Node& node) -> const std::string& { return node.Domain(); };
  the_global_api.node_get_index = [](const Node& node) -> size_t { return node.Index(); };
  the_global_api.node_get_name = [](const Node& node) -> const std::string& { return node.Name(); };
  the_global_api.node_description = [](const Node& node) -> const std::string& { return node.Description(); };
  the_global_api.node_get_attributes = [](Node& node) -> NodeAttributes& {
    return const_cast<NodeAttributes&>(node.GetAttributes());
  };
  the_global_api.node_type_is_fused = [](const Node& node) { return node.NodeType() == Node::Type::Fused; };
  the_global_api.node_get_function_body = [](const Node& node) -> const auto& {
    assert(node.GetFunctionBody() != nullptr);
    return node.GetFunctionBody()->Body();
  };

  // node_arg
  the_global_api.node_arg_get_name_unsafe =
      [](const NodeArg& node_arg) -> const std::string& { return node_arg.Name(); };
  the_global_api.node_arg_clone = vaip::node_arg_clone;
  the_global_api.node_arg_new = vaip::node_arg_new;
  the_global_api.node_arg_is_exists = [](const NodeArg& node_arg) { return node_arg.Exists(); };
  the_global_api.node_arg_is_constant = vaip::node_arg_is_constant;
  the_global_api.node_arg_get_shape_i64_unsafe = vaip::node_arg_get_shape_i64;
  the_global_api.node_arg_set_shape_i64 = vaip::node_arg_set_shape_i64;
  the_global_api.node_arg_get_denotation_unsafe = vaip::node_arg_get_denotation;

  the_global_api.node_arg_set_denotation = vaip::node_arg_set_denotation;
  the_global_api.node_arg_get_const_data_as_tensor = vaip::node_arg_get_const_data_as_tensor;

  the_global_api.node_arg_get_element_type = vaip::node_arg_get_element_type;
  the_global_api.node_arg_set_element_type = vaip::node_arg_set_element_type;
  /// attr proto
  the_global_api.attr_proto_delete = [](ONNX_NAMESPACE::AttributeProto* v) { delete v; };
  the_global_api.attr_proto_clone = [](const ONNX_NAMESPACE::AttributeProto& v) -> ONNX_NAMESPACE::AttributeProto* {
    auto ret = ONNX_NAMESPACE::AttributeProto::Create();
    *ret = v;
    return ret.release();
  };
  the_global_api.attr_proto_get_name = [](const auto& attr_proto) -> const std::string& { return attr_proto.name(); };
  the_global_api.attr_proto_set_name = [](auto* attr_proto, const auto& name) { attr_proto->set_name(name); };
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
  the_global_api.attr_proto_get_type = [](const ONNX_NAMESPACE::AttributeProto& attr) -> int { return attr.type(); };

  /// node attributes
  the_global_api.node_attributes_new = []() { return NodeAttributes::Create().release(); };
  the_global_api.node_attributes_add = [](NodeAttributes& p, ONNX_NAMESPACE::AttributeProto&& attr) {
    p.insert_or_assign(attr.name(), std::move(attr));
  };

  the_global_api.node_attributes_delete = [](NodeAttributes* p) { delete p; };
  the_global_api.node_attributes_get =
      [](const NodeAttributes& attr, const std::string& name) -> const ONNX_NAMESPACE::AttributeProto* {
    if (attr.count(name)) {
      return &attr.at(name);
    }
    return nullptr;
  };
  the_global_api.node_attributes_get_keys = [](NodeAttributes& attr) -> vaip_core::DllSafe<std::vector<std::string>> {
    auto ret = std::vector<std::string>();
    ret.reserve(attr.size());
    for (auto& it : attr) {
      ret.push_back(it.first);
    }
    return vaip_core::DllSafe(std::move(ret));
  };
  /// tensor proto
  the_global_api.tensor_proto_get_shape_unsafe = vaip::tensor_proto_get_shape;
  the_global_api.tensor_proto_data_type = [](const ONNX_NAMESPACE::TensorProto& t) -> int { return t.data_type(); };
  the_global_api.tensor_proto_delete = [](ONNX_NAMESPACE::TensorProto* tp) { delete tp; };
  the_global_api.tensor_proto_new_floats = vaip::tensor_proto_new_floats;
  the_global_api.tensor_proto_new_i32 = vaip::tensor_proto_new_i32;
  the_global_api.tensor_proto_new_i64 = vaip::tensor_proto_new_i64;
  the_global_api.tensor_proto_new_i8 = vaip::tensor_proto_new_i8;
  the_global_api.tensor_proto_raw_data_size = [](const auto& tensor) { return tensor.raw_data().size(); };
  the_global_api.tensor_proto_as_raw = vaip::tensor_proto_as_raw;
  the_global_api.tensor_proto_get_name = [](const auto& tensor) -> const std::string& { return tensor.name(); };

  the_global_api.get_lib_name = []() -> vaip_core::DllSafe<std::string> {
    return vaip_core::DllSafe(std::string("onnxruntime.") + std::string(ORT_VERSION));
  };

  the_global_api.get_lib_id = []() -> vaip_core::DllSafe<std::string> {
    return vaip_core::DllSafe(std::string(GIT_COMMIT_ID));
  };
  return &the_global_api;
}
