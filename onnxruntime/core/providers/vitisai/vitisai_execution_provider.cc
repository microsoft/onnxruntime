// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "vitisai_execution_provider.h"

// Standard headers/libs.
#include <cassert>
#include <fstream>
#include <istream>
#include <filesystem>

// 1st-party headers/libs.
#include "core/platform/env_var_utils.h"

#include "vaip/capability.h"
#include "vaip/global_api.h"
#include "ep_context_utils.h"

using namespace ONNX_NAMESPACE;

namespace fs = std::filesystem;

namespace onnxruntime {
constexpr const char* VITISAI = "VITISAI";

VitisAIExecutionProvider::VitisAIExecutionProvider(
    const ProviderOptions& info)
    // const ProviderOptions& info, const SessionOptions* p_sess_opts)
    : IExecutionProvider{onnxruntime::kVitisAIExecutionProvider}, info_(info) {
  CreateKernelRegistry();

  auto it = info_.find("ep_context_enable");
  ep_ctx_enabled_ = it != info_.end() && it->second == "1";
  it = info_.find("ep_context_embed_mode");
  ep_ctx_embed_mode_ = it != info_.end() && it->second != "0";
  // ep_ctx_embed_mode_ = it == info_.end() || it->second != "0";
  it = info_.find("ep_context_file_path");
  ep_ctx_model_path_cfg_ = it == info_.end() ? "" : it->second;
  LOGS_DEFAULT(VERBOSE) << "EP Context cache enabled: " << ep_ctx_enabled_;
  LOGS_DEFAULT(VERBOSE) << "EP context cache embed mode: " << ep_ctx_embed_mode_;
  LOGS_DEFAULT(VERBOSE) << "User specified EP context cache path: " << ep_ctx_model_path_cfg_;
}

#if 0
VitisAIExecutionProvider::~VitisAIExecutionProvider() {
  // TODO: EP context related resources.
}
#endif

void VitisAIExecutionProvider::LoadEPContexModelFromFile() const {
  // XXX: should "p_ep_ctx_model_" be checked or not?
  if (!p_ep_ctx_model_ && !ep_ctx_model_file_loc_.empty()) {
    auto p_model_proto = ONNX_NAMESPACE::ModelProto::Create();
    auto status = Model::Load(ep_ctx_model_file_loc_, *p_model_proto);
    if (!status.IsOK()) {
      ORT_THROW("Loading EP context model failed from ", ep_ctx_model_file_loc_);
    }
    auto& logger = logging::LoggingManager::DefaultLogger();
    p_ep_ctx_model_ = Model::Create(std::move(*p_model_proto), ep_ctx_model_file_loc_, nullptr, logger);
    LOGS_DEFAULT(VERBOSE) << "Loaded EP context model from: " << ep_ctx_model_file_loc_;
  } else if (ep_ctx_model_file_loc_.empty()) {
    LOGS_DEFAULT(WARNING) << "Cannot load an EP-context model due to bad file path";
  }
}

void VitisAIExecutionProvider::CreateKernelRegistry() {
  for (const auto& domain : get_domains_vitisaiep()) {
    for (const auto* op : domain->custom_ops_) {
      vitisai_optypes_.insert(op->GetName(op));
    }
  }
}

std::shared_ptr<KernelRegistry> VitisAIExecutionProvider::GetKernelRegistry() const { return get_kernel_registry_vitisaiep(); }

// This method is called after both `GetComputeCapabilityOps()` and `Compile()`.
// This timing is required to work with both compliation-based EPs and non-compilation-based EPs.
const InlinedVector<const Node*> VitisAIExecutionProvider::GetEpContextNodes() const {
  InlinedVector<const Node*> ep_context_node_ptrs;
  // All preconditions are supposed to have happened.
  if (p_ep_ctx_model_) {
    auto& graph = p_ep_ctx_model_->MainGraph();
    for (const auto* p_node : graph.Nodes()) {
      ep_context_node_ptrs.push_back(p_node);
    }
  }
  return ep_context_node_ptrs;
}

// Create EP context model and dump it for future use.
// This implementation here is only working for non-compilation-based EPs.
// This version of implementation (vs the overloaded version of implementation below)
// is more universally applicable and less coupled with the closed-source backend of VitisAI EP.
// The two vesions have respective pros and cons.
void VitisAIExecutionProvider::FulfillEPContextEnablement(
    const std::vector<std::unique_ptr<ComputeCapability>>& capability_ptrs,
    const onnxruntime::GraphViewer& graph_viewer) const {
  auto& logger = logging::LoggingManager::DefaultLogger();
  auto model_path_str = GetTopLevelModelPath(graph_viewer).ToPathString();
  auto ep_ctx_payload = SerializeCapabilities(capability_ptrs, graph_viewer.GetGraph());
  if (!ep_ctx_embed_mode_) {
    if (!GetEPContextModelFileLocation(ep_ctx_model_path_cfg_, model_path_str, false, ep_ctx_model_file_loc_)) {
      ORT_THROW("Failed to figure out a path for storing the EP-context ONNX model");
    }
    auto ep_ctx_cache_path_str = GetEPContextCacheFileLocation(ep_ctx_model_file_loc_, model_path_str);
    std::ofstream ep_ctx_cache_ofs(ep_ctx_cache_path_str.c_str(), std::ios::trunc | std::ios::binary);
    if (!ep_ctx_cache_ofs.is_open()) {
      ORT_THROW("Failed to open a file to write EP context cache: ", ep_ctx_cache_path_str.c_str());
    }
    ep_ctx_cache_ofs.write(ep_ctx_payload.c_str(), ep_ctx_payload.length());
    if (!ep_ctx_cache_ofs.good()) {
      ep_ctx_cache_ofs.close();
      ORT_THROW("Exception writing EP context cache file: ", ep_ctx_cache_path_str.c_str());
    }
    ep_ctx_cache_ofs.close();
    p_ep_ctx_model_ = CreateEPContexModel(graph_viewer, "", ep_ctx_cache_path_str, 0, &logger);
  } else {
    p_ep_ctx_model_ = CreateEPContexModel(graph_viewer, ep_ctx_payload, "", 1, &logger);
  }
  DumpEPContextModel(p_ep_ctx_model_, ep_ctx_model_file_loc_);
}

// This version of implementation (vs the overloaded version of implementation above)
// is more VitisAI specific and more tightly coupled with the closed-source backend of VitisAI EP.
// The two vesions have respective pros and cons.
void VitisAIExecutionProvider::FulfillEPContextEnablement(
    const onnxruntime::GraphViewer& graph_viewer) const {
  auto cache_dir = GetBackendCompileCacheDir();
  auto cache_key = GetBackendCompileCacheKey(graph_viewer);
  fs::path backend_cache_file_loc(cache_dir + '/' + cache_key + "/context.json");
  auto backend_cache_str = GetBackendCompileCache(backend_cache_file_loc);
  auto& logger = logging::LoggingManager::DefaultLogger();
  auto model_path_str = GetTopLevelModelPath(graph_viewer).ToPathString();
  if (!ep_ctx_embed_mode_) {
    if (!GetEPContextModelFileLocation(ep_ctx_model_path_cfg_, model_path_str, false, ep_ctx_model_file_loc_)) {
      ORT_THROW("Failed to figure out a path for storing the EP-context ONNX model");
    }
    auto ep_ctx_cache_path_str = GetEPContextCacheFileLocation(ep_ctx_model_file_loc_, model_path_str);
    std::ofstream ep_ctx_cache_ofs(ep_ctx_cache_path_str.c_str(), std::ios::trunc);
    if (!ep_ctx_cache_ofs.is_open()) {
      ORT_THROW("Failed to open a file to write EP context cache: ", ep_ctx_cache_path_str.c_str());
    }
    ep_ctx_cache_ofs.write(backend_cache_str.c_str(), backend_cache_str.length());
    if (!ep_ctx_cache_ofs.good()) {
      ep_ctx_cache_ofs.close();
      ORT_THROW("Exception writing EP context cache file: ", ep_ctx_cache_path_str.c_str());
    }
    ep_ctx_cache_ofs.close();
    p_ep_ctx_model_ = CreateEPContexModel(graph_viewer, "", ep_ctx_cache_path_str, 0, &logger);
  } else {
    p_ep_ctx_model_ = CreateEPContexModel(graph_viewer, backend_cache_str, "", 1, &logger);
  }
  DumpEPContextModel(p_ep_ctx_model_, ep_ctx_model_file_loc_);
}

std::vector<std::unique_ptr<ComputeCapability>> VitisAIExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer, const IKernelLookup& /*kernel_lookup*/) const {
  bool is_ep_ctx_model = GraphHasEPContextNode(graph_viewer);
  auto model_path_str = GetTopLevelModelPath(graph_viewer).ToPathString();
  // XXX: One of the potential problems is the existing EP-context model file may be stale.
  if (GetEPContextModelFileLocation(
          ep_ctx_model_path_cfg_, model_path_str, is_ep_ctx_model, ep_ctx_model_file_loc_)) {
#if 0
    // XXX: For now we are intentionally keeping this part.
    // This part is corresponding to the 1st version of `FulfillEPContextEnablement()`.
    // Once we are done with the verification of functionalities and performance
    // of both implementations, we may eliminate this part.
    if (is_ep_ctx_model) {
      auto ep_ctx_payload = RetrieveEPContextCache(graph_viewer.GetGraph());
      std::vector<std::unique_ptr<ComputeCapability>> capability_ptrs;
      DeserializeCapabilities(ep_ctx_payload, capability_ptrs);
      // FIXME
      // 1) `execution_providers_` is used by `Compiler()` as well, so, even in this case, we need to make it ready.
      // 2) Computing (using cache) `execution_providers_` needs the original model/graph for signature match.
      // 3) The closed-source backend of VitisAI EP has complication cache, so no real overhead.
      // 4) In next iteration of EP context model implementaiton, we are getting rid of this dependency.
      if (!execution_providers_) {
        auto p_orig_graph_viewer = RetrieveOriginalGraph(graph_viewer.GetGraph(), ep_ctx_model_file_loc_);
        execution_providers_ = std::make_unique<my_ep_t>(compile_onnx_model(*p_orig_graph_viewer, *GetLogger(), info_));
      }
      return capability_ptrs;
    } else {
      LOGS_DEFAULT(WARNING) << "The inference session was created with a normal ONNX model "
                            << "but a model file with EP context cache exists at " << ep_ctx_model_file_loc_.c_str();
      LoadEPContexModelFromFile();
      auto ep_ctx_payload = RetrieveEPContextCache(p_ep_ctx_model_->MainGraph(), ep_ctx_model_file_loc_);
      std::vector<std::unique_ptr<ComputeCapability>> capability_ptrs;
      DeserializeCapabilities(ep_ctx_payload, capability_ptrs);
      // FIXME: ditto.
      if (!execution_providers_) {
        execution_providers_ = std::make_unique<my_ep_t>(compile_onnx_model(graph_viewer, *GetLogger(), info_));
        // Alternative with some nuance.
        // auto p_orig_graph_viewer = RetrieveOriginalGraph(p_ep_ctx_model_->MainGraph());
        // execution_providers_ = std::make_unique<my_ep_t>(compile_onnx_model(*p_orig_graph_viewer, *GetLogger(), info_));
      }
      return capability_ptrs;
    }
#endif
    // This part is corresponding to the 2nd version of `FulfillEPContextEnablement()`.
    if (is_ep_ctx_model) {
      auto cache_dir = GetBackendCompileCacheDir();
      auto cache_key = GetBackendCompileCacheKey(graph_viewer);
      fs::path backend_cache_file_loc(cache_dir + "/" + cache_key + "/context.json");
      LOGS_DEFAULT(VERBOSE) << "Trying getting compilation cache from " << backend_cache_file_loc.string();
      auto ep_ctx_payload = RetrieveEPContextCache(graph_viewer.GetGraph(), ep_ctx_model_file_loc_, false);
      RestoreBackendCompileCache(backend_cache_file_loc, ep_ctx_payload);
    } else {
      LOGS_DEFAULT(WARNING) << "The inference session was created with a normal ONNX model "
                            << "but a model file with EP context cache exists at " << ep_ctx_model_file_loc_.c_str();
      LoadEPContexModelFromFile();
      auto cache_dir = GetBackendCompileCacheDir();
      auto cache_key = GetBackendCompileCacheKey(graph_viewer);
      fs::path backend_cache_file_loc(cache_dir + '/' + cache_key + "/context.json");
      LOGS_DEFAULT(VERBOSE) << "Trying getting compilation cache from " << backend_cache_file_loc.string();
      auto ep_ctx_payload = RetrieveEPContextCache(p_ep_ctx_model_->MainGraph(), ep_ctx_model_file_loc_, false);
      RestoreBackendCompileCache(backend_cache_file_loc, ep_ctx_payload);
    }
  } else {
    LOGS_DEFAULT(WARNING) << "Failed to get EP context model file";
  }

  if (graph_viewer.IsSubgraph()) {
    // VITIS AI EP not support sungraph. Assigned to CPU.
    return {};
  }
  if (execution_providers_) {
    // Only compiling a model once is currently supported
    return {};
  }
  execution_providers_ = std::make_unique<my_ep_t>(compile_onnx_model(graph_viewer, *GetLogger(), info_));
  auto result = vaip::GetComputeCapabilityOps(graph_viewer, execution_providers_.get(), vitisai_optypes_);
  size_t index = 0u;
  for (auto& ep : **execution_providers_) {
    result.emplace_back(vaip::XirSubgraphToComputeCapability1(graph_viewer, ep.get(), index));
    index = index + 1;
  }
  if (ep_ctx_enabled_) {
#if 0
    FulfillEPContextEnablement(result, graph_viewer);
#endif
    FulfillEPContextEnablement(graph_viewer);
  }
  return result;
}

common::Status VitisAIExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                                 std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_graph : fused_nodes_and_graphs) {
    NodeComputeInfo compute_info;
    auto& attrs = fused_node_graph.fused_node.get().GetAttributes();
    assert(attrs.count("index"));
    size_t index = attrs.at("index").i();
    compute_info.create_state_func = [this, index](ComputeContext* context, FunctionState* state) {
      auto* p = (**this->execution_providers_)[index]->compile().release();
      *state = p;
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      if (state) {
        delete reinterpret_cast<vaip_core::CustomOp*>(state);
      }
    };
    compute_info.compute_func = [](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
      reinterpret_cast<vaip_core::CustomOp*>(state)->Compute(api, context);
      return Status::OK();
    };
    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}

std::string VitisAIExecutionProvider::GetBackendCompileCacheDir() const {
  if (info_.count("cacheDir") > 0) {
    const std::string& cache_dir = info_.at("cacheDir");
    if (!cache_dir.empty()) {
      return cache_dir;
    }
  }
  std::string cache_dir = ParseEnvironmentVariableWithDefault<std::string>("XLNX_CACHE_DIR", "");
  if (!cache_dir.empty()) {
    return cache_dir;
  }
  auto user_name = ParseEnvironmentVariableWithDefault<std::string>(
      "USERNAME", ParseEnvironmentVariableWithDefault<std::string>("USER", ""));
  std::string temp_dir =
#ifdef _WIN32
      "C:/temp/";
#else
      "/tmp/";
#endif
  if (!user_name.empty()) {
    temp_dir.append(user_name);
  }
  temp_dir.append("/vaip/.cache");
  return temp_dir;
}

std::string VitisAIExecutionProvider::GetBackendCompileCacheKey(
    const GraphViewer& graph_viewer) const {
  if (info_.count("cacheKey") > 0) {
    std::string& cache_key = info_.at("cacheKey");
    if (!cache_key.empty()) {
      return cache_key;
    }
  }
  // Model metadata key "vaip_model_md5sum".
  const auto& graph = graph_viewer.GetGraph();
  const auto& model_metadata = graph.GetModel().MetaData();
  if (model_metadata.count("vaip_model_md5sum") > 0) {
    return model_metadata.at("vaip_model_md5sum");
  }
  if (ParseEnvironmentVariableWithDefault<std::string>(
          "XLNX_ENABLE_FILE_BASED_CACHE_KEY", "0") != "0") {
    const Path& model_path = graph_viewer.ModelPath();
    if (!model_path.IsEmpty()) {
      return HashFileContentWithMD5(PathToUTF8String(model_path.ToPathString()));
    }
  }
  return GetModelSignature(graph_viewer);
}

}  // namespace onnxruntime
