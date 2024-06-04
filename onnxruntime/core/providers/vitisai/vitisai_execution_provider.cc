// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "vitisai_execution_provider.h"

// Standard headers/libs.
#include <cassert>
#include <fstream>
#include <istream>
#include <filesystem>

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

#if 0
  if (p_sess_opts) {
    ep_ctx_enabled_ = p_sess_opts->config_options.GetConfigOrDefault(
        kOrtSessionOptionEpContextEnable, "0") == "1";
    std::string embed_mode = p_sess_opts->config_options.GetConfigOrDefault(
        kOrtSessionOptionEpContextEmbedMode, "1");
    if ("1" == embed_mode) {
      ep_ctx_embed_mode_ = true;
    } else if ("0" == embed_mode) {
      ep_ctx_embed_mode_ = false;
    } else {
      LOGS_DEFAULT(VERBOSE) << "Invalid ep.context_embed_mode: " << embed_mode << " only 0 or 1 allowed. Set to 1.";
      ep_ctx_embed_mode_ = true;
    }
    ep_ctx_model_path_cfg_ = p_sess_opts->config_options.GetConfigOrDefault(
        kOrtSessionOptionEpContextFilePath, "");
  } else {
#endif
  auto it = info_.find("ep_context_enable");
  ep_ctx_enabled_ = it != info_.end() && it->second == "1";
  it = info_.find("ep_context_embed_mode");
  ep_ctx_embed_mode_ = it != info_.end() && it->second != "0";
  // ep_ctx_embed_mode_ = it == info_.end() || it->second != "0";
  it = info_.find("ep_context_file_path");
  ep_ctx_model_path_cfg_ = it == info_.end() ? "" : it->second;
#if 0
  }
#endif
  LOGS_DEFAULT(VERBOSE) << "EP Context cache enabled: " << ep_ctx_enabled_;
  LOGS_DEFAULT(VERBOSE) << "EP context cache embed mode: " << ep_ctx_embed_mode_;
  LOGS_DEFAULT(VERBOSE) << "User specified EP context cache path: " << ep_ctx_model_path_cfg_;
}

#if 0
VitisAIExecutionProvider::~VitisAIExecutionProvider() {
  // TODO: EP context related sources.
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
void VitisAIExecutionProvider::FulfillEPContextEnablement(
    const std::vector<std::unique_ptr<ComputeCapability>>& capability_ptrs,
    const onnxruntime::GraphViewer& graph_viewer) const {
  auto& logger = logging::LoggingManager::DefaultLogger();
  auto model_path_str = GetTopLevelModelPath(graph_viewer).ToPathString();
  auto ep_ctx_payload = SerializeCapabilities(capability_ptrs, graph_viewer.GetGraph());
  if (!ep_ctx_embed_mode_) {
    GetEPContextModelFileLocation(ep_ctx_model_path_cfg_, model_path_str, false, ep_ctx_model_file_loc_);
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
    p_ep_ctx_model_ = CreateEPContexModel(graph_viewer, ep_ctx_payload, ep_ctx_cache_path_str, 0, &logger);
  } else {
    p_ep_ctx_model_ = CreateEPContexModel(graph_viewer, ep_ctx_payload, "", 1, &logger);
  }
  DumpEPContextModel(p_ep_ctx_model_, ep_ctx_model_file_loc_);
}

std::vector<std::unique_ptr<ComputeCapability>> VitisAIExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer, const IKernelLookup& /*kernel_lookup*/) const {
  bool is_ep_ctx_model = GraphHasEPContextNode(graph_viewer);
  if (is_ep_ctx_model) {
    auto ep_ctx_payload = RetrieveEPContextCache(graph_viewer.GetGraph());
    std::vector<std::unique_ptr<ComputeCapability>> capability_ptrs;
    DeserializeCapabilities(ep_ctx_payload, capability_ptrs);
    return capability_ptrs;
  } else {
    // FIXME: Will it make sense to do this?
    // One of the potential problems is the existing EP-context model file may be stale.
    auto model_path_str = GetTopLevelModelPath(graph_viewer).ToPathString();
    if (GetEPContextModelFileLocation(
            ep_ctx_model_path_cfg_, model_path_str, false, ep_ctx_model_file_loc_)) {
      LOGS_DEFAULT(WARNING) << "The inference session was created with a normal ONNX model "
                            << "but a model file with EP context cache exists at " << ep_ctx_model_file_loc_.c_str();
      LoadEPContexModelFromFile();
      auto ep_ctx_payload = RetrieveEPContextCache(p_ep_ctx_model_->MainGraph());
      std::vector<std::unique_ptr<ComputeCapability>> capability_ptrs;
      DeserializeCapabilities(ep_ctx_payload, capability_ptrs);
      return capability_ptrs;
    }
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
    FulfillEPContextEnablement(result, graph_viewer);
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
}  // namespace onnxruntime
