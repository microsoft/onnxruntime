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
#include "core/common/exceptions.h"

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
    auto status = Model::Load(ep_ctx_model_file_loc_, *p_ep_ctx_model_proto_);
    if (!status.IsOK()) {
      ORT_THROW("Loading EP context model failed from ", PathToUTF8String(ep_ctx_model_file_loc_));
    }
    p_ep_ctx_model_ = Model::Create(std::move(*p_ep_ctx_model_proto_), ep_ctx_model_file_loc_, nullptr, *GetLogger());
    LOGS_DEFAULT(VERBOSE) << "Loaded EP context model from: " << PathToUTF8String(ep_ctx_model_file_loc_);
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

#if 0
// ONLY uncomment this method when the "Approach 3" mentioned below is switched ON.
// This method is called after both `GetComputeCapabilityOps()` and `Compile()`.
// This timing is required to work with both compilation-based EPs and non-compilation-based EPs.
const InlinedVector<const Node*> VitisAIExecutionProvider::GetEpContextNodes() const {
  LOGS_DEFAULT(VERBOSE) << "`IExecutionProvider::GetEpContextNodes()` is called";
  InlinedVector<const Node*> ep_context_node_ptrs;
  // All preconditions are supposed to have happened.
  if (p_ep_ctx_model_) {
    auto& graph = p_ep_ctx_model_->MainGraph();
    LOGS_DEFAULT(VERBOSE) << "Collecting EP context nodes: " << graph.Nodes().size();
    for (const auto* p_node : graph.Nodes()) {
      ep_context_node_ptrs.push_back(p_node);
    }
  }
  return ep_context_node_ptrs;
}
#endif

// For "Approach 1".
void VitisAIExecutionProvider::FulfillEPContextEnablement(
    const std::vector<std::unique_ptr<ComputeCapability>>& capability_ptrs,
    const onnxruntime::GraphViewer& graph_viewer) const {
  LOGS_DEFAULT(VERBOSE) << "Trying creating EP context model";
  if (model_path_str_.empty()) {
    model_path_str_ = GetTopLevelModelPath(graph_viewer).ToPathString();
  }
  if (ep_ctx_model_file_loc_.empty()) {
    if (!GetEPContextModelFileLocation(ep_ctx_model_path_cfg_, model_path_str_, false, ep_ctx_model_file_loc_)) {
      ORT_THROW("Failed to figure out a path for storing the EP-context ONNX model");
    }
  }
  auto ep_ctx_payload = SerializeCapabilities(capability_ptrs, graph_viewer.GetGraph());
  LOGS_DEFAULT(VERBOSE) << "Done serialized capabilities with byte length: " << ep_ctx_payload.length();
  if (!ep_ctx_embed_mode_) {
    auto ep_ctx_cache_path_str = GetEPContextCacheFileLocation(ep_ctx_model_file_loc_, model_path_str_);
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
    p_ep_ctx_model_proto_.reset(CreateEPContexModel(graph_viewer, "", PathToUTF8String(ep_ctx_cache_path_str), 0, "", "", true, GetLogger()));
    p_ep_ctx_model_ = Model::Create(std::move(*p_ep_ctx_model_proto_), ep_ctx_model_file_loc_, nullptr, *GetLogger());
  } else {
    p_ep_ctx_model_proto_.reset(CreateEPContexModel(graph_viewer, ep_ctx_payload, "", 1, "", "", true, GetLogger()));
    p_ep_ctx_model_ = Model::Create(std::move(*p_ep_ctx_model_proto_), ep_ctx_model_file_loc_, nullptr, *GetLogger());
  }
  LOGS_DEFAULT(VERBOSE) << "EP context modeld created";
  DumpEPContextModel(p_ep_ctx_model_proto_, PathToUTF8String(ep_ctx_model_file_loc_));
}

// For "Approach 2".
void VitisAIExecutionProvider::FulfillEPContextEnablement(
    const onnxruntime::GraphViewer& graph_viewer) const {
  LOGS_DEFAULT(VERBOSE) << "Trying creating EP context model";
  auto cache_dir = GetBackendCompileCacheDir();
  auto cache_key = GetBackendCompileCacheKey(graph_viewer);
  LOGS_DEFAULT(VERBOSE) << "Cache dir: " << cache_dir << ". Cache key: " << cache_key;
  info_["cacheDir"] = cache_dir;
  info_["cacheKey"] = cache_key;
  fs::path backend_cache_file_loc(cache_dir + '/' + cache_key + "/context.json");
  auto backend_cache_str = GetBackendCompileCache(backend_cache_file_loc);
  if (model_path_str_.empty()) {
    model_path_str_ = GetTopLevelModelPath(graph_viewer).ToPathString();
  }
  if (ep_ctx_model_file_loc_.empty()) {
    if (!GetEPContextModelFileLocation(ep_ctx_model_path_cfg_, model_path_str_, false, ep_ctx_model_file_loc_)) {
      ORT_THROW("Failed to figure out a path for storing the EP-context ONNX model");
    }
  }
  if (!ep_ctx_embed_mode_) {
    auto ep_ctx_cache_path_str = GetEPContextCacheFileLocation(ep_ctx_model_file_loc_, model_path_str_);
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
    p_ep_ctx_model_proto_.reset(CreateEPContexModel(graph_viewer, "", PathToUTF8String(ep_ctx_cache_path_str), 0, cache_dir, cache_key, false, GetLogger()));
    p_ep_ctx_model_ = Model::Create(std::move(*p_ep_ctx_model_proto_), ep_ctx_model_file_loc_, nullptr, *GetLogger());
    if (GraphHasEPContextNode(p_ep_ctx_model_->MainGraph())) {
      LOGS_DEFAULT(VERBOSE) << "Created model has EP context nodes";
    }
  } else {
    p_ep_ctx_model_proto_.reset(CreateEPContexModel(graph_viewer, backend_cache_str, "", 1, cache_dir, cache_key, false, GetLogger()));
    p_ep_ctx_model_ = Model::Create(std::move(*p_ep_ctx_model_proto_), ep_ctx_model_file_loc_, nullptr, *GetLogger());
    if (GraphHasEPContextNode(p_ep_ctx_model_->MainGraph())) {
      LOGS_DEFAULT(VERBOSE) << "Created model has EP context nodes";
    }
  }
  LOGS_DEFAULT(VERBOSE) << "EP context model created";
  DumpEPContextModel(p_ep_ctx_model_proto_, PathToUTF8String(ep_ctx_model_file_loc_));
}

void VitisAIExecutionProvider::PrepareEPContextEnablement(
    const onnxruntime::GraphViewer& graph_viewer) const {
  auto cache_dir = GetBackendCompileCacheDir();
  auto cache_key = GetBackendCompileCacheKey(graph_viewer);
  LOGS_DEFAULT(VERBOSE) << "Cache dir: " << cache_dir << ". Cache key: " << cache_key;
  // This way enforces the backend to use this cache dir and cache key.
  info_["cacheDir"] = cache_dir;
  info_["cacheKey"] = cache_key;
  if (model_path_str_.empty()) {
    model_path_str_ = GetTopLevelModelPath(graph_viewer).ToPathString();
  }
  LOGS_DEFAULT(VERBOSE) << "Original model path: " << PathToUTF8String(model_path_str_);
  // Create a new model, reusing the graph name, the op-domain-to-opset-version map,
  // the op schema registry of the current graph, etc.
  p_ep_ctx_model_ = graph_viewer.CreateModel(*GetLogger());
  if (ep_ctx_model_file_loc_.empty()) {
    if (!GetEPContextModelFileLocation(ep_ctx_model_path_cfg_, model_path_str_, false, ep_ctx_model_file_loc_)) {
      ORT_THROW("Failed to figure out a path for storing the EP-context ONNX model");
    }
  }
}

// For "Approach 3".
void VitisAIExecutionProvider::FulfillEPContextEnablement(
    const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs) {
  LOGS_DEFAULT(VERBOSE) << "Trying creating EP context model";
  auto& ep_ctx_graph = p_ep_ctx_model_->MainGraph();
  fs::path backend_cache_file_loc(info_["cacheDir"] + '/' + info_["cacheKey"] + "/context.json");
  auto backend_cache_str = GetBackendCompileCache(backend_cache_file_loc);
  if (!ep_ctx_embed_mode_) {
    auto ep_ctx_cache_path_str = GetEPContextCacheFileLocation(ep_ctx_model_file_loc_, model_path_str_);
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
    CreateEPContexNodes(&ep_ctx_graph, fused_nodes_and_graphs, "", PathToUTF8String(ep_ctx_cache_path_str), 0, info_["cacheDir"], info_["cacheKey"], false, GetLogger());
    if (GraphHasEPContextNode(ep_ctx_graph)) {
      LOGS_DEFAULT(VERBOSE) << "Created model has EP context nodes";
    }
  } else {
    CreateEPContexNodes(&ep_ctx_graph, fused_nodes_and_graphs, backend_cache_str, "", 1, info_["cacheDir"], info_["cacheKey"], false, GetLogger());
    if (GraphHasEPContextNode(ep_ctx_graph)) {
      LOGS_DEFAULT(VERBOSE) << "Created model has EP context nodes";
    }
  }
  LOGS_DEFAULT(VERBOSE) << "EP context model created";
}

#if 0
// Approach 1 for making an EP context model:
// 1)
// Directly serializes and caches the list of compute capabilities.
// This makes sense because VitisAI EP does heavy lifting work in `GetCapability()`.
// 2)
// In order for `Compile()` to work, serializes the original model/graph in the EP context node, which will be used for
// (a) making custom `execution_providers_`, and (b) making signatures used by closed-source backend compilers.
// 3)
// Does not touch the VitisAI EP specific backend compilation cache,
// so this approach is not tightly coupled with VitisAI EP and could universally be applied.
// 4)
// Can not achieve "compile once, run everywhere".
// 5)
// Creates and dumps the EP context model at the end of `GetCapability()`.
// 6)
// Does not (and even should not) implement/override `IExecutionProvider::GetEpContextNodes()`,
// so dumps the EP context model by itself.
std::vector<std::unique_ptr<ComputeCapability>> VitisAIExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer, const IKernelLookup& /*kernel_lookup*/) const {
  bool is_ep_ctx_model = GraphHasEPContextNode(graph_viewer.GetGraph());
  model_path_str_ = GetTopLevelModelPath(graph_viewer).ToPathString();
  LOGS_DEFAULT(VERBOSE) << "Loaded model path: " << model_path_str_.c_str();
  if (GetEPContextModelFileLocation(
          ep_ctx_model_path_cfg_, model_path_str_, is_ep_ctx_model, ep_ctx_model_file_loc_)) {
    if (is_ep_ctx_model) {
      LOGS_DEFAULT(VERBOSE) << "An EP context model passed in";
      ValidateEPContextNode(graph_viewer.GetGraph());
      if (!execution_providers_) {
        auto p_orig_graph_viewer = RetrieveOriginalGraph(graph_viewer.GetGraph());
        LOGS_DEFAULT(VERBOSE) << "Creating custom execution providers";
        try {
          execution_providers_ = std::make_unique<my_ep_t>(compile_onnx_model(*p_orig_graph_viewer, *GetLogger(), info_));
        } catch (const std::exception& ex) {
          LOGS_DEFAULT(WARNING) << "Backend compilation failed: " << ex.what();
          throw;
        } catch (...) {
          LOGS_DEFAULT(WARNING) << "Backend compilation failed: unknown exception";
          throw;
        }
        LOGS_DEFAULT(VERBOSE) << "Created custom execution providers";
      }
      std::vector<std::unique_ptr<ComputeCapability>> capability_ptrs;
      auto ep_ctx_payload = RetrieveEPContextCache(graph_viewer.GetGraph(), ep_ctx_model_file_loc_);
      LOGS_DEFAULT(VERBOSE) << "Retrieved cache with byte length " << ep_ctx_payload.length();
      if (!ep_ctx_payload.empty()) {
        DeserializeCapabilities(ep_ctx_payload, capability_ptrs);
        LOGS_DEFAULT(VERBOSE) << "Deserialized ComputeCapability";
      } else {
        LOGS_DEFAULT(WARNING) << "[VitisAI EP] The original graph generated zero compute capabilities so now no cached available";
      }
      return capability_ptrs;
    } else {
      if (fs::exists(ep_ctx_model_file_loc_) && fs::is_regular_file(ep_ctx_model_file_loc_) && ep_ctx_enabled_) {
        ORT_THROW("The inference session was created with a normal ONNX model but a model file with EP context cache exists at ",
                  PathToUTF8String(ep_ctx_model_file_loc_), ". Please remove the EP context model manually if you want to re-generate it.");
        // Disable the flexibility implemented below by throwing an exception.
        // Now the code below is unreachable but DCE will take care of it.
        // We might want to re-enable it in future, so we keep it as is.
        LoadEPContexModelFromFile();
        ValidateEPContextNode(p_ep_ctx_model_->MainGraph());
        if (!execution_providers_) {
          auto p_orig_graph_viewer = RetrieveOriginalGraph(p_ep_ctx_model_->MainGraph());
          execution_providers_ = std::make_unique<my_ep_t>(compile_onnx_model(*p_orig_graph_viewer, *GetLogger(), info_));
        }
        auto ep_ctx_payload = RetrieveEPContextCache(p_ep_ctx_model_->MainGraph(), ep_ctx_model_file_loc_);
        std::vector<std::unique_ptr<ComputeCapability>> capability_ptrs;
        DeserializeCapabilities(ep_ctx_payload, capability_ptrs);
        LOGS_DEFAULT(VERBOSE) << "Deserialized ComputeCapability";
        return capability_ptrs;
      }
    }
  } else {
    LOGS_DEFAULT(WARNING) << "Failed to get EP context model file location";
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
  if (ep_ctx_enabled_ && !is_ep_ctx_model) {
    FulfillEPContextEnablement(result, graph_viewer);
  }
  return result;
}
#endif

#if 1
// Approach 2 for making an EP context model:
// 1)
// Can achieve "compile once, run everywhere".
// 2)
// Uses exactly the same alorithm and logic as the closed-source backend to
// get the cache dir and cache key for closed-source backend compilers.
// 3)
// Caches both the backend compilation cache and the aforementioned cache dir/key in the EP context node.
// 4)
// Is tightly coupled with VitisAI EP.
// 5)
// Creates and dumps the EP context model at the end of `GetCapability()`.
// 6)
// Does not (and even should not) implement/override `IExecutionProvider::GetEpContextNodes()`,
// so dumps the EP context model by itself.
std::vector<std::unique_ptr<ComputeCapability>> VitisAIExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer, const IKernelLookup& /*kernel_lookup*/) const {
  bool is_ep_ctx_model = GraphHasEPContextNode(graph_viewer.GetGraph());
  model_path_str_ = GetTopLevelModelPath(graph_viewer).ToPathString();
  LOGS_DEFAULT(VERBOSE) << "Loaded model path: " << model_path_str_.c_str();
  if (GetEPContextModelFileLocation(
          ep_ctx_model_path_cfg_, model_path_str_, is_ep_ctx_model, ep_ctx_model_file_loc_)) {
    if (is_ep_ctx_model) {
      LOGS_DEFAULT(VERBOSE) << "An EP context model passed in";
      ValidateEPContextNode(graph_viewer.GetGraph());
      std::string cache_dir, cache_key;
      RetrieveBackendCacheInfo(graph_viewer.GetGraph(), cache_dir, cache_key);
      LOGS_DEFAULT(VERBOSE) << "Cache dir: " << cache_dir << ". Cache key: " << cache_key;
      // This way enforces the backend to use this cache dir and cache key.
      info_["cacheDir"] = cache_dir;
      info_["cacheKey"] = cache_key;
      fs::path backend_cache_file_loc(cache_dir + "/" + cache_key + "/context.json");
      if (!fs::exists(backend_cache_file_loc) || !fs::is_regular_file(backend_cache_file_loc) || fs::file_size(backend_cache_file_loc) == 0) {
        LOGS_DEFAULT(VERBOSE) << "Trying getting compilation cache from " << PathToUTF8String(ep_ctx_model_file_loc_);
        auto ep_ctx_payload = RetrieveEPContextCache(graph_viewer.GetGraph(), ep_ctx_model_file_loc_, false);
        RestoreBackendCompileCache(backend_cache_file_loc, ep_ctx_payload);
      } else {
        LOGS_DEFAULT(VERBOSE) << "Backend compilation cache file already there " << backend_cache_file_loc.string();
      }
    } else {
      if (fs::exists(ep_ctx_model_file_loc_) && fs::is_regular_file(ep_ctx_model_file_loc_) && ep_ctx_enabled_) {
        ORT_THROW("The inference session was created with a normal ONNX model but a model file with EP context cache exists at ",
                  PathToUTF8String(ep_ctx_model_file_loc_), ". Please remove the EP context model manually if you want to re-generate it.");
        // Disable the flexibility implemented below by throwing an exception.
        // Now the code below is unreachable but DCE will take care of it.
        // We might want to re-enable it in future, so we keep it as is.
        LoadEPContexModelFromFile();
        ValidateEPContextNode(p_ep_ctx_model_->MainGraph());
        auto cache_dir = GetBackendCompileCacheDir();
        auto cache_key = GetBackendCompileCacheKey(graph_viewer);
        LOGS_DEFAULT(VERBOSE) << "Cache dir: " << cache_dir << ". Cache key: " << cache_key;
        // This way enforces the backend to use this cache dir and cache key.
        info_["cacheDir"] = cache_dir;
        info_["cacheKey"] = cache_key;
        fs::path backend_cache_file_loc(cache_dir + '/' + cache_key + "/context.json");
        if (!fs::exists(backend_cache_file_loc) || !fs::is_regular_file(backend_cache_file_loc) || fs::file_size(backend_cache_file_loc) == 0) {
          LOGS_DEFAULT(VERBOSE) << "Trying getting compilation cache from " << PathToUTF8String(ep_ctx_model_file_loc_);
          auto ep_ctx_payload = RetrieveEPContextCache(p_ep_ctx_model_->MainGraph(), ep_ctx_model_file_loc_, false);
          RestoreBackendCompileCache(backend_cache_file_loc, ep_ctx_payload);
        } else {
          LOGS_DEFAULT(VERBOSE) << "Backend compilation cache file already there " << backend_cache_file_loc.string();
        }
      }
    }
  } else {
    LOGS_DEFAULT(WARNING) << "Failed to get EP context model file location";
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
  if (ep_ctx_enabled_ && !is_ep_ctx_model) {
    FulfillEPContextEnablement(graph_viewer);
  }
  return result;
}
#endif

#if 0
// Approach 3 for making an EP context model:
// 0)
// Note: This will NOT happen if `GetCapability()` returns zero compute capabilities - which makes sense.
// 1)
// Can achieve "compile once, run everywhere".
// 2)
// Uses exactly the same alorithm and logic as the closed-source backend to
// get the cache dir and cache key for closed-source backend compilers.
// 3)
// Caches both the backend compilation cache and the aforementioned cache dir/key in the EP context node.
// 4)
// Is tightly coupled with VitisAI EP.
// 5)
// Creates the EP context model at the end of `Compile()` instead of `GetCapability()`.
// 6)
// Implements/overrides `IExecutionProvider::GetEpContextNodes()`, so dumps the EP context model
// in `CreateEpContextModel()` in the file "graph_partitioner.cc"
std::vector<std::unique_ptr<ComputeCapability>> VitisAIExecutionProvider::GetCapability(
    const onnxruntime::GraphViewer& graph_viewer, const IKernelLookup& /*kernel_lookup*/) const {
  bool is_ep_ctx_model = GraphHasEPContextNode(graph_viewer.GetGraph());
  model_path_str_ = GetTopLevelModelPath(graph_viewer).ToPathString();
  LOGS_DEFAULT(VERBOSE) << "Loaded model path: " << model_path_str_.c_str();
  if (GetEPContextModelFileLocation(
          ep_ctx_model_path_cfg_, model_path_str_, is_ep_ctx_model, ep_ctx_model_file_loc_)) {
    if (is_ep_ctx_model) {
      LOGS_DEFAULT(VERBOSE) << "An EP context model passed in";
      ValidateEPContextNode(graph_viewer.GetGraph());
      std::string cache_dir, cache_key;
      RetrieveBackendCacheInfo(graph_viewer.GetGraph(), cache_dir, cache_key);
      LOGS_DEFAULT(VERBOSE) << "Cache dir: " << cache_dir << ". Cache key: " << cache_key;
      // This way enforces the backend to use this cache dir and cache key.
      info_["cacheDir"] = cache_dir;
      info_["cacheKey"] = cache_key;
      fs::path backend_cache_file_loc(cache_dir + "/" + cache_key + "/context.json");
      if (!fs::exists(backend_cache_file_loc) || !fs::is_regular_file(backend_cache_file_loc) || fs::file_size(backend_cache_file_loc) == 0) {
        LOGS_DEFAULT(VERBOSE) << "Trying getting compilation cache from " << PathToUTF8String(ep_ctx_model_file_loc_);
        auto ep_ctx_payload = RetrieveEPContextCache(graph_viewer.GetGraph(), ep_ctx_model_file_loc_, false);
        RestoreBackendCompileCache(backend_cache_file_loc, ep_ctx_payload);
      } else {
        LOGS_DEFAULT(VERBOSE) << "Backend compilation cache file already there " << backend_cache_file_loc.string();
      }
    } else {
      if (fs::exists(ep_ctx_model_file_loc_) && fs::is_regular_file(ep_ctx_model_file_loc_) && ep_ctx_enabled_) {
        ORT_THROW("The inference session was created with a normal ONNX model but a model file with EP context cache exists at ",
                  PathToUTF8String(ep_ctx_model_file_loc_), ". Please remove the EP context model manually if you want to re-generate it.");
        // Disable the flexibility implemented below by throwing an exception.
        // Now the code below is unreachable but DCE will take care of it.
        // We might want to re-enable it in future, so we keep it as is.
        LoadEPContexModelFromFile();
        ValidateEPContextNode(p_ep_ctx_model_->MainGraph());
        // This way enforces the backend to use this cache dir and cache key.
        auto cache_dir = GetBackendCompileCacheDir();
        auto cache_key = GetBackendCompileCacheKey(graph_viewer);
        LOGS_DEFAULT(VERBOSE) << "Cache dir: " << cache_dir << ". Cache key: " << cache_key;
        // This way enforces the backend to use this cache dir and cache key.
        info_["cacheDir"] = cache_dir;
        info_["cacheKey"] = cache_key;
        fs::path backend_cache_file_loc(cache_dir + '/' + cache_key + "/context.json");
        if (!fs::exists(backend_cache_file_loc) || !fs::is_regular_file(backend_cache_file_loc) || fs::file_size(backend_cache_file_loc) == 0) {
          LOGS_DEFAULT(VERBOSE) << "Trying getting compilation cache from " << PathToUTF8String(ep_ctx_model_file_loc_);
          auto ep_ctx_payload = RetrieveEPContextCache(p_ep_ctx_model_->MainGraph(), ep_ctx_model_file_loc_, false);
          RestoreBackendCompileCache(backend_cache_file_loc, ep_ctx_payload);
        } else {
          LOGS_DEFAULT(VERBOSE) << "Backend compilation cache file already there " << backend_cache_file_loc.string();
        }
      }
    }
  } else {
    LOGS_DEFAULT(WARNING) << "Failed to get EP context model file location";
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
  if (ep_ctx_enabled_ && !is_ep_ctx_model) {
    PrepareEPContextEnablement(graph_viewer);
  }
  return result;
}
#endif

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
#if 0
  // ONLY uncomment this piece of code for the "Approach 3" mentioned above is switched ON.
  if (ep_ctx_enabled_ && p_ep_ctx_model_) {
    FulfillEPContextEnablement(fused_nodes_and_graphs);
  }
#endif
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
    const std::string& cache_key = info_.at("cacheKey");
    if (!cache_key.empty()) {
      LOGS_DEFAULT(VERBOSE) << "User configured cache key " << cache_key;
      return cache_key;
    }
  }
  // Model metadata key "vaip_model_md5sum".
  const auto& graph = graph_viewer.GetGraph();
  const auto& model_metadata = graph.GetModel().MetaData();
  if (model_metadata.count("vaip_model_md5sum") > 0) {
    const auto& cache_key = model_metadata.at("vaip_model_md5sum");
    if (!cache_key.empty()) {
      LOGS_DEFAULT(VERBOSE) << "Model metadata cache key " << cache_key;
      return cache_key;
    }
  }
  if (ParseEnvironmentVariableWithDefault<std::string>(
          "XLNX_ENABLE_FILE_BASED_CACHE_KEY", "0") != "0") {
    const Path& model_path = graph_viewer.ModelPath();
    if (!model_path.IsEmpty()) {
      LOGS_DEFAULT(VERBOSE) << "Model file MD5 cache key";
      return HashFileContentWithMD5(PathToUTF8String(model_path.ToPathString()));
    }
  }
  LOGS_DEFAULT(VERBOSE) << "Model signature cache key";
  return GetModelSignature(graph_viewer);
}

}  // namespace onnxruntime
